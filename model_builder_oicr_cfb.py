from functools import wraps
import importlib
import logging
import queue
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.pcl.pcl import PCL
from model.pcl.oicr_momentum import OICR, OICRLosses, FGIM
from model.pcl.oicr_bs3 import _smooth_l1_loss
from model.regression.bbox_transform import bbox_transform_inv
from model.pcl_losses.functions.pcl_losses import PCLLosses
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.pcl_heads as pcl_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.boxes as box_utils
import utils.vgg_weights_helper as vgg_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
            self.Conv_Body.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale, return_fc6=True)
        self.Box_MIL_Outs = pcl_heads.mil_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES)
        self.Box_Refine_Outs = pcl_heads.refine_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)
        self.RCNN_Cls_Reg = pcl_heads.cls_regress_outputs(
            self.Box_Head.dim_out, cfg.MODEL.NUM_CLASSES + 1)

        self.Cls_Loss = OICRLosses()

        self.Refine_Losses = [OICRLosses() for i in range(cfg.REFINE_TIMES)]

        self.queue_num_classes = cfg.MODEL.NUM_CLASSES
        self.cls_feats = [[] for i in range(self.queue_num_classes)]
        self.cls_feats_scores = [[] for i in range(self.queue_num_classes)]
        self.cls_feats_iters = [[] for i in range(self.queue_num_classes)]

        self.queue_lenth = cfg.OICR.FEAT_Queue_Lenth
        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            vgg_utils.load_pretrained_imagenet_weights(self)

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def update_feature(self, pcl_output, step):
        '''
        step: current step
        pcl_output: dict
        pcl_output['box_feats']: dict, {cls: box_feat}
        pcl_output['box_feats_ind']: dict, {cls: corresponding key}
        pcl_output['box_feats_scores']: dict, {cls: corresponding scores}
        '''
        for k, v in pcl_output['box_feats'].items():
            ind = pcl_output['box_feats_ind'][k]
            if ind == -1:
                if len(self.cls_feats[k]) == self.queue_lenth:
                    iters = np.array(self.cls_feats_iters[k])
                    ind_ori = np.argmin(iters)
                    self.cls_feats[k].pop(ind_ori)
                    self.cls_feats[k].append(v)
                    self.cls_feats_scores[k].pop(ind_ori)
                    self.cls_feats_scores[k].append(pcl_output['box_feats_scores'][k])
                    self.cls_feats_iters[k].pop(ind_ori)
                    self.cls_feats_iters[k].append(step)
                else:
                    self.cls_feats[k].append(v)
                    self.cls_feats_scores[k].append(pcl_output['box_feats_scores'][k])
                    self.cls_feats_iters[k].append(step)
            else:
                feat_cur = v
                score_cur = pcl_output['box_feats_scores'][k]
                feat_ori = self.cls_feats[k][ind]
                score_ori = self.cls_feats_scores[k][ind]
                ratio = score_cur / (score_cur + score_ori + 1e-4)
                feat_new = ratio * feat_cur + (1 - ratio) * feat_ori
                score_new = ratio * score_cur + (1 - ratio) * score_ori
                self.cls_feats[k][ind] = feat_new
                self.cls_feats_scores[k][ind] = score_new
                self.cls_feats_iters[k][ind] = step

    def update_feature_fifo(self, pcl_output, step):
        '''
        pcl_output: dict
        pcl_output['box_feats']: dict, {cls: box_feat}
        pcl_output['box_feats_scores']: dict, {cls: corresponding scores}
        '''
        for k, v in pcl_output['box_feats'].items():
            if len(self.cls_feats[k]) == self.queue_lenth:
                self.cls_feats[k].pop(0)
                self.cls_feats[k].append(v)
                self.cls_feats_scores[k].pop(0)
                self.cls_feats_scores[k].append(pcl_output['box_feats_scores'][k])
                self.cls_feats_iters[k].pop(0)
                self.cls_feats_iters[k].append(step)
            else:
                self.cls_feats[k].append(v)
                self.cls_feats_scores[k].append(pcl_output['box_feats_scores'][k])
                self.cls_feats_iters[k].append(step)



    def forward(self, data, rois, labels, step=0):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, rois, labels, step)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, rois, labels, step)

    def _forward(self, data, rois, labels, step=0):
        im_data = data
        if self.training:
            rois = rois.squeeze(dim=0).type(im_data.dtype)
            labels = labels.squeeze(dim=0).type(im_data.dtype)

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        blob_conv = self.Conv_Body(im_data)

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        box_feat, box_feat_fc6 = self.Box_Head(blob_conv, rois)
        mil_score = self.Box_MIL_Outs(box_feat)
        refine_score = self.Box_Refine_Outs(box_feat)
        cls_score, bbox_pred = self.RCNN_Cls_Reg(box_feat)

        if self.training:
            return_dict['losses'] = {}

            # image classification loss
            im_cls_score = mil_score.sum(dim=0, keepdim=True)
            loss_im_cls = pcl_heads.mil_losses(im_cls_score, labels)
            return_dict['losses']['loss_im_cls'] = loss_im_cls

            # refinement loss
            boxes = rois.data.cpu().numpy()
            im_labels = labels.data.cpu().numpy()
            boxes = boxes[:, 1:]

            for i_refine, refine in enumerate(refine_score):
                # [Branch 0 represents FGIM]
                if i_refine == 0:
                    pcl_output = FGIM(boxes, mil_score, im_labels, refine, box_feat, self.cls_feats)

                    update_type = cfg.OICR.FEAT_Update_Type # [1: Weighting Strategy; 0: FIFO]
                    if update_type == 1:
                        self.update_feature(pcl_output, step)
                    else:
                        self.update_feature_fifo(pcl_output, step)
                # [Other branches represent OIR]
                else:
                    pcl_output = OICR(boxes, refine_score[i_refine - 1],
                                        im_labels, refine)
                refine_loss = self.Refine_Losses[i_refine](refine, pcl_output['labels'],
                                                    pcl_output['cls_loss_weights'])

                return_dict['losses']['refine_loss%d' % i_refine] = refine_loss.clone()
            
            # regression
            rois_label = pcl_output['rois_labels']
            rois_target = pcl_output['bbox_targets']
            rois_inside_ws = pcl_output['bbox_inside_weights']
            cls_loss_ws = pcl_output['cls_loss_weights']

            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

            RCNN_loss_cls = self.Cls_Loss(cls_score, pcl_output['labels'], cls_loss_ws)
            return_dict['losses']['cls_loss'] = RCNN_loss_cls.clone()

            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, cls_loss_ws)
            return_dict['losses']['reg_loss'] = RCNN_loss_bbox.clone()

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rois
            return_dict['mil_score'] = mil_score
            return_dict['refine_score'] = refine_score
            return_dict['refine_score'].append(cls_score)
            # decode regression
            rois = rois[:, 1:]
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                box_deltas = box_deltas.view(-1, 4 * (cfg.MODEL.NUM_CLASSES + 1))
            pred_boxes = bbox_transform_inv(rois, box_deltas, 1)
            im_shape = data.shape[-2:]
            pred_boxes = box_utils.clip_boxes_2(pred_boxes, im_shape)
            return_dict['rois'] = pred_boxes

        return return_dict

    def roi_feature_transform(self, blobs_in, rois, method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        # Single feature level
        # rois: holds R regions of interest, each is a 5-tuple
        # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
        # rectangle (x1, y1, x2, y2)
        if method == 'RoIPoolF':
            xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
        elif method == 'RoICrop':
            grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
            xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                xform_out = F.max_pool2d(xform_out, 2, 2)
        elif method == 'RoIAlign':
            xform_out = RoIAlignFunction(
                resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        return blob_conv

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value