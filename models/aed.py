# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
model and criterion classes.
"""
import copy
import math
import datetime
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import List
from util import box_ops, checkpoint
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid,
                       linear_assignment)

from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .deformable_transformer_plus import build_deforamble_transformer, pos2posemb
from .query_updating import build as build_query_updating_layer
from .deformable_detr import SetCriterion, MLP, focal_loss
from .query_buffer import QueryBuffer
from .loss import MultiPosCrossEntropyLoss, L2Loss


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes,
                        matcher,
                        weight_dict,
                        losses,
                        match_thresh):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher  # abandoned
        self.weight_dict = weight_dict
        self.losses = losses
        self.match_thresh = match_thresh
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0

        self.mpce_loss = MultiPosCrossEntropyLoss()

    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}
        self.num_frames = len(gt_instances)

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float32, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss: str, outputs: dict):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'pt_weights': self.loss_pt_weights,
            'pp_weights': self.loss_pp_weights,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs)

    def loss_boxes(self, outputs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        gt_instances = outputs['gt_instances']
        indices = outputs['indices']
        num_boxes = outputs['num_boxes']
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses
    
    def loss_pt_weights(self, outputs):
        weights_cos = outputs['pt_weights_cos']
        weights_mm = outputs['pt_weights_mm']
        if weights_cos.numel() > 0:
            gt = torch.zeros_like(weights_cos)
            pt_matched_indices = outputs['pt_matched_indices']  # shape: [num_matched_proposals, num_tack_queries]
            gt[pt_matched_indices[:, 0], pt_matched_indices[:, 1]] = 1
            loss_1 = focal_loss(weights_cos,
                            gt,
                            alpha=1-1/weights_cos.shape[1],
                            gamma=2,
                            num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss_2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss_1 + loss_2
        else:
            loss = torch.tensor(0.0).to(self.sample_device)
        # if loss weight is nan
        # if torch.isnan(loss):
        #     raise ValueError("loss weight is nan")
        losses = {'loss_pt_weight': loss}
        return losses

    def loss_pp_weights(self, outputs):
        weights_cos = outputs['pp_weights_cos']
        weights_mm = outputs['pp_weights_mm']
        if weights_cos.numel() > 0:
            gt = torch.eye(weights_cos.shape[1], dtype=torch.float32, device=weights_cos.device)
            loss1 = focal_loss(weights_cos,
                              gt,
                              alpha=1-1/weights_cos.shape[1],
                              gamma=2,
                              num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss1 + loss2
        else:
            loss = torch.tensor(0.0).to(self.sample_device)
        # if loss weight is nan
        # if torch.isnan(loss):
        #     raise ValueError("loss weight is nan")
        losses = {'loss_pp_weight': loss}
        return losses
    
    def loss_cross_clip(self, weights_cos, weights_mm, gt):
        loss = torch.tensor(0.0).to(self.sample_device)
        if weights_cos.numel() > 0:
            loss1 = focal_loss(weights_cos,
                              gt,
                              alpha=1-1/weights_cos.shape[1],
                              gamma=2,
                              num_boxes=weights_cos.shape[0], mean_in_dim0=False)
            loss2 = self.mpce_loss(weights_mm, gt, (gt.sum(dim=1) > 0).float())
            loss = loss1 + loss2
        else:
            loss = torch.tensor(0.0).to(weights_cos)
        self.losses_dict.update({'weight_loss_cross_clip': loss})
    
    def match_for_single_frame(self, outputs: dict, num_proposals: int):
        gt_instances_i = self.gt_instances[self._current_frame_idx]  # gt instances of i-th image.
        track_instances: Instances = outputs['track_instances']
        track_instances.matched_gt_ids[:] = -1
        pt_weights_cos = outputs['pt_weights_cos'][0]
        pt_weights_mm = outputs['pt_weights_mm'][0]

        # filter out the extra dets (gt_ids == -2) in proposals
        is_valid = track_instances.gt_ids[:num_proposals] >= -1
        valid_proposal_instances = track_instances[:num_proposals][is_valid]
        invalid_proposal_instances = track_instances[:num_proposals][~is_valid]

        track_query_instances = track_instances[num_proposals:]
        proposal_instances = track_instances[:num_proposals]

        # step1. match proposals and gts
        i, j = torch.where(valid_proposal_instances.gt_ids[:, None] == gt_instances_i.obj_ids)
        matched_indices = torch.stack([i, j], dim=1).to(pt_weights_cos.device)
        valid_proposal_instances.matched_gt_ids[i] = j

        # step2. inherit id and calculate iou
        valid_proposal_instances.obj_ids[matched_indices[:, 0]] = gt_instances_i.obj_ids[matched_indices[:, 1]].long()
        valid_proposal_instances.matched_gt_ids[matched_indices[:, 0]] = matched_indices[:, 1]
        assert torch.all(valid_proposal_instances.obj_ids >= 0) and torch.all(valid_proposal_instances.matched_gt_ids >= 0), \
        "matched gt ids should be >= 0, get {} and {}".format(valid_proposal_instances.obj_ids, valid_proposal_instances.matched_gt_ids)
        active_idxes = (valid_proposal_instances.obj_ids >= 0) & (valid_proposal_instances.matched_gt_ids >= 0)
        active_track_boxes = valid_proposal_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_i.boxes[valid_proposal_instances.matched_gt_ids[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            valid_proposal_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step3. remove valid_proposal_instances with iou < 0.5 (ambiguous) & update valid
        neg_valid_proposal_mask = valid_proposal_instances.iou < 0.5
        neg_is_valid_idx = torch.where(is_valid)[0][neg_valid_proposal_mask]
        is_valid_mask = torch.ones_like(is_valid, dtype=torch.bool)  # is_valid and iou < 0.5, shape: [num_proposals]
        is_valid_mask[neg_is_valid_idx] = False
        new_is_valid = is_valid[is_valid_mask]
        new_valid_proposal_instances = valid_proposal_instances[~neg_valid_proposal_mask]
        pt_weights_cos, pt_weights_mm = pt_weights_cos[is_valid_mask], pt_weights_mm[is_valid_mask]
        valid_pt_weights_cos = pt_weights_cos[new_is_valid]  # shape: [num_valid_proposals, num_track_queries]
        valid_pt_weights_mm = pt_weights_mm[new_is_valid]  # shape: [num_valid_proposals, num_track_queries]

        # step4. match proposals and track queries
        i, j = torch.where(new_valid_proposal_instances.gt_ids[:, None] == track_query_instances.gt_ids)
        pt_matched_indices = torch.stack([i, j], dim=1).to(pt_weights_cos.device)

        with torch.no_grad():
            for (valid_p_idx, t_idx) in pt_matched_indices:  # matching for valid proposals (not newly apprear)
                similarity = valid_pt_weights_cos[valid_p_idx, t_idx]
                if similarity < 1 - self.match_thresh:
                    new_valid_proposal_instances.obj_ids[valid_p_idx] = -1
                    new_valid_proposal_instances.gt_ids[valid_p_idx] = -2
                else:
                    track_query_instances.obj_ids[t_idx] = -1
                    new_valid_proposal_instances.matched_track_embedding[valid_p_idx] = track_query_instances.query_pos[t_idx]
                    new_valid_proposal_instances.new[valid_p_idx] = False
            
            pt_weight_np = pt_weights_cos.detach().clone().cpu().numpy().astype('float32')
            matches, unmatched_ps, unmatched_ts = linear_assignment(1-pt_weight_np, thresh=self.match_thresh)
            for invalid_p_idx in torch.where(~new_is_valid)[0].cpu().numpy():  # matching for invalid proposals
                assert invalid_p_idx not in pt_matched_indices.cpu().numpy()[:, 0]
                p_idx = (~new_is_valid)[:invalid_p_idx].sum()  # idx in invalid_proposal_instances, different from invalid_p_idx
                if invalid_p_idx in matches[:, 0]:  # matched invalid proposals
                    t_idx = matches[matches[:, 0] == invalid_p_idx][0][1]
                    assert new_is_valid[invalid_p_idx] == False
                    if track_query_instances.gt_ids[t_idx] >= 0:  # mistakenly matched with a valid track query
                        invalid_proposal_instances.gt_ids[p_idx] = -3
                    else:  # matched successfully
                        track_query_instances.gt_ids[t_idx] = -3
                        invalid_proposal_instances.matched_track_embedding[p_idx] = track_query_instances.query_pos[t_idx]
                        invalid_proposal_instances.new[p_idx] = False
                else:  # unmatched invalid proposals
                    invalid_proposal_instances.new[p_idx] = True

        # step5. calculate losses.
        num_boxes = max(len(valid_proposal_instances.pred_boxes), 1)
        outputs_last = {
                        'pred_boxes': valid_proposal_instances.pred_boxes.unsqueeze(0),
                        'indices': [(matched_indices[:, 0], matched_indices[:, 1])],
                        'gt_instances': [gt_instances_i],
                        'pt_weights_cos': valid_pt_weights_cos,
                        'pt_weights_mm': valid_pt_weights_mm,
                        'pt_matched_indices': pt_matched_indices,
                        'pp_weights_cos': outputs['pp_weights_cos'][0],
                        'pp_weights_mm': outputs['pp_weights_mm'][0],
                        'num_boxes': num_boxes,
                        }
        
        self.sample_device = pt_weights_cos.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_last)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                outputs_layer = {
                    'pred_boxes': aux_outputs['pred_boxes'][0, is_valid].unsqueeze(0),
                    'indices': [(matched_indices[:, 0], matched_indices[:, 1])],
                    'gt_instances': [gt_instances_i],
                    'pt_weights_cos': aux_outputs['pt_weights_cos'][0, is_valid_mask][new_is_valid],
                    'pt_weights_mm': aux_outputs['pt_weights_mm'][0, is_valid_mask][new_is_valid],
                    'pt_matched_indices': pt_matched_indices,
                    'pp_weights_cos': aux_outputs['pp_weights_cos'][0],
                    'pp_weights_mm': aux_outputs['pp_weights_mm'][0],
                    'num_boxes': num_boxes,
                }
                
                for loss in self.losses:
                    l_dict = self.get_loss(loss,
                                           outputs=outputs_layer)
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})
        self._step()
        return Instances.cat([new_valid_proposal_instances, invalid_proposal_instances, track_query_instances])

    def forward(self, outputs, input_data: dict):
        # losses of each frame are calculated during the model's forwarding 
        # and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        for loss_name, loss in losses.items():
            losses[loss_name] /= self.num_frames
        return losses


class RuntimeTrackerBase(object):
    def __init__(self, val_match_high_thresh, val_match_low_thresh, miss_tolerance, match_high_score):
        self.val_match_high_thresh = val_match_high_thresh
        self.val_match_low_thresh = val_match_low_thresh
        self.miss_tolerance = miss_tolerance
        self.match_high_score = match_high_score
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances, weights):  # two stage
        # weight: [num_proposals, track_queries]
        num_proposals = weights.shape[0]
        proposal_instances = track_instances[:num_proposals]
        # split high and low proposals by det score
        high_score_mask = proposal_instances.det_scores > self.match_high_score
        low_score_mask = ~high_score_mask
        high_proposal_instances = proposal_instances[high_score_mask]
        low_proposal_instances = proposal_instances[low_score_mask]
        # get track queries
        track_query_instances = track_instances[num_proposals:]
        device = proposal_instances.obj_ids.device
        assert torch.all(proposal_instances.disappear_time==0)

        # get high score proposals-track queries weights
        high_weights = weights[high_score_mask].cpu().numpy().astype('float32')
        # matching
        high_matches, high_unmatched_p, high_unmatched_t = linear_assignment(1-high_weights, thresh=self.val_match_high_thresh)
        # update
        high_proposal_instances.obj_ids[high_matches[:, 0]] = track_query_instances.obj_ids[high_matches[:, 1]]
        high_proposal_instances.new[high_matches[:, 0]] = False
        high_proposal_instances.matched_track_embedding[high_matches[:, 0]] = track_query_instances.query_pos[high_matches[:, 1]]
        # delet the matched track queries
        track_query_instances.obj_ids[high_matches[:, 1]] = -1

        # get low score proposals-track queries weights
        low_weights = weights[low_score_mask][:, high_unmatched_t].cpu().numpy().astype('float32')
        # matching
        low_matches, low_unmatched_p, low_unmatched_t = linear_assignment(1-low_weights, thresh=self.val_match_low_thresh)
        # update
        low_proposal_instances.obj_ids[low_matches[:, 0]] = track_query_instances.obj_ids[high_unmatched_t[low_matches[:, 1]]]
        low_proposal_instances.new[low_matches[:, 0]] = False
        low_proposal_instances.matched_track_embedding[low_matches[:, 0]] = track_query_instances.query_pos[high_unmatched_t[low_matches[:, 1]]]
        # delet the matched track queries
        track_query_instances.obj_ids[high_unmatched_t[low_matches[:, 1]]] = -1
        
        # assign id for hight new proposals
        num_new_objs = high_unmatched_p.size
        high_proposal_instances.obj_ids[high_unmatched_p] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        # delete low new proposals
        low_proposal_instances.obj_ids[low_unmatched_p] = -1  # init is -1, so this is not necessary?

        # update disappear time
        track_query_instances.disappear_time[high_unmatched_t[low_unmatched_t]] += 1

        # delete
        to_del = track_query_instances.disappear_time >= self.miss_tolerance
        track_query_instances.obj_ids[to_del] = -1

        return Instances.cat([high_proposal_instances, low_proposal_instances, track_query_instances])


class TrackerPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size) -> Instances:
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_bbox = track_instances.pred_boxes

        if len(out_bbox) != 0:
            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
            # clip boxes
            boxes = boxes.clamp(min=0, max=1)
            # and from relative [0, 1] to absolute [0, height] coordinates
            img_h, img_w = target_size
            scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes)
            boxes = boxes * scale_fct[None, :]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=out_bbox.device)

        track_instances.boxes = boxes
        return track_instances


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class AED(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_feature_levels, criterion, track_embed,
                 aux_loss=True, with_box_refine=False, two_stage=False, buffer=None, use_checkpoint=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes, for AED it is always 1
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR, not used in AED
        """
        super().__init__()
        self.num_clip = 0
        self.debug_file_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.track_embed = track_embed
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes # not used in AED
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.det_embed = nn.Embedding(1, hidden_dim)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            self.bbox_embed = None
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.post_process = TrackerPostProcess()
        self.track_base = None  # will be created during testing
        self.criterion = criterion
        self.buffer = buffer
        self.weight_attn = self.transformer.decoder.layers[-1].weight_attn

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def _generate_empty_tracks(self, proposals=None):
        track_instances = Instances((1, 1))
        _, d_model = self.det_embed.weight.shape
        if proposals is None:
            track_instances.ref_pts = torch.empty((0, 4), dtype=torch.float32)
            track_instances.query_pos = torch.empty((0, d_model), dtype=torch.float32)  # query
            track_instances.labels = torch.empty((0), dtype=torch.long)
            track_instances.det_scores = torch.empty((0), dtype=torch.float32)
        else:
            track_instances.ref_pts = proposals[:, :4]  # [xc, yc, w, h]
            track_instances.query_pos = pos2posemb(proposals[:, 4:5], d_model) + self.det_embed.weight  # query
            track_instances.labels = proposals[:, 5].long()
            track_instances.det_scores = proposals[:, 4].float()
        track_instances.output_embedding = torch.zeros((len(track_instances), d_model))
        track_instances.obj_ids = torch.full((len(track_instances),), -1, dtype=torch.long)
        track_instances.matched_gt_ids = torch.full((len(track_instances),), -1, dtype=torch.long)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long)
        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float32)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float32)
        track_instances.save_period = torch.zeros((len(track_instances), ), dtype=torch.float32)
        track_instances.matched_track_embedding = torch.zeros((len(track_instances), d_model), dtype=torch.float32)
        
        track_instances.new = torch.ones((len(track_instances),), dtype=torch.bool)  # bool is new or not
        if self.training:
            if proposals is None:
                track_instances.gt_ids = torch.empty((0), dtype=torch.long)
            else:
                # gt_ids = -3: invalid extra dets (need to be removed)
                # gt_ids = -2: valid extra dets (e.g. false positives or unlabeled dets)
                # gt_ids = -1: untracked
                # gt_ids >= 0: tracked
                track_instances.gt_ids = proposals[:, 6].long()
        return track_instances.to(self.det_embed.weight.device)

    def clear(self):
        if not self.training:
            self.track_base.clear()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, pt_weights_cos, pp_weights_cos, pt_weights_mm, pp_weights_mm):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': a, 'pt_weights_cos': b, 'pp_weights_cos': c, 'pt_weights_mm': d, 'pp_weights_mm': e}
                for a, b, c, d, e in zip(outputs_coord[:-1], pt_weights_cos[:-1], pp_weights_cos[:-1], pt_weights_mm[:-1], pp_weights_mm[:-1])]

    def _forward_single_image(self, samples, track_instances: Instances, num_proposals: int, gtboxes=None):
        features, pos = self.backbone(samples)  # extract features and create position embeddings
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):  # project the features to a common feature channel (256)
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):  # add an extra level
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if gtboxes is not None:
            raise ValueError("gtboxes is not None")
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        p_hs, init_p_ref_pts, inter_p_ref_pts, pt_weights_cos, pp_weights_cos, pt_weights_mm, pp_weights_mm = \
            self.transformer(srcs, masks, pos, num_proposals, query_embed, ref_pts=ref_pts, attn_mask=attn_mask)

        outputs_coords = []
        for lvl in range(p_hs.shape[0]):
            if lvl == 0:
                reference = init_p_ref_pts
            else:
                reference = inter_p_ref_pts[lvl - 1]
            if self.bbox_embed is not None:
                reference = inverse_sigmoid(reference)
                # box refinement
                tmp = self.bbox_embed[lvl](p_hs[lvl])
                if reference.shape[-1] == 4:
                    tmp += reference
                else:
                    assert reference.shape[-1] == 2
                    tmp[..., :2] += reference
                outputs_coord = tmp.sigmoid()
            else:
                outputs_coord = reference
            outputs_coords.append(outputs_coord)

        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_boxes': outputs_coord[-1], 'pt_weights_cos': pt_weights_cos[-1], 'pp_weights_cos': pp_weights_cos[-1],
               'pt_weights_mm': pt_weights_mm[-1], 'pp_weights_mm': pp_weights_mm[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_coord, pt_weights_cos, pp_weights_cos, pt_weights_mm, pp_weights_mm)
        out['hs'] = p_hs[-1]
        return out

    def _post_process_single_image(self, frame_res, track_instances, num_proposals, is_last):
        track_instances.pred_boxes[:num_proposals] = frame_res['pred_boxes'][0]
        track_instances.output_embedding[:num_proposals] = frame_res['hs'][0, :num_proposals]
        # matched_track_embedding inits as its own embedding
        track_instances.matched_track_embedding[:num_proposals] = frame_res['hs'][0, :num_proposals]
        
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            # only proposal_instances are keeped
            track_instances = self.criterion.match_for_single_frame(frame_res, num_proposals)
            # track_instances = track_instances[:num_proposals]
            if self.buffer is not None:
                self.buffer(track_instances[:num_proposals])
                if is_last:
                    weights_ids = []
                    embeddings = torch.empty((0, self.det_embed.weight.shape[1]), dtype=torch.float32, device=track_instances.query_pos.device)
                    ids = torch.empty((0, ), dtype=torch.long, device=track_instances.query_pos.device)
                    for id, tmp in self.buffer.memory.items():
                        embeddings = torch.cat([embeddings, tmp['embeddings']], dim=0)
                        ids = torch.cat([ids, torch.full((tmp['embeddings'].shape[0],), id, dtype=torch.long, device=track_instances.query_pos.device)], dim=0)
                    q = embeddings.clone()
                    k = embeddings.clone()
                    gt = (ids.unsqueeze(0) == ids.unsqueeze(1)).float().to(q.device)
                    weights_cos, weights_mm = self.weight_attn(q[None, :], k[None, :])
                    self.criterion.loss_cross_clip(weights_cos[0], weights_mm[0], gt)
        else:
            # each track will be assigned an unique global id by the track base.
            pt_weights = frame_res['pt_weights_cos']
            track_instances = self.track_base.update(track_instances, pt_weights[0])
            
            # for weight visualization
            track_instances.gt_ids = track_instances.obj_ids.clone()
            if self.buffer is not None:
                self.buffer(track_instances[:num_proposals])
                if is_last:
                    weights_ids = {}
                    embeddings = torch.empty((0, self.det_embed.weight.shape[1]), dtype=torch.float32, device=track_instances.query_pos.device)
                    for id, tmp in self.buffer.memory.items():
                        embeddings = torch.cat([embeddings, tmp['embeddings']], dim=0)
                    total_q = embeddings.clone()
                    total_k = embeddings.clone()
                    weights_cos, _ = self.weight_attn(total_q[None, :], total_k[None, :])
                    weights_ids['total'] = weights_cos[0]
                    frame_res['cross_clip_weight'] = weights_ids

        tmp = {}
        tmp['track_instances'] = track_instances
        out_track_instances, num_active_proposals, active_idxes = self.track_embed(tmp, num_proposals)
        frame_res['track_instances'] = out_track_instances
        frame_res['active_idxes'] = active_idxes

        return frame_res, num_active_proposals

    @torch.no_grad()
    def inference_single_image(self, img, ori_img_size, num_proposals, track_instances=None, proposals=None, is_last=False):
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensor_list(img)
        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])
        
        res = self._forward_single_image(img,
                                         track_instances=track_instances,
                                         num_proposals=num_proposals)
        res, num_active_proposals = self._post_process_single_image(res, track_instances, num_proposals, is_last)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances, 'num_active_proposals': num_active_proposals, 'res': res}
        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.Tensor([img_w, img_h]).to(ref_pts)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts
        return ret

    def forward(self, data: dict):
        self.num_clip += 1
        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
        frames = data['imgs']
        outputs = {
            'pred_boxes': [],
        }
        track_instances = None
        keys = list(self._generate_empty_tracks()._fields.keys())
        if self.buffer is not None:
            self.buffer.clear()
        for frame_index, (frame, gt, proposals, ori_img) in enumerate(zip(frames, data['gt_instances'], data['proposals'], data['ori_imgs'])):
            frame.requires_grad = False
            is_last = frame_index == len(frames) - 1
            num_proposals = len(proposals) if proposals is not None else 0
            gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_checkpoint and frame_index < len(frames) - 1:
                def fn(frame, gtboxes, num_proposals, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._forward_single_image(frame, tmp, num_proposals, gtboxes)
                    return (
                        frame_res['pred_boxes'],
                        frame_res['hs'],
                        frame_res['pt_weights_cos'],
                        frame_res['pp_weights_cos'],
                        frame_res['pt_weights_mm'],
                        frame_res['pp_weights_mm'],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']],
                        *[aux['pt_weights_cos'] for aux in frame_res['aux_outputs']],
                        *[aux['pp_weights_cos'] for aux in frame_res['aux_outputs']],
                        *[aux['pt_weights_mm'] for aux in frame_res['aux_outputs']],
                        *[aux['pp_weights_mm'] for aux in frame_res['aux_outputs']],
                    )

                num_proposals = torch.tensor(num_proposals, dtype=torch.long, device=frame.device)
                args = [frame, gtboxes, num_proposals] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_boxes': tmp[0],
                    'hs': tmp[1],
                    'pt_weights_cos': tmp[2],
                    'pp_weights_cos': tmp[3],
                    'pt_weights_mm': tmp[4],
                    'pp_weights_mm': tmp[5],
                    'aux_outputs': [{
                        'pred_boxes': tmp[6+i],
                        'pt_weights_cos': tmp[6+5+i],
                        'pp_weights_cos': tmp[6+10+i],
                        'pt_weights_mm': tmp[6+15+i],
                        'pp_weights_mm': tmp[6+20+i],
                    } for i in range(5)],  # 5 = dec_layers - 1, if dec_layers is changed, this should be changed
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame, track_instances, num_proposals, gtboxes)
            frame_res, _ = self._post_process_single_image(frame_res, track_instances, int(num_proposals), is_last)

            track_instances = frame_res['track_instances']
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs


def build(args):
    num_classes = 1
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)
    d_model = transformer.d_model
    hidden_dim = args.dim_feedforward
    query_updating_layer = build_query_updating_layer(args)

    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            'frame_{}_loss_pt_weight'.format(i): args.pt_weight_loss_coef,
                            'frame_{}_loss_pp_weight'.format(i): args.pp_weight_loss_coef,
                            })

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_aux{}_loss_pt_weight'.format(i, j): args.pt_weight_loss_coef,
                                    'frame_{}_aux{}_loss_pp_weight'.format(i, j): args.pp_weight_loss_coef,
                                    })
            for j in range(args.dec_layers):
                weight_dict.update({'frame_{}_ps{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_ps{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    'frame_{}_ps{}_loss_pt_weight'.format(i, j): args.pt_weight_loss_coef,
                                    'frame_{}_ps{}_loss_pp_weight'.format(i, j): args.pp_weight_loss_coef,
                                    })
    # buffer = None
    buffer = QueryBuffer()
    if buffer is not None:
        weight_dict.update({'weight_loss_cross_clip': args.cross_clip_weight_loss_coef})
    losses = ['pt_weights', 'pp_weights']
    if args.with_box_refine:
        losses += ['boxes']
    criterion = ClipMatcher(num_classes, matcher=None, weight_dict=weight_dict, losses=losses, match_thresh=args.train_match_thresh)
    criterion.to(device)
    postprocessors = {}
    model = AED(
        backbone=backbone,
        transformer=transformer,
        track_embed=query_updating_layer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        buffer=buffer,
        use_checkpoint=args.use_checkpoint,
    )
    return model, criterion, postprocessors