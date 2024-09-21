import math
import torch
from torch import nn

from util import box_ops
from models.structures import Boxes, Instances, pairwise_iou

class QueryUpdating(nn.Module):
    def __init__(self, args, dim_in, hidden_dim, dim_out, max_track_num=200, ema_weight=0.9, dropout=0):
        super().__init__()
        self.random_drop = args.random_drop
        self.max_track_num = max_track_num
        self.ema_weight = ema_weight

    def _random_drop_proposals(self, track_instances: Instances, num_proposals) -> Instances:
        if self.random_drop > 0 and num_proposals > 0:
            keep_idxes = torch.ones_like(track_instances.new_score, dtype=torch.bool)
            tmp = torch.rand_like(track_instances.new_score[:num_proposals]) > self.random_drop
            new_num_proposals = tmp.sum()
            keep_idxes[:num_proposals] = tmp
            track_instances = track_instances[keep_idxes]
            return track_instances, new_num_proposals
        else:
            return track_instances, num_proposals

    def _select_active_tracks(self, data: dict, num_proposals) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_ids >= 0) | (track_instances.gt_ids == -2)
            num_active_proposals = active_idxes[:num_proposals].sum()
            active_track_instances = track_instances[active_idxes]
            active_track_instances, num_active_proposals = self._random_drop_proposals(active_track_instances, num_active_proposals)

        else:
            active_idxes = track_instances.obj_ids >= 0
            num_active_proposals = active_idxes[:num_proposals].sum()
            active_track_instances = track_instances[active_idxes]

        return active_track_instances, num_active_proposals, active_idxes

    def _update_track_embedding(self, track_instances: Instances, num_proposals) -> Instances:
        p_output_embeddings = track_instances.output_embedding[:num_proposals]
        track_instances.query_pos[:num_proposals] = p_output_embeddings

        track_instances.ref_pts[:num_proposals] = track_instances.pred_boxes.detach().clone()[:num_proposals]
        return track_instances

    def forward(self, data, num_proposals) -> Instances:
        active_track_instances, num_active_proposals, active_idxes = self._select_active_tracks(data, num_proposals)
        active_track_instances = self._update_track_embedding(active_track_instances, num_active_proposals)
        return active_track_instances, num_active_proposals, active_idxes
    
class QueryUpdatingEMA(QueryUpdating):
    def __init__(self, args, dim_in, hidden_dim, dim_out, max_track_num=200, ema_weight=0.9, dropout=0):
        super().__init__(args, dim_in, hidden_dim, dim_out, max_track_num, ema_weight, dropout)

    def _update_track_embedding(self, track_instances: Instances, num_proposals) -> Instances:
        p_output_embeddings = track_instances.output_embedding[:num_proposals]
        is_old = track_instances.new[:num_proposals] == False
        p_matched_track_embeddings = track_instances.matched_track_embedding[:num_proposals]
        p_output_embeddings[is_old] = self.ema_weight * p_output_embeddings[is_old] + (1-self.ema_weight) * p_matched_track_embeddings[is_old]

        assert torch.all(p_matched_track_embeddings[is_old]==0, dim=1).sum() == 0

        track_instances.query_pos[:num_proposals] = p_output_embeddings

        track_instances.ref_pts[:num_proposals] = track_instances.pred_boxes.detach().clone()[:num_proposals]
        return track_instances


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


def build(args, dim_in, hidden_dim, dim_out):
    return QueryUpdatingEMA(args, dim_in, hidden_dim, dim_out, args.max_track_num, ema_weight=args.ema_weight)
