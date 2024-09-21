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
MOT dataset which returns image_id for evaluation.
"""

import json
import os
import cv2
import numpy as np
import torch
import torch.utils.data
import sys
import datasets.transforms as T
import random
import time
import copy

from PIL import Image
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from models.structures import Instances
from util.box_ops import box_iou
from util.misc import linear_assignment
from random import randint

class SportsMOTDataset(Dataset):
    vis_num = 3
    def __init__(self, args, seqs_folder, transform):
        self.args = args
        self.transform = transform
        self.num_frames_per_batch = max(args.sampler_lengths)
        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval
        self.video_dict = {}
        self.mot_path = os.path.join(args.mot_path, 'SportsMOT/dataset')
        self.file_name_width = 6

        self.labels_full = defaultdict(lambda : defaultdict(list))
        self.video_hw = {}
        def add_mot_folder(split_dir):
            print("Adding", split_dir)
            split = split_dir.split('/')[-1]
            for vid in os.listdir(os.path.join(self.mot_path, split_dir)):
                # vid = os.path.join(split_dir, vid)
                gt_path = os.path.join(self.mot_path, split_dir, vid, 'gt', 'gt.txt')
                first_img_path = os.path.join(self.mot_path, split_dir, vid, 'img1', f'{1:0{self.file_name_width}d}.jpg')
                w, h = Image.open(first_img_path).size
                self.video_hw[os.path.join(split, vid)] = (h, w)
                for l in open(gt_path):
                    # <frame> starts from 1, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, 1, 1, 1
                    t, i, *xywh, mark, label, visibility = l.strip().split(',')
                    t, i, mark, label = map(int, (t, i, mark, label))
                    visibility = float(visibility)
                    if visibility < 0.25:
                        continue
                    if mark == 0:
                        continue
                    if label in [3, 4, 5, 6, 9, 10, 11]:  # Non-person
                        continue
                    crowd = False
                    x, y, w, h = map(float, (xywh))
                    self.labels_full[os.path.join(split, vid)][t].append([x, y, w, h, i, crowd])

        # load gt
        add_mot_folder("train")
        vid_files = list(self.labels_full.keys())

        self.indices = []
        self.vid_tmax = {}
        clip_gap = args.clip_gap
        for vid in vid_files:
            self.video_dict[vid] = len(self.video_dict)
            t_min = min(self.labels_full[vid].keys())
            t_max = max(self.labels_full[vid].keys()) + 1
            self.vid_tmax[vid] = t_max - 1
            for t in range(t_min, t_max - self.num_frames_per_batch, clip_gap):
                self.indices.append((vid, t))
        print(f"Found {len(vid_files)} videos, {len(self.indices)} frames in SportsMOT")

        self.lengths: list = args.sampler_lengths
        print("sampler_lengths={}".format(self.lengths))

        print('loading detections...')
        if args.train_det_path:
            with open(os.path.join(args.mot_path, args.train_det_path)) as f:
                self.det_db = json.load(f)
        else:
            self.det_db = defaultdict(list)

    def parse_det_line(self, line):
        *box, s, c = map(float, line.split(','))
        return box, s
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.has_vised = 0

    def step_epoch(self):
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        is_gt = targets['gt_flag']
        gt_instances.boxes = targets['boxes'][is_gt]
        gt_instances.labels = targets['labels'][is_gt]
        gt_instances.obj_ids = targets['obj_ids'][is_gt]
        return gt_instances

    def _pre_single_frame(self, vid, idx: int):
        max_det = self.args.train_max_det_num
        img_path = os.path.join(self.mot_path, vid, 'img1', f'{idx:0{self.file_name_width}d}.jpg')
        targets = {}
        obj_idx_offset = self.video_dict[vid] * 100000  # 100000 unique ids is enough for a video.
        video_h, video_w = self.video_hw[vid]

        targets['dataset'] = 'SportsMOT'
        targets['boxes'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['scores'] = []
        gt_boxes, gt_scores, gt_track_ids, gt_labels = [], [], [], []
        det_boxes, det_scores, det_labels = [], [], []
        targets['image_id'] = torch.as_tensor(idx)
        for *box, id, crowd in self.labels_full[vid][idx]:
            # box: l, t, w, h -> x1, y1, x2, y2
            box[2] += box[0]
            box[3] += box[1]

            assert not crowd
            targets['iscrowd'].append(crowd)
            gt_boxes.append(box)
            gt_scores.append(1.)
            gt_track_ids.append(id + obj_idx_offset)
            gt_labels.append(0)
        txt_key = os.path.join(vid, 'img1', f'{idx:0{self.file_name_width}d}.txt')
        if '_half' in txt_key:
            txt_key = txt_key.replace('_half', '')
        for line in self.det_db[txt_key]:
            box, s = self.parse_det_line(line)
            # box: l, t, w, h -> x1, y1, x2, y2
            box[2] += box[0]
            box[3] += box[1]

            det_boxes.append(box)
            det_scores.append(s)
            det_labels.append(0)

        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        gt_scores = torch.as_tensor(gt_scores, dtype=torch.float32)
        gt_track_ids = torch.as_tensor(gt_track_ids, dtype=torch.float64)
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)
        det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32)
        det_scores = torch.as_tensor(det_scores, dtype=torch.float32)
        det_labels = torch.as_tensor(det_labels, dtype=torch.long)
        assert (gt_boxes[:, 2:] >= gt_boxes[:, :2]).all(), "boxes2: {}".format(gt_boxes)
        assert (det_boxes[:, 2:] >= det_boxes[:, :2]).all(), "boxes2: {}".format(det_boxes)

        gt_box_iou, _ = box_iou(gt_boxes, gt_boxes)
        gt_box_iou = gt_box_iou.fill_diagonal_(0)

        matched_boxes, matched_scores, matched_labels, matched_track_ids, det_unmached = self._match_det_gt(det_boxes, 
                                                                                                            det_scores, 
                                                                                                            det_labels, 
                                                                                                            gt_boxes, 
                                                                                                            gt_track_ids, 
                                                                                                            gt_labels,
                                                                                                            gt_scores)
        assert len(matched_boxes) <= max_det, f"too many boxes: {len(gt_boxes)}, max is {max_det}"
        if self.args.add_extra_dets:
            # add extra detections from detection results first
            extra_det_boxes, extra_det_scores, extra_det_labels, extra_det_iscrowd, extra_track_ids = self._gen_extra_dets(
                det_boxes, det_labels, det_scores, det_unmached, self.args.train_score_thresh)
            # only keep top k extra dets
            if len(extra_det_boxes)+len(matched_boxes) > max_det:
                print('warning: too many dets in image {} ({}), only keep top {}.'
                        .format(img_path, len(extra_det_boxes)+len(matched_boxes), max_det))
                extra_det_scores, indices = torch.topk(extra_det_scores, max_det-len(matched_boxes))
                extra_det_boxes = extra_det_boxes[indices]
                extra_det_labels = extra_det_labels[indices]
                extra_det_iscrowd = extra_det_iscrowd[indices]
                extra_track_ids = extra_track_ids[indices]
                
        else:
            extra_det_boxes = torch.empty((0, 4), dtype=torch.float32)
            extra_det_scores = torch.empty((0,), dtype=torch.float32)
            extra_det_labels = torch.empty((0,), dtype=torch.long)
            extra_det_iscrowd = torch.empty((0,), dtype=torch.bool)
            extra_track_ids = torch.empty((0,), dtype=torch.float32)
        
        assert len(extra_det_boxes)+len(matched_boxes) <= max_det, f"too many boxes: {len(extra_det_boxes)+len(matched_boxes)}, max is {max_det}"

        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])  # only has gt
        targets['labels'] = torch.cat((gt_labels, matched_labels, extra_det_labels), dim=0)
        targets['obj_ids'] = torch.cat((gt_track_ids, matched_track_ids, extra_track_ids), dim=0)
        targets['scores'] = torch.cat((gt_scores, matched_scores, extra_det_scores), dim=0)
        targets['boxes'] = torch.cat((gt_boxes, matched_boxes, extra_det_boxes), dim=0)
        gt_flag = torch.zeros((len(targets['scores']), ), dtype=torch.bool)
        gt_flag[:len(gt_labels)] = True
        targets['gt_flag'] = gt_flag
        return img_path, targets, gt_box_iou
    
    def _match_det_gt(self, det_boxes, det_scores, det_labels, gt_boxes, gt_obj_ids, gt_labels, gt_scores, amend_use_gt=False):
        assert gt_boxes.shape[0] != 0
        iou, union = box_iou(det_boxes, gt_boxes)
        matches, det_unmached, _ = linear_assignment((1.0-iou).numpy(), thresh=1-self.args.train_iou_thresh)
        if amend_use_gt:
            matched_obj_ids = gt_obj_ids.clone()
            matched_labels = gt_labels.clone()
            matched_boxes = gt_boxes.clone()
            matched_scores = gt_scores.clone()
            matched_scores[matches[:, 1]] = det_scores[matches[:, 0]]
            matched_boxes[matches[:, 1]] = det_boxes[matches[:, 0]]
        else:
            matched_boxes = det_boxes[matches[:, 0]]
            matched_scores = det_scores[matches[:, 0]]
            matched_labels = gt_labels[matches[:, 1]]
            matched_obj_ids = gt_obj_ids[matches[:, 1]]
        # assert matched_boxes.shape == gt_boxes.shape
        return matched_boxes, matched_scores, matched_labels, matched_obj_ids, det_unmached
    
    def _gen_extra_dets(self, det_boxes, det_labels, det_scores, det_unmached, score_thresh):
        # generate extra detections from unmatched detection results
        is_pos = det_scores[det_unmached] > score_thresh
        if is_pos.sum() == 0:
            det_boxes = torch.empty((0, 4), dtype=torch.float32)
            det_scores = torch.empty((0,), dtype=torch.float32)
            det_labels = torch.empty((0,), dtype=torch.int64)
            det_iscrowd = torch.empty((0,), dtype=torch.bool)
            det_idxes = torch.empty((0,), dtype=torch.float32)
            return det_boxes, det_scores, det_labels, det_iscrowd, det_idxes
        det_boxes = det_boxes[det_unmached][is_pos]
        det_scores = det_scores[det_unmached][is_pos]
        det_labels = det_labels[det_unmached][is_pos]
        det_iscrowd = torch.zeros_like(det_labels, dtype=torch.bool)
        det_idxes = torch.ones_like(det_labels, dtype=torch.float32) * (-2)
        return det_boxes, det_scores, det_labels, det_iscrowd, det_idxes

    def _get_sample_range(self, start_idx):
        # take default sampling method for normal dataset.
        assert self.sample_mode in ['fixed_interval', 'random_interval'], 'invalid sample mode: {}'.format(self.sample_mode)
        if self.sample_mode == 'fixed_interval':
            sample_interval = self.sample_interval
        elif self.sample_mode == 'random_interval':
            sample_interval = np.random.randint(1, self.sample_interval + 1)
        default_range = start_idx, start_idx + (self.num_frames_per_batch - 1) * sample_interval + 1, sample_interval
        return default_range

    def pre_continuous_frames(self, vid, indices):
        return zip(*[self._pre_single_frame(vid, i) for i in indices])

    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        tmax = self.vid_tmax[vid]
        if self.sample_mode == 'random_interval':
            rate = randint(1, self.sample_interval + 1)
            ids = [f_index + rate * i for i in range(self.num_frames_per_batch)]
        return [min(i, tmax) for i in ids]
    
    def vis_one_clip(self, ori_images, ori_targets):
        # vis gt, proposal respectively then save
        if self.has_vised < self.vis_num:
            def cxcywh_to_xyxy(box):
                box[0] -= box[2]/2
                box[1] -= box[3]/2
                box[2] += box[0]
                box[3] += box[1]
                return box
            def draw(img_i, box, obj_id, suffix: str):
                if 'gt' in suffix:
                    color = (0, 0, 255)
                elif 'det' in suffix:
                    color = (0, 255, 0)
                else:
                    raise ValueError('suffix should be gt or det')
                box *= np.array([w, h, w, h], dtype=np.float32)
                box = cxcywh_to_xyxy(box).astype(np.int32)
                img_i = cv2.rectangle(img_i, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(img_i, str(int(obj_id))+f'_{suffix}', (box[0], box[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                return img_i
            
            print('visualizing clip {}'.format(self.has_vised))
            images = copy.deepcopy(ori_images)
            targets = copy.deepcopy(ori_targets)
            frame = 0
            for img_i, targets_i in zip(images, targets):
                img_i = img_i.permute(1, 2, 0).cpu().numpy()
                # reverse normalize
                img_i *= np.array([0.229, 0.224, 0.225])
                img_i += np.array([0.485, 0.456, 0.406])
                img_i *= 255.0
                h, w, _ = img_i.shape
                img_i = cv2.UMat(img_i)
                img_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR)
                if 'gt_flag' in targets_i:  # not empty
                    n_gt = targets_i['gt_flag'].sum()
                    boxes_gt = targets_i['boxes'][:n_gt].numpy()
                    labels_gt = targets_i['labels'][:n_gt].numpy()
                    scores_gt = targets_i['scores'][:n_gt].numpy()
                    obj_ids_gt = targets_i['obj_ids'][:n_gt].numpy()
                    boxes_det = targets_i['boxes'][n_gt:].numpy()
                    labels_det = targets_i['labels'][n_gt:].numpy()
                    scores_det = targets_i['scores'][n_gt:].numpy()
                    obj_ids_det = targets_i['obj_ids'][n_gt:].numpy()
                    for box, label, score, obj_id in zip(boxes_det, labels_det, scores_det, obj_ids_det):
                        img_i = draw(img_i, box, obj_id, f'det {score:.2f}')
                    for box, label, score, obj_id in zip(boxes_gt, labels_gt, scores_gt, obj_ids_gt):
                        img_i = draw(img_i, box, obj_id, 'gt')
                cv2.imwrite('vis/epoch_{}_clip_{}_frame_{}.jpg'.format(self.current_epoch, self.has_vised, frame), img_i)
                frame += 1
            self.has_vised += 1

    def _generate_proposal(self, target: dict):
        is_det = ~target['gt_flag']
        obj_ids = target['obj_ids']
        gt_obj_ids = obj_ids[target['gt_flag']]
        # fiter out the dets that are not in gt because of random crop or other reasons
        for i, is_det_i in enumerate(is_det):
            if is_det_i and obj_ids[i] >=0 and obj_ids[i] not in gt_obj_ids:
                is_det[i] = False
        
        det_boxes = target['boxes'][is_det]
        det_scores = target['scores'][is_det][..., None]
        det_labels = target['labels'][is_det][..., None]
        det_idxes = target['obj_ids'][is_det][..., None].float()
        proposal = torch.cat([det_boxes, det_scores, det_labels, det_idxes], dim=1)
        # assert torch.all(det_idxes[:len(gt_obj_ids)].flatten() == gt_obj_ids)
        return proposal

    def __getitem__(self, idx):
        vid, f_index = self.indices[idx]
        img_paths, targets, _ = self.pre_continuous_frames(vid, self.sample_indices(vid, f_index))
        ori_images = []
        for img_path in img_paths:
            img = Image.open(img_path)
            w, h = img._size
            assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
            ori_images.append(img)
        if self.transform is not None:
            images, targets = self.transform(ori_images, targets)
        # self.vis_one_clip(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = self._targets_to_instances(targets_i, img_i.shape[1:3])
            gt_instances.append(gt_instances_i)
            n_gt = len(targets_i['iscrowd'])
            proposal = self._generate_proposal(targets_i)
            proposals.append(proposal)
        return {
            'imgs': images,
            'gt_instances': gt_instances,
            'ori_imgs': ori_images,
            'proposals': proposals,
        }

    def __len__(self):
        return len(self.indices)
            
class SportsMOTDatasetVal(SportsMOTDataset):
    def __init__(self, args, seqs_folder, transform):
        super().__init__(args, seqs_folder, transform)

def make_transforms_for_SportsMOT(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([800, 1000, 1200]),
                    T.FixedMotRandomCrop(800, 1200),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            T.MOTHSV(),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_transform(args, image_set):
    train = make_transforms_for_SportsMOT('train', args)
    test = make_transforms_for_SportsMOT('val', args)

    if image_set == 'train':
        return train
    elif image_set == 'val':
        return test
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    transform = build_transform(args, image_set)
    if image_set == 'train':
        dataset = SportsMOTDataset(args, seqs_folder=root, transform=transform)
    if image_set == 'val':
        dataset = SportsMOTDatasetVal(args, seqs_folder=root, transform=transform)
    return dataset