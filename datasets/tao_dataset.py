import torch
import copy
import sys
import cv2
import random
import os.path as osp
import numpy as np
import json
import datasets.transforms as T

from tao.toolkit.tao import Tao
from torch.utils.data import Dataset
from PIL import Image
from models.structures import Instances
from collections import defaultdict
from torchvision.ops import nms
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from util.misc import linear_assignment
from random import choice, randint


class TAODatasetTrain(Dataset):  # TAO dataset
    vis_num = 3
    def __init__(self, tao_root: str, args, logger, transform=None, one_class=True):
        super().__init__()
        self.root = tao_root
        annotation_path = osp.join(tao_root, 'annotations', 'train_ours_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, logger)
        cats = tao.cats
        vid_img_map = tao.vid_img_map
        img_ann_map = tao.img_ann_map
        vids = tao.vids  # key: video id, value: video info
        self.args = args
        self.num_frames_per_batch = args.sampler_lengths[0]
        self.transform = transform
        self.one_class = one_class  # if true, take all objects as foreground
        self.all_frames_with_gt, self.all_indices, self.vid_tmax, categories_counter = self._generate_train_imgs(args.train_det_path, vid_img_map, img_ann_map, vids, cats,
                                                                                                                 args.train_base, args.train_max_det_num)
        categories_counter = sorted(categories_counter.items(), key=lambda x: x[0])
        print('found {} videos, {} imgs'.format(len(vids), len(self.all_indices)))
        print('number of categories: {}'.format(len(categories_counter)))

        self.sample_mode = args.sample_mode
        self.sample_interval = args.sample_interval

    def _generate_train_imgs(self, det_path, vid_img_map, img_ann_map, vids, cats, base_only, max_det=200):
        if base_only:
            print('only use base classes')
        all_frames_with_gt = {}
        all_indices = []
        vid_tmax = {}
        categories_counter = defaultdict(int)
        det = json.load(open(det_path, 'r'))
        det_box = defaultdict(list)
        for d in det:
            det_box[d['image_id']].append(d)
        for vid_info in vids.values():
            vid_id = vid_info['id']
            imgs = vid_img_map[vid_id]
            imgs = sorted(imgs, key=lambda x: x['frame_index'])
            num_imgs = len(imgs)
            targets = []  # gt and detection results
            img_infos = []
            cur_vid_indices = []
            for i in range(len(imgs)):
                img = imgs[i]
                cur_vid_indices.append((vid_id, i))
                gt_boxes, gt_labels, gt_track_ids, gt_scores, gt_iscrowd = [], [], [], [], []
                height, width = float(img['height']), float(img['width'])
                anns = img_ann_map[img['id']]
                img_id = img['id']
                img_infos.append({'file_name': img['file_name'],
                                'height': height,
                                'width': width,
                                'frame_index': img['frame_index'],
                                'image_id': img_id,
                                'video_id': vid_id})
                for ann in anns:
                    assert ann['iscrowd'] != 1
                    if base_only and cats[ann['category_id']]['frequency'] == 'r':  # ignore rare classes
                        continue
                    box = ann['bbox']  # x0, y0, w, h
                    box[2] += box[0]
                    box[3] += box[1]
                    gt_boxes.append(box)
                    categories_counter[ann['category_id']] += 1
                    gt_labels.append(ann['category_id'])  # category
                    gt_track_ids.append(ann['track_id'])
                    gt_scores.append(1.0)
                    gt_iscrowd.append(ann['iscrowd'])
                if len(gt_track_ids) == 0:
                    targets.append({})
                    continue
                gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
                gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)
                gt_scores = torch.as_tensor(gt_scores, dtype=torch.float32)
                gt_track_ids = torch.as_tensor(gt_track_ids, dtype=torch.float64)
                gt_iscrowd = torch.as_tensor(gt_iscrowd, dtype=torch.bool)
                # collect det results
                det_boxes, det_labels, det_track_ids, det_scores, det_iscrowd = [], [], [], [], []
                det_box_n = det_box[img_id]
                if len(det_box_n) != 0:
                    for d in det_box_n:
                        if base_only:
                            if cats[d['category_id']]['frequency'] == 'r':  # ignore rare classes
                                continue
                        box = d['bbox']  # x0, y0, w, h
                        box[2] += box[0]
                        box[3] += box[1]
                        det_boxes.append(box)
                        det_labels.append(d['category_id'])  # category
                        det_scores.append(d['score'])
                        det_iscrowd.append(False)
                    det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32)
                    det_labels = torch.as_tensor(det_labels, dtype=torch.long)
                    det_scores = torch.as_tensor(det_scores, dtype=torch.float32)
                    det_iscrowd = torch.as_tensor(det_iscrowd, dtype=torch.bool)
                    # nms
                    keep  = nms(det_boxes, det_scores, self.args.val_nms_thresh)
                    det_boxes = det_boxes[keep]
                    det_scores = det_scores[keep]
                    det_labels = det_labels[keep]
                else:
                    print('warning: no detection results for image id {}, number of gt box is {}.'.format(img_id, len(gt_track_ids)))
                    det_boxes = torch.empty((0, 4), dtype=torch.float32)
                    det_labels = torch.empty((0,), dtype=torch.long)
                    det_scores = torch.empty((0,), dtype=torch.float32)

                matched_boxes, matched_scores, matched_labels, matched_track_ids, matched_iscrowd, det_unmached = self._match_det_gt(det_boxes, 
                                                                                                                                     det_scores, 
                                                                                                                                     det_labels, 
                                                                                                                                     det_iscrowd,
                                                                                                                                     gt_boxes, 
                                                                                                                                     gt_track_ids, 
                                                                                                                                     gt_labels,
                                                                                                                                     gt_scores,
                                                                                                                                     gt_iscrowd)
                assert len(matched_boxes) <= max_det, 'too many matched dets in image {} ({}), max is {}.'.format(img_id, len(matched_boxes), max_det)
                if self.args.add_extra_dets:
                    extra_det_boxes, extra_det_scores, extra_det_labels, extra_det_iscrowd, extra_track_ids = self._gen_extra_dets(
                        det_boxes, det_labels, det_scores, det_unmached, self.args.train_score_thresh)
                    # only keep top k extra dets
                    if len(extra_det_boxes)+len(matched_boxes) > max_det:
                        print('warning: too many dets in image {} ({}), only keep top {}.'
                                .format(img_id, len(extra_det_boxes)+len(matched_boxes), max_det))
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
                # assert len(gt_boxes) == len(matched_boxes)
                boxes = torch.cat((matched_boxes, extra_det_boxes), dim=0)
                scores = torch.cat((matched_scores, extra_det_scores), dim=0)
                labels = torch.cat((matched_labels, extra_det_labels), dim=0)
                iscrowd = torch.cat((matched_iscrowd, extra_det_iscrowd), dim=0)
                track_ids = torch.cat((matched_track_ids, extra_track_ids), dim=0)
                gt_flag = torch.zeros((len(gt_scores)+len(scores),), dtype=torch.bool)
                gt_flag[:len(gt_scores)] = True
                targets.append({'boxes': torch.cat((gt_boxes, boxes), dim=0),  # x0, y0, x1, y1
                                'labels': torch.cat((gt_labels, labels), dim=0),
                                'scores': torch.cat((gt_scores, scores), dim=0),
                                'obj_ids': torch.cat((gt_track_ids, track_ids), dim=0),
                                'iscrowd': torch.cat((gt_iscrowd, iscrowd), dim=0),
                                'gt_flag': gt_flag})
            vid_tmax[vid_id] = i
            cur_vid_indices = [cur_vid_indices[i] for i in range(0, len(cur_vid_indices), self.args.clip_gap)]
            all_indices.extend(cur_vid_indices)
            all_frames_with_gt[vid_id] = {'img_infos': img_infos,
                                          'targets': targets,
                                          }
        return all_frames_with_gt, all_indices, vid_tmax, categories_counter
    
    def _targets_to_instances(self, targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        n_gt = targets['gt_flag'].sum()
        gt_instances.boxes = targets['boxes'][:n_gt]
        if self.one_class:
            gt_instances.labels = torch.zeros_like(targets['labels'][:n_gt], dtype=targets['labels'].dtype)
        else:
            gt_instances.labels = targets['labels'][:n_gt]
        gt_instances.obj_ids = targets['obj_ids'][:n_gt]
        return gt_instances
    
    def _generate_empty_instance(self, img_shape):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = torch.empty((0, 4), dtype=torch.float32)
        gt_instances.labels = torch.empty((0,), dtype=torch.int64)
        gt_instances.obj_ids = torch.empty((0,), dtype=torch.float64)
        return gt_instances
    
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
        if self.one_class:
            det_labels = torch.zeros_like(target['labels'][is_det], dtype=target['labels'].dtype)[..., None]
        else:
            det_labels = target['labels'][is_det][..., None]
        det_idxes = target['obj_ids'][is_det][..., None].float()
        proposal = torch.cat([det_boxes, det_scores, det_labels, det_idxes], dim=1)
        # assert torch.all(det_idxes[:len(gt_obj_ids)].flatten() == gt_obj_ids)
        return proposal
    
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

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        self.has_vised = 0
        return
    
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)

    def _match_det_gt(self, det_boxes, det_scores, det_labels, det_iscrowd, gt_boxes, gt_obj_ids, gt_labels, gt_scores, gt_iscrowd, amend_use_gt=False):
        assert gt_boxes.shape[0] != 0
        iou, union = box_iou(det_boxes, gt_boxes)
        matches, det_unmached, _ = linear_assignment((1.0-iou).numpy(), thresh=1-self.args.train_iou_thresh)
        if amend_use_gt:
            matched_obj_ids = gt_obj_ids.clone()
            matched_labels = gt_labels.clone()
            matched_boxes = gt_boxes.clone()
            matched_scores = gt_scores.clone()
            matched_iscrowd = gt_iscrowd.clone()
            matched_scores[matches[:, 1]] = det_scores[matches[:, 0]]
            matched_boxes[matches[:, 1]] = det_boxes[matches[:, 0]]
            matched_iscrowd[matches[:, 1]] = det_iscrowd[matches[:, 0]]
        else:
            matched_boxes = det_boxes[matches[:, 0]]
            matched_scores = det_scores[matches[:, 0]]
            matched_labels = gt_labels[matches[:, 1]]
            matched_obj_ids = gt_obj_ids[matches[:, 1]]
            matched_iscrowd = gt_iscrowd[matches[:, 1]]
        # assert matched_boxes.shape == gt_boxes.shape
        return matched_boxes, matched_scores, matched_labels, matched_obj_ids, matched_iscrowd, det_unmached

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

    def __len__(self):
        return len(self.all_indices)
    
    def sample_indices(self, vid, f_index):
        assert self.sample_mode == 'random_interval'
        tmax = self.vid_tmax[vid]
        ids = [f_index]
        for i in range(1, self.num_frames_per_batch):
            id_ = ids[-1] + randint(1, self.sample_interval)
            while id_ > tmax:
                id_ = id_ - tmax - 1
            ids.append(id_)
        return ids
    
    def __getitem__(self, idx: int):
        vid, f_index = self.all_indices[idx]
        indices = self.sample_indices(vid, f_index)
        img_infos, targets = [], []
        for i in indices:
            img_infos.append(self.all_frames_with_gt[vid]['img_infos'][i])
            targets.append(self.all_frames_with_gt[vid]['targets'][i])
        
        ori_images = [Image.open(osp.join(self.root, 'frames', img_info['file_name'])) \
                  for img_info in img_infos]
        if self.transform is not None:
            images, targets = self.transform(ori_images, targets)
        else:
            raise ValueError('transform is None')
        # self.vis_one_clip(images, targets)
        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            if 'gt_flag' in targets_i:
                # gt
                # n_gt = targets_i['gt_flag'].sum()
                # assert n_gt <= len(targets_i['labels'])/2
                # assert torch.all(targets_i['obj_ids'][:n_gt] == targets_i['obj_ids'][n_gt:2*n_gt])
                gt_instance = self._targets_to_instances(targets_i, img_i.shape[1:3])
                gt_instances.append(gt_instance)
                # det results
                proposal = self._generate_proposal(targets_i)
                # add extra dets
                proposals.append(proposal)
                # assert len(gt_instance) <= len(proposal)  # TODO will reprot an error？？？
            else:
                # gt
                gt_instances.append(self._generate_empty_instance(img_i.shape[1:3]))
                # det results
                proposals.append(torch.empty((0,7), dtype=torch.float32))
        if self.args.shuffle_clip:
            indices = list(range(len(images)))
            random.shuffle(indices)
            images = [images[i] for i in indices]
            img_infos = [img_infos[i] for i in indices]
            gt_instances = [gt_instances[i] for i in indices]
            ori_images = [ori_images[i] for i in indices]
            proposals = [proposals[i] for i in indices]
        return {
            'imgs': images,
            'img_infos': img_infos,
            'gt_instances': gt_instances,
            'ori_imgs': ori_images, 
            'proposals': proposals,  # labels for proposals are not used in training
        }
    
class TAODatasetVal(Dataset):  # TAO dataset
    def __init__(self, tao_root: str, args, logger, transform=None, one_class=True):
        super().__init__()
        self.root = tao_root
        if args.split == 'val':
            annotation_path = osp.join(tao_root, 'annotations', 'validation_ours_v1.json')
        else:
            annotation_path = osp.join(tao_root, 'annotations', 'tao_test_burst_v1.json')
        print('using annotation path {}'.format(annotation_path))
        tao = Tao(annotation_path, logger)
        self.tao = tao
        cats = tao.cats
        vid_img_map = tao.vid_img_map
        img_ann_map = tao.img_ann_map
        vids = tao.vids  # key: video id, value: video info
        self.args = args
        self.num_frames_per_batch = args.sampler_lengths[0]
        self.transform = transform
        # thresh
        self.score_thresh = args.val_score_thresh
        self.nms_thresh = args.val_nms_thresh
        self.one_class = one_class  # if true, take all objects as foreground
        self.clips = self._generate_val_clips(args.val_det_path, vid_img_map, img_ann_map, vids,
                                              args.val_max_det_num)
        print('found {} videos, {} clips'.format(len(vids), len(self.clips)))

    def _generate_val_clips(self, det_path, vid_img_map, img_ann_map, vids, max_det=200):
        clips = []
        det = json.load(open(det_path, 'r'))
        det_box = defaultdict(list)
        for d in det:
            det_box[d['image_id']].append(d)
        for vid_info in vids.values():
            vid_id = vid_info['id']
            imgs = vid_img_map[vid_id]
            imgs = sorted(imgs, key=lambda x: x['frame_index'])
            num_imgs = len(imgs)
            clip = imgs
            targets = []
            img_infos = []
            for img in clip:
                img_id = img['id']
                height, width = float(img['height']), float(img['width'])
                img_infos.append({'file_name': img['file_name'],
                                'height': height,
                                'width': width,
                                'frame_index': img['frame_index'],
                                'image_id': img_id,
                                'video_id': vid_id})
                det_boxes, det_labels, det_track_ids, det_scores, det_iscrowd = [], [], [], [], []
                det_box_n = det_box[img_id]
                for d in det_box_n:
                    score = d['score']
                    if score < self.score_thresh:
                        continue
                    box = d['bbox']  # x0, y0, w, h
                    box[2] += box[0]
                    box[3] += box[1]
                    # box = clip_box(box, height, width)
                    det_boxes.append(box)
                    det_labels.append(d['category_id'])  # category
                    det_scores.append(score)
                if len(det_boxes) != 0:
                    if self.nms_thresh < 1.0:
                        det_boxes = torch.as_tensor(det_boxes, dtype=torch.float32)
                        det_labels = torch.as_tensor(det_labels, dtype=torch.long)
                        det_scores = torch.as_tensor(det_scores, dtype=torch.float32)
                        keep  = nms(det_boxes, det_scores, self.nms_thresh)
                        det_boxes = det_boxes[keep]
                        det_labels = det_labels[keep]
                        det_scores = det_scores[keep]
                    # only keep top k dets
                    if len(det_boxes) > max_det:
                        print('warning: too many dets in image {} ({}), only keep top {}.'
                              .format(img_id, len(det_boxes), max_det))
                        det_scores, indices = torch.topk(det_scores, max_det)
                        det_boxes = det_boxes[indices]
                        det_labels = det_labels[indices]
                    targets.append({'boxes': det_boxes,  # x0, y0, x1, y1
                                    'labels': det_labels,
                                    'scores': det_scores})
                else:
                    print('warning: no detection results for image id {}'.format(img_id))
                    targets.append({})
            clips.append([img_infos, targets])
        return clips
    
    def _generate_empty_instance(self, img_shape):
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = torch.empty((0, 4), dtype=torch.float32)
        gt_instances.labels = torch.empty((0,), dtype=torch.int64)
        gt_instances.obj_ids = torch.empty((0,), dtype=torch.float64)
        return gt_instances
    
    def _generate_proposal(self, target: dict):
        det_boxes = target['boxes']
        det_scores = target['scores'][..., None]
        if self.one_class:
            labels = torch.zeros_like(target['labels'], dtype=target['labels'].dtype)[..., None]
        else:
            labels = target['labels'][..., None]
        proposal = torch.cat([det_boxes, det_scores, labels], dim=1)
        return proposal
    
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        return
    
    def step_epoch(self):
        # one epoch finishes.
        print("Dataset: epoch {} finishes".format(self.current_epoch))
        self.set_epoch(self.current_epoch + 1)
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx: int):
        img_infos, targets = copy.deepcopy(self.clips[idx])
        ori_images = [Image.open(osp.join(self.root, 'frames', img_info['file_name'])) \
                  for img_info in img_infos]
        if self.transform is not None:
            images, targets = self.transform(ori_images, targets)
        else:
            raise ValueError('transform is None')
        proposals = []
        for img_i, targets_i in zip(images, targets):
            if 'boxes' in targets_i:  # not empty
                # det results
                proposal = self._generate_proposal(targets_i)
                proposals.append(proposal)
            else:
                # det results
                proposals.append(None)
        return {
            'imgs': images,
            'img_infos': img_infos,  # for inference
            'ori_imgs': ori_images, 
            'proposals': proposals,  # labels for proposals are used in inference
        }
    
    
def clip_box(box, h, w):
    box[0] = max(0, box[0])
    box[1] = max(0, box[1])
    box[2] = min(w, box[2])
    box[3] = min(h, box[3])
    return box
    
def make_transforms_for_TAO(image_set, args=None):

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
    train = make_transforms_for_TAO('train', args)
    test = make_transforms_for_TAO('val', args)

    if image_set == 'train':
        return train
    elif image_set == 'val' or image_set == 'exp' or image_set == 'val_gt':
        return test
    else:
        raise NotImplementedError()
    
def build(image_set, args):
    root = osp.join(args.mot_path, 'TAO')
    assert osp.exists(root), 'provided MOT path {} does not exist'.format(root)
    transform = build_transform(args, image_set)
    if image_set == 'train':
        dataset = TAODatasetTrain(root, args, logger=None, transform=transform, one_class=True)
    if image_set == 'val':
        dataset = TAODatasetVal(root, args, logger=None, transform=transform, one_class=False)
    return dataset