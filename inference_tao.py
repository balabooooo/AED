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


import json
import datetime
import os
import argparse
import torchvision.transforms.functional as F
import util.misc as utils
import torch
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from models import build_model, RuntimeTrackerBase
from datasets import build_dataset
from util.tool import load_model
from main import get_args_parser
from tao.toolkit.tao import Tao
from util.evaluation import teta_eval
from copy import deepcopy

from models.structures import Instances
from torch.utils.data import Dataset, DataLoader


class ListImgDataset(Dataset):
    def __init__(self, mot_path, img_list, det_db) -> None:
        super().__init__()
        self.mot_path = mot_path
        self.img_list = img_list
        self.det_db = det_db

        '''
        common settings
        '''
        self.img_height = 800
        self.img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_img_from_file(self, f_path):
        cur_img = cv2.imread(os.path.join(self.mot_path, f_path))
        assert cur_img is not None, f_path
        cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
        proposals = []
        im_h, im_w = cur_img.shape[:2]
        for line in self.det_db[f_path[:-4] + '.txt']:
            l, t, w, h, s = list(map(float, line.split(',')))
            l = max(0, min(l, im_w - 1))
            t = max(0, min(t, im_h - 1))
            w = max(0, min(w, im_w - l))
            h = max(0, min(h, im_h - t))
            proposals.append([(l + w / 2) / im_w,
                                (t + h / 2) / im_h,
                                w / im_w,
                                h / im_h,
                                s])
        return cur_img, torch.as_tensor(proposals).reshape(-1, 5)

    def init_img(self, img, proposals):
        ori_img = img.copy()
        self.seq_h, self.seq_w = img.shape[:2]
        scale = self.img_height / min(self.seq_h, self.seq_w)
        if max(self.seq_h, self.seq_w) * scale > self.img_width:
            scale = self.img_width / max(self.seq_h, self.seq_w)
        target_h = int(self.seq_h * scale)
        target_w = int(self.seq_w * scale)
        img = cv2.resize(img, (target_w, target_h))
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)
        return img, ori_img, proposals

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img, proposals = self.load_img_from_file(self.img_list[index])
        return self.init_img(img, proposals)

class MyTracker:
    def __init__(self, args, model, data):
        self.args = args
        self.detr = model
        self.imgs = data['imgs']
        self.img_infos = data['img_infos']
        self.ori_images = data['ori_imgs']
        self.proposals = data['proposals']
        self.seq_name = self.img_infos[0]['file_name'].split('/')[:3]

        self.img_len = len(self.imgs)
        self.result = list()

    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]

    def detect(self, area_threshold=100):
        total_dts = 0
        track_instances = None
        for i, (cur_img, img_info, ori_img, proposal) in enumerate(zip(self.imgs, self.img_infos, self.ori_images, self.proposals)):
            image_id = img_info['image_id']
            video_id = img_info['video_id']
            # ori_img is PIL image
            ori_img_tensor = F.to_tensor(ori_img)
            cur_img = cur_img.unsqueeze(0)
            # proposal x, y, w, h, score
            if proposal is not None:
                proposal = proposal.cuda()
            num_proposals = len(proposal) if proposal is not None else 0
            
            cur_img = cur_img.cuda()
            if track_instances is not None:
                track_instances.remove('boxes')
            _, seq_h, seq_w= ori_img_tensor.shape

            assert seq_h == img_info['height'] and seq_w == img_info['width']

            res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), num_proposals, track_instances, proposal)
            track_instances = res['track_instances']

            dt_instances = deepcopy(track_instances[:num_proposals])

            dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

            total_dts += len(dt_instances)

            bbox_xyxy = dt_instances.boxes.tolist()
            identities = dt_instances.obj_ids.tolist()
            labels = dt_instances.labels.tolist()
            scores = dt_instances.det_scores.tolist()
            track_ids = dt_instances.obj_ids.tolist()
            is_news = dt_instances.new.tolist()

            if len(labels) != 0:
                for xyxy, track_id, label, score, track_id, is_new in zip(bbox_xyxy, identities, labels, scores, track_ids, is_news):
                    if track_id < 0 or track_id is None:
                        raise ValueError('track_id < 0 or track_id is None')
                    x1, y1, x2, y2 = xyxy
                    w, h = x2 - x1, y2 - y1
                    self.result.append({
                        'image_id': image_id,
                        'category_id': label,
                        'bbox': [x1, y1, w, h],
                        'score': score,
                        'track_id': track_id,
                        'video_id': video_id,
                        'is_new': is_new,
                    })
        return self.result, total_dts

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--miss_tolerance', default=10, type=int)
    parser.add_argument('--exp_name', default='tracker', type=str)
    parser.add_argument('--split', default='val', type=str, choices=['val', 'test'])
    args = parser.parse_args()
    args.with_box_refine = False
    print(args)
    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_base = RuntimeTrackerBase(args.val_match_high_thresh, args.val_match_low_thresh, args.miss_tolerance, args.match_high_score)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    dataset_val = build_dataset(image_set='val', args=args)
    collate_fn = utils.mot_collate_fn
    data_loader_val = DataLoader(dataset_val, collate_fn=collate_fn, num_workers=args.num_workers, batch_size=1,
                                   shuffle=False, drop_last=False, pin_memory=True)

    result = list()
    pbar = tqdm(data_loader_val)
    for data in pbar:
        tracker = MyTracker(args, model=detr, data=data)
        result_i, total_dts = tracker.detect()
        pbar.set_description("{} dts in total".format(total_dts))
        result.extend(result_i)

    # save and evaluate
    predict_path = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(predict_path, exist_ok=True)
    # add date and time
    result_name = 'infer_result' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.json'
    result_path = os.path.join(predict_path, result_name)
    print('saving inference result to {}'.format(result_path))
    json.dump(result, open(result_path, 'w'), indent=4)
