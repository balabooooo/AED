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


import os
import argparse
import torch
import shutil
from pathlib import Path
from models import build_model
from util.tool import load_model
from main import get_args_parser

from inference_dancetrack import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    parser.add_argument('--score_threshold', default=0.5, type=float)
    parser.add_argument('--update_score_threshold', default=0.5, type=float)
    parser.add_argument('--miss_tolerance', default=180, type=int)
    parser.add_argument('--exp_name', default='sportsmot_result', type=str)
    parser.add_argument('--split', default='test', type=str, choices=['val', 'test'])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load model and weights
    detr, _, _ = build_model(args)
    detr.track_base = RuntimeTrackerBase(args.val_match_high_thresh, args.val_match_low_thresh, args.miss_tolerance, args.match_high_score)
    checkpoint = torch.load(args.resume, map_location='cpu')
    detr = load_model(detr, args.resume)
    detr.eval()
    detr = detr.cuda()

    # for SportsMOT submittion
    args.mot_path = os.path.join(args.mot_path, 'SportsMOT/dataset')
    sub_dir = f'{args.split}'
    seq_nums = os.listdir(os.path.join(args.mot_path, sub_dir))
    if 'seqmap' in seq_nums:
        seq_nums.remove('seqmap')
    vids = [os.path.join(sub_dir, seq) for seq in seq_nums]

    for i, vid in enumerate(vids):
        print(vid, i+1, '/', len(vids))
        detr.track_base.clear()
        det = Detector(args, model=detr, vid=vid)
        det.detect()