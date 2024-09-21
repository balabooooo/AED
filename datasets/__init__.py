# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from .tao_dataset import build as build_tao
from .dancetrack_dataset import build as build_dancetrack
from .sportsmot_dataset import build as build_sportsmot

def build_dataset(image_set, args):
    if args.dataset_file == 'tao':
        return build_tao(image_set, args)
    elif args.dataset_file == 'dancetrack':
        return build_dancetrack(image_set, args)
    elif args.dataset_file == 'sportsmot':
        return build_sportsmot(image_set, args)
    else:
        raise ValueError(f'dataset {args.dataset_file} not supported')
