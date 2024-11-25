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



import argparse
import datetime
import random
import time
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from util.tool import load_model
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset
from engine import train_one_epoch_mot
from models import build_model
from torch.utils.tensorboard import SummaryWriter


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets',], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=2, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+',
                        help='if not None, it will override lr_drop')
    parser.add_argument('--save_period', default=1, type=int,
                        help='save checkpoint every epochs')
    parser.add_argument('--print_freq', default=50, type=int,
                        help='number of interval to print results')
    parser.add_argument('--clip_max_norm', default=1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--meta_arch', default='AED', type=str)
    parser.add_argument('--sgd', action='store_true',
                        help='use sgd optimizer, use adam otherwise')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true',
                        help='Refine box in every decoder layer, otherwise use detection results.')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, 
                        help='number of feature levels, which is the input of transformer')

    # Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the sim-decoder")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--decoder_cross_self', default=False, action='store_true',
                        help='cross attention performs before self attention in decoder')
    parser.add_argument('--sigmoid_attn', default=False, action='store_true',
                        help='attrion weight use sigmoid instead of softmax')
    parser.add_argument('--extra_track_attn', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (compute loss at each layer)")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--pt_weight_loss_coef', default=1, type=float)  # temporal
    parser.add_argument('--pp_weight_loss_coef', default=1, type=float)  # spatial
    parser.add_argument('--cross_clip_weight_loss_coef', default=1, type=float)  # cross-clip

    # dataset settings
    parser.add_argument('--dataset_file', default='coco',
                        help='dataset name')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pretrained', default=None, 
                        help='load checkpoint')
    parser.add_argument('--cache_mode', default=False, action='store_true', 
                        help='whether to cache images on memory')

    parser.add_argument('--train_det_path', default='', type=str, 
                        help='path to detecition reuslts (json file)')
    parser.add_argument('--val_det_path', default='', type=str, 
                        help='path to detecition reuslts (json file)')
    parser.add_argument('--max_size', default=1333, type=int)
    parser.add_argument('--train_iou_thresh', type=float, default=0.5,
                        help='min iou for matching gt and det')
    parser.add_argument('--val_nms_thresh', type=float, default=0.7,
                        help='nms thresh for detection box')
    parser.add_argument('--val_score_thresh', type=float, default=0.5,
                        help='score thresh for detection box')
    parser.add_argument('--train_score_thresh', type=float, default=0.5,
                        help='score thresh for detection box')
    parser.add_argument('--add_extra_dets', default=False, action='store_true',
                        help='whether to add extra detections (e.g. false positives or unlabeled boxes) in training')
    parser.add_argument('--train_base', default=False, action='store_true',
                        help='whether use base class only (define in OVTrack) during training, tao dataset only')
    parser.add_argument('--train_match_thresh', default=0.8, type=float,
                        help='matching thresh in hungarian matcher')
    parser.add_argument('--val_max_det_num', default=100, type=int,
                        help='max detections per image in validation')
    parser.add_argument('--train_max_det_num', default=100, type=int,
                        help='max detections per image in training')
    parser.add_argument('--shuffle_clip', default=False, action='store_true',
                        help='whether to shuffle clips in training')
    parser.add_argument('--remove_unmatched_dets', default=False, action='store_true',
                        help='remove detections unmatched with any GT in training, invalid for tao dataset')
    parser.add_argument('--sample_interval', type=int, default=1,
                        help='interval will be randomly sampled from [1, sample_interval]')
    parser.add_argument('--clip_gap', type=int, default=1)
    parser.add_argument('--sampler_lengths', type=int, default=[5], nargs='*',
                        help='number of frames per clip')
    parser.add_argument('--sample_mode', type=str, default='random_interval', choices=('fixed_interval', 'random_interval'),
                        help='mode of sampling frames from datasets')
    
    # mot settings
    parser.add_argument('--mot_path', type=str, required=True,
                        help='root path to mot datasets')
    parser.add_argument('--random_drop', type=float, default=0,
                        help='random track drop ratio in QueryUpdating.')
    parser.add_argument('--use_checkpoint', action='store_true', default=False,
                        help='use checkpoint to save GPU memory')
    parser.add_argument('--ema_weight', default=0.5, type=float,
                        help='ema weight in QueryUpdating')
    parser.add_argument('--match_high_score', type=float, default=0.7)
    parser.add_argument('--val_match_high_thresh', type=float, default=0.95)
    parser.add_argument('--val_match_low_thresh', type=float, default=0.6)

    # other settings
    parser.add_argument('--occupy_mem', default=False, action='store_true',
                        help='whether to occupy GPU memory')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    if args.occupy_mem:
        print('occupying mem...')
        utils.occupy_mem(utils.get_rank())

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    collate_fn = utils.mot_collate_fn
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    if args.lr_drop_epochs is None:
        print(f'lr_drop_epochs is empty, use lr_drop = {args.lr_drop} instead.')
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    else:
        print(f'lr_drop_epochs and lr_drop confilct, use lr_drop_epochs = {args.lr_drop_epochs} instead.')
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs, gamma=0.1)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    if args.pretrained is not None:
        args.pretrained = os.path.abspath(args.pretrained)
        model_without_ddp = load_model(model_without_ddp, args.pretrained)

    output_dir = Path(args.output_dir)
    if args.resume:
        print(f'resume from checkpoint {args.resume}')
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

    # init tensorboard
    if args.output_dir and int(os.environ.get("RANK", 0))==0:
        print(f'init tensorboard')
        writer = SummaryWriter(output_dir/'tensorboard_logs')
    else:
        writer = None
    print("Start training")
    start_time = time.time()

    # dataset_train.set_epoch(args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch_mot(
            model, criterion, data_loader_train, optimizer, device, epoch, args.dataset_file, args.clip_max_norm, writer, args.print_freq,)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint_last.pth']
            if (epoch + 1) % args.save_period == 0:
                checkpoint_paths.append(output_dir / f'checkpoint_epoch_{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        # dataset_train.step_epoch()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def check_args(args):
    print('checking args...')
    assert args.extra_track_attn == False, 'extra_track_attn is not supported yet.'
    assert args.cache_mode == False, 'cache_mode seems to be wrong.'
    assert len(args.sampler_lengths) == 1, 'only support one sampler length now.'
    assert args.batch_size == 1, 'batch_size only support 1 now.'
    assert args.decoder_cross_self == True, 'decoder_cross_self only support True now.'
    assert args.two_stage == False, 'two_stage only support False now.'
    assert args.train_iou_thresh >= 0.5, 'train_iou_thresh should be larger than 0.5.'

    assert args.random_drop >= 0, 'random_drop should be non-negative.'
    if args.train_base:
        assert args.dataset_file == 'tao', 'only support tao dataset for training base classes.'

    if args.use_checkpoint:
        print('using checkpoint to save GPU memory.')
    
    if not args.with_box_refine:
        print('warning: not using box refinement.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    check_args(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
