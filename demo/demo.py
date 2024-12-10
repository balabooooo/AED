import argparse
import os
import sys
import cv2
import datetime

import numpy as np
import torch

import groundingdino.datasets.transforms as T
import torchvision.transforms.functional as F

from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models import build_model as build_aed
from models import RuntimeTrackerBase
from main import get_args_parser
from util.tool import load_model
from torchvision.ops import nms


def get_color(idx):
    idx = int(idx)
    idx += 14
    idx *= 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_boxes_to_image(img, bbox_xyxy, identities, labels, scores, pred_phrases):
    for i in range(len(bbox_xyxy)):
        x1, y1, x2, y2 = bbox_xyxy[i].astype(int)
        labels_i = labels[i]
        id = identities[i]
        phrase = pred_phrases[labels_i]
        score = scores[i]
        phrase = str(id) + ": " + phrase
        color = get_color(id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, phrase, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img


def transform_image(image_ori, height=800, width=1333, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
    h, w = image_ori.shape[:2]
    scale = min(height / h, width / w)
    image = cv2.resize(image_ori, (int(w * scale), int(h * scale)))
    image = F.normalize(F.to_tensor(image), mean, std)
    image = image.unsqueeze(0)
    return image_ori, image


def build_groundingdino(model_config_path, model_checkpoint_path):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    with torch.no_grad():
        outputs = model(image, captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
        logits_filt = logits_filt.max(dim=1, keepdim=True)[0]
    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        logits_filt = torch.cat(all_logits, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, logits_filt, pred_phrases

def gen_proposals(boxes, logits):
    proposals = []
    for i, (box, logit) in enumerate(zip(boxes, logits)):
        x, y, w, h = box
        proposals.append([x, y, w, h, logit, i])
    return torch.as_tensor(proposals).reshape(-1, 6)

def track_on_video(aed, dino, video_path, output_dir, text_prompt, box_threshold, text_threshold, token_spans, nms_thresh):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_path = os.path.join(output_dir, datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.mp4')
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

    device = torch.device("cuda")
    track_instances = None
    print("Start tracking...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_ori, image = transform_image(frame)
        image = image.to(device)
        # boxes_filt: x, y, w, h (normalized)
        boxes_filt, logits_filt, pred_phrases = get_grounding_output(
            dino, image, text_prompt, box_threshold, text_threshold, token_spans=eval(f"{token_spans}")
        )
        # nms
        keep = nms(boxes_filt, logits_filt.squeeze(1), nms_thresh)
        boxes_filt = boxes_filt[keep]
        logits_filt = logits_filt[keep]
        pred_phrases = [pred_phrases[i] for i in keep]
        proposals = gen_proposals(boxes_filt, logits_filt).to(device)
        num_proposals = len(proposals)
        if track_instances is not None:
            track_instances.remove('boxes')
        res = aed.inference_single_image(image, (frame_height, frame_width), num_proposals, track_instances, proposals)
        track_instances = res['track_instances']
        num_active_proposals = res['num_active_proposals']
        dt_instances = track_instances[:num_active_proposals]
        bbox_xyxy = dt_instances.boxes.cpu().numpy()
        identities = dt_instances.obj_ids.cpu().numpy()
        labels = dt_instances.labels.cpu().numpy()
        scores = dt_instances.det_scores.cpu().numpy()
        image_with_box = plot_boxes_to_image(image_ori, bbox_xyxy, identities, labels, scores, pred_phrases)
        image_with_box = cv2.cvtColor(image_with_box, cv2.COLOR_RGB2BGR)
        vid_writer.write(image_with_box)
    vid_writer.release()
    cap.release()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True, parents=[get_args_parser()])
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--video_path", "-v", type=str, required=True, help="path to image file or use webcam")
    parser.add_argument("--text_prompt", "-t", type=str, required=True, help="text prompt")
    parser.add_argument(
        "--demo_output_dir", "-o", type=str, default=None, required=True, help="output directory"
    )

    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--token_spans", type=str, default=None, help=
                        "The positions of start and end positions of phrases of interest. \
                        For example, a caption is 'a cat and a dog', \
                        if you would like to detect 'cat', the token_spans should be '[[[2, 5]], ]', since 'a cat and a dog'[2:5] is 'cat'. \
                        if you would like to detect 'a cat', the token_spans should be '[[[0, 1], [2, 5]], ]', since 'a cat and a dog'[0:1] is 'a', and 'a cat and a dog'[2:5] is 'cat'. \
                        ")
    parser.add_argument('--miss_tolerance', default=30, type=int)

    args = parser.parse_args()

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    video_path = args.video_path
    text_prompt = args.text_prompt
    output_dir = args.demo_output_dir
    box_threshold = args.val_score_thresh
    text_threshold = args.text_threshold
    token_spans = args.token_spans
    nms_thresh = args.val_nms_thresh

    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # model
    aed, _, _ = build_aed(args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    aed = load_model(aed, args.resume)
    aed.track_base = RuntimeTrackerBase(args.val_match_high_thresh, args.val_match_low_thresh, args.miss_tolerance, args.match_high_score)
    aed.eval()
    dino = build_groundingdino(config_file, checkpoint_path)
    aed = aed.cuda()
    dino = dino.cuda()
    track_on_video(aed, dino, video_path, output_dir, text_prompt, box_threshold, text_threshold, token_spans, nms_thresh)
