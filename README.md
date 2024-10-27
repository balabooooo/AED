# Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown
[![arXiv](https://img.shields.io/badge/arXiv-2409.09293-red.svg)](https://arxiv.org/abs/2409.09293)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associate-everything-detected-facilitating/multi-object-tracking-on-tao)](https://paperswithcode.com/sota/multi-object-tracking-on-tao?p=associate-everything-detected-facilitating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associate-everything-detected-facilitating/multi-object-tracking-on-sportsmot)](https://paperswithcode.com/sota/multi-object-tracking-on-sportsmot?p=associate-everything-detected-facilitating)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/associate-everything-detected-facilitating/multi-object-tracking-on-dancetrack)](https://paperswithcode.com/sota/multi-object-tracking-on-dancetrack?p=associate-everything-detected-facilitating)

This repository is an official implementation of the paper [Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown](https://arxiv.org/abs/2409.09293).

This repository is still under development, and feel free to raise any issues at any time.
## News🔥
* (2024/9/14) Our paper is available at [arXiv](https://arxiv.org/abs/2409.09293).
## Abstract
Multi-object tracking (MOT) emerges as a pivotal and highly promising branch in the field of computer vision. Classical closed-vocabulary MOT (CV-MOT) methods aim to track objects of predefined categories. Recently, some open-vocabulary MOT (OV-MOT) methods have successfully addressed the problem of tracking unknown categories. However, we found that the CV-MOT and OV-MOT methods each struggle to excel in the tasks of the other. In this paper, we present a unified framework, Associate Everything Detected (AED), that simultaneously tackles CV-MOT and OV-MOT by integrating with any off-the-shelf detector and supports unknown categories. Different from existing tracking-by-detection MOT methods, AED gets rid of prior knowledge (e.g. motion cues) and relies solely on highly robust feature learning to handle complex trajectories in OV-MOT tasks while keeping excellent performance in CV-MOT tasks. Specifically, we model the association task as a similarity decoding problem and propose a sim-decoder with an association-centric learning mechanism. The sim-decoder calculates similarities in three aspects: spatial, temporal, and cross-clip. Subsequently, association-centric learning leverages these threefold similarities to ensure that the extracted features are appropriate for continuous tracking and robust enough to generalize to unknown categories. Compared with existing powerful OV-MOT and CV-MOT methods, AED achieves superior performance on TAO, SportsMOT, and DanceTrack without any prior knowledge.
## Main Results
### TAO Test Set

| **Method** | **Training Data** | **Detector** | **Base-TETA** | **Base-AssocA** | **Novel-TETA** | **Novel-AssocA** | URL                                                                                            |
| ---------- | ----------------- | ------------ | ------------- | --------------- | -------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| AED        | TAO-train         | RegionCLIP   | 37.2          | 40.4            | 27.8           | 29.1             | ⬇️                                                                                             |
| AED        | TAO-train         | Co-DETR      | 54.8          | 54.1            | 48.9           | 51.8             | [model](https://drive.google.com/file/d/1YBPGstt9slY9UZ0CgC-d_v34K5aBuEcT/view?usp=drive_link) |
### SportsMOT Test Set

| **Method** | **Training Data** | **HOTA** | **IDF1** | **AssA** | **MOTA** | URL                                                                                         |
| ---------- | ----------------- | -------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------- |
| AED        | TAO-train         | 72.8     | 76.8     | 61.4     | 95.0     |                                                                                             |
| AED        | SportsMOT-train   | 77.0     | 80.0     | 68.1     | 95.1     | [model](https://drive.google.com/file/d/1tHLywmmj3YZhqOtOEkjCjoyvSySdThIU/view?usp=sharing) |
### DanceTrack Test Set

| **Method** | **Training Data** | **HOTA** | **IDF1** | **AssA** | **MOTA** | URL                                                                                         |
| ---------- | ----------------- | -------- | -------- | -------- | -------- | ------------------------------------------------------------------------------------------- |
| AED        | TAO-train         | 55.2     | 57.0     | 37.8     | 91.0     |                                                                                             |
| AED        | DanceTrack-train  | 66.6     | 69.7     | 54.3     | 92.2     | [model](https://drive.google.com/file/d/1vli10hglrE_jBbgTgoG4A5eUXMpgKDlH/view?usp=sharing) |
## Installation
The codebase is built on top of [MOTRv2](https://github.com/megvii-research/MOTRv2).
### Requirements
* Install pytorch using conda (optional), PyTorch>=1.5.1, torchvision>=0.6.1
```bash
conda create -n aed python=3.7
conda activate aed
# pytorch installation please refer to https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.10.1 torchvision==0.11.2 -c pytorch
```
* Other Requirements
```bash
pip install -r requirements.txt
```
* Build MultiScaleDeformableAttention
```bash
cd ./models/ops
sh ./make.sh
```
## Dataset preparation
It is recommended to symlink the dataset root to `$AED/data`.
### TAO Dataset
1. Pleases download TAO from [here](https://motchallenge.net/tao_download.php).
2. Note that you need to fill in this [form](https://motchallenge.net/tao_download_secure.php) to request missing AVA and HACS videos in the TAO dataset.
3. Convert TAO to COCO format and generate TAO val & test v1 filefollowing [OVTrack](https://github.com/SysCV/ovtrack), or you can simply download from [here](https://drive.google.com/file/d/1S2s9sbvhPrHh4XXpb2Qjce1G6i6W5lVh/view?usp=sharing).
### SportsMOT Dataset
Pleases download SportsMOT from [SportsMOT](https://github.com/MCG-NJU/SportsMOT).
### DanceTrack Dataset
Pleases download DanceTrack from [DanceTrack](https://github.com/DanceTrack/DanceTrack).
### Detection Results
We've run the inference phase on two detectors, RegionCLIP and Co-DETR, and saved their detection results as JSON files.

For YOLOX, we get the detection results from [MixSort](https://github.com/MCG-NJU/MixSort) and [MOTRv2](https://github.com/megvii-research/MOTRv2) for SportsMOT and DanceTrack respectively.

All of the detection results can be downloaded from [here](https://drive.google.com/file/d/1dcdcwIpxZ0E7mni1_pnbhuIRiufHmUpB/view?usp=sharing).

Here are the details for the json files:

| JSON File                              | Dataset                        | **Detector**                                                                                              |
| -------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `TAO_Co-DETR_test.json`                | TAO (base + novel), test       | [Co-DETR (LVIS)](https://github.com/Sense-X/Co-DETR)                                                      |
| `TAO_Co-DETR_train.json`               | TAO (base + novel), train      | [Co-DETR (LVIS)](https://github.com/Sense-X/Co-DETR)                                                      |
| `TAO_Co-DETR_val.json`                 | TAO (base + novel), val        | [Co-DETR (LVIS)](https://github.com/Sense-X/Co-DETR)                                                      |
| `TAO_RegionCLIP_test.json`             | TAO (base + novel), test       | [RegionCLIP (regionclip_finetuned-lvis_rn50 + rpn_lvis_866_lsj)](https://github.com/microsoft/RegionCLIP) |
| `TAO_RegionCLIP_train.json`            | TAO (base + novel), train      | [RegionCLIP (regionclip_finetuned-lvis_rn50 + rpn_lvis_866_lsj)](https://github.com/microsoft/RegionCLIP) |
| `TAO_RegionCLIP_val.json`              | TAO (base + novel), val        | [RegionCLIP (regionclip_finetuned-lvis_rn50 + rpn_lvis_866_lsj)](https://github.com/microsoft/RegionCLIP) |
| `YOLOX_DanceTrack_train_val_test.json` | DanceTrack, train + val + test | [YOLOX from MOTRv2](https://github.com/megvii-research/MOTRv2)                                            |
| `YOLOX_SportsMOT_train_val_test.json`  | SportsMOT, train + val + test  | [YOLOX from MixSort](https://github.com/MCG-NJU/MixSort)                                                  |

**When the downloads are complete, the folder structure should follow:**
```
├── configs
│   ├── dancetrack.args
│   ├── sportsmot.args
│   └── tao.args
├── data
│   ├── DanceTrack
│   │   ├── dancetrack_url.xlsx
│   │   ├── test
│   │   │   ├── dancetrack0003
│   │   │   └── ...
│   │   ├── train
│   │   │   ├── dancetrack0001
│   │   │   └── ...
│   │   └── val
│   │       ├── dancetrack0004
│   │       └── ...
│   ├── detections
│   │   ├── TAO_Co-DETR_test.json
│   │   └── ...
│   ├── SportsMOT
│   │   ├── dataset
│   │   │   ├── annotations
│   │   │   ├── test
│   │   │   ├── train
│   │   │   └── val
│   │   └── splits_txt
│   │       ├── basketball.txt
│   │       ├── football.txt
│   │       ├── test.txt
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── volleyball.txt
│   └── TAO
│       ├── annotations
│       │   ├── checksums
│       │   ├── README.md
│       │   ├── tao_test_burst_v1.json
│       │   ├── train_ours_v1.json
│       │   ├── validation_ours_v1.json
│       │   └── ...
│       └── frames
│           ├── test
│           ├── train
│           └── val
└──...
   
```
## Training
Download coco pretrained weight from [here (Deformable DETR + iterative bounding box refinement)](https://drive.google.com/file/d/1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI/view) first.
Then put the downloaded weight into `$AED/pretrained`.
Please make sure you set the right **absolute** path of `--pretrained`, `--mot_path`, `--train_det_path`, and `--val_det_path`.
```bash
# TAO
# e.g. bash ./tools/train_tao.sh configs/tao.args 0
bash ./tools/train_tao.sh [config path] [GPU index]
# SportsMOT
bash ./tools/train_sportsmot.sh [config path] [GPU index]
# DanceTrack
bash ./tools/train_dancetrack.sh [config path] [GPU index]
```
Multi-GPU is not supported yet.
After training, the results are saved in `$AED/exps/[dataset name]`
## Inference
Put the downloaded weight into `$AED/pretrained` like:
```
pretrained
├── dancetrack_ckpt_train.pth
├── r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
├── sportsmot_ckpt_train.pth
└── tao_ckpt_train_base.pth
```
start inference
```bash
# TAO
# e.g. bash tools/inference_tao.sh pretrained/tao_ckpt_train_base.pth configs/tao.args test 0
# Remember to choose the right --val_det_path in the config to specify a detector.
bash tools/inference_tao.sh [checkpoint path] [config path] [split (val / test)] [GPU index]
# SportsMOT
bash tools/inference_sportsmot.sh [checkpoint path] [config path] [split (val / test)] [GPU index]
# DanceTrack
bash tools/inference_dancetrack.sh [checkpoint path] [config path] [split (val / test)] [GPU index]
```
After inference, the results are saved in `$AED/exps/[dataset name]_infer_results`.

For SportsMOT and DanceTrack, you can upload the results to [codalab](https://codalab.lisn.upsaclay.fr/) to get the final score.
## Evaluations (Optional)
### TAO
```bash
# e.g. python tools/eval_tao.py --ann_file ./data/validation_ours_v1.json --res_path exps/tao_infer_results/infer1/inference_result/infer_result.json
python tools/eval_tao.py --ann_file path_to_annotations --res_path path_to_results
```
### SportsMOT & DanceTrack
You need to use [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation (val set).
```bash
# move to the path of AED
cd $AED
git clone https://github.com/JonathonLuiten/TrackEval.git
# e.g. 
# bash eval_sportsmot.sh \
# ./data/SportsMOT/dataset/val \
# ./data/SportsMOT/splits_txt/val.txt \
# exps/sportsmot_infer_results/infer1/result_txt \
# exps/sportsmot_infer_results/infer1
bash ./tools/eval_dancetrack.sh [GT path] [split txt path] [result_txt path] [output path]
bash ./tools/eval_sportsMOT.sh [GT path] [split txt path] [result_txt path] [output path]
```
Split txt of DanceTrack can be found in [here](https://github.com/DanceTrack/DanceTrack/tree/main/dancetrack).
## Acknowledgements & Citation
We would like to express our sincere gratitude to the following works (in no particular order): [MOTRv2](https://github.com/megvii-research/MOTRv2), [OVTrack](https://github.com/SysCV/ovtrack), [QDTrack](https://github.com/SysCV/qdtrack), [RegionCLIP](https://github.com/microsoft/RegionCLIP), [Co-DETR](https://github.com/Sense-X/Co-DETR),[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).

If you find this work useful, please consider to cite our paper:
```
@article{fang2024associate,
  title={Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown},
  author={Fang, Zimeng and Liang, Chao and Zhou, Xue and Zhu, Shuyuan and Li, Xi},
  journal={arXiv preprint arXiv:2409.09293},
  year={2024}
}
```