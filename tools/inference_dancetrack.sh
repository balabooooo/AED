#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

set -x
set -o pipefail

rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

cleanup() {
  rmpyc
  echo " ...Done"
}

trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=$4

OUTPUT_BASE="exps/dancetrack_infer_results"

mkdir $OUTPUT_BASE

for INFER in $(seq 100); do
  ls $OUTPUT_BASE | grep infer$INFER && continue
  OUTPUT_DIR=$OUTPUT_BASE/infer$INFER$3
  mkdir $OUTPUT_DIR && break
done

args=$(cat $2)

rmpyc && cp -r models datasets util main.py engine.py inference_dancetrack.py $2 $OUTPUT_DIR

ckpt_path=$(realpath "$1")

pushd $OUTPUT_DIR

echo "using checkpoint: $ckpt_path"

python3 inference_dancetrack.py ${args} --exp_name result_txt --resume $ckpt_path --split $3