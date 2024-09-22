#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
set -o pipefail

ckpt_path=$(realpath "$1")
export CUDA_VISIBLE_DEVICES=$4

# clean up *.pyc files
rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

cleanup() {
  rmpyc
  echo " ...Done"
}

trap cleanup EXIT

OUTPUT_BASE="exps/tao_infer_results"

mkdir -p $OUTPUT_BASE

for INFER in $(seq 100); do
  ls $OUTPUT_BASE | grep infer$INFER && continue
  OUTPUT_DIR=$OUTPUT_BASE/infer$INFER$3
  mkdir $OUTPUT_DIR && break
done

args=$(cat $2)

rmpyc && cp -r models datasets util main.py engine.py inference_tao.py $2 $OUTPUT_DIR

pushd $OUTPUT_DIR

python3 inference_tao.py ${args} --exp_name inference_result --resume $ckpt_path --split $3