#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Modified from MOTRv2 (https://github.com/megvii-research/MOTRv2)
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

set -x
set -o pipefail

export CUDA_VISIBLE_DEVICES=$2

IFS=',' read -ra gpu_list <<< "$2"
gpu_num=${#gpu_list[@]}

echo "Using $gpu_num GPUs: $2"

PY_ARGS=${@:2}

set -o pipefail

OUTPUT_BASE=$(echo $1 | sed -e "s/configs/exps/g" | sed -e "s/.args$//g")
mkdir -p $OUTPUT_BASE

for RUN in $(seq 100); do
  ls $OUTPUT_BASE | grep run$RUN && continue
  OUTPUT_DIR=$OUTPUT_BASE/run$RUN
  mkdir $OUTPUT_DIR && break
done

rmpyc() {
  rm -rf $(find -name __pycache__)
  rm -rf $(find -name "*.pyc")
}

echo "Backing up to log dir: $OUTPUT_DIR"
rmpyc && cp -r models datasets util main.py engine.py inference_tao.py $1 $OUTPUT_DIR
echo " ...Done"

cleanup() {
  rmpyc
  echo " ...Done"
}

args=$(cat $1)

pushd $OUTPUT_DIR

mkdir vis

trap cleanup EXIT

echo $PY_ARGS > desc
echo " ...Done"

python main.py ${args} --output_dir . |& tee -a output.log