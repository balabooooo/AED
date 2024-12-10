#!/usr/bin/env bash

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

export CUDA_VISIBLE_DEVICES=$2

# uncomment the following line if encountering the error 
# "CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`"
# unset LD_LIBRARY_PATH

xargs python3 demo/demo.py < $1