#!/bin/bash

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_USE_MODELSCOPE=false
export MODELSCOPE_DISABLE_HF_PATCH=1
export NVIDIA_VISIBLE_DEVICES=all
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# apt update && apt install -y git
# pip install git+https://github.com/huggingface/transformers.git

# pip install mooncake-transfer-engine

vllm serve /data/models/google/gemma-4-31B \
    --served-model-name 'google/gemma-4-31B' \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2\
    --enable-auto-tool-choice \
    --reasoning-parser gemma4 \
    --tool-call-parser gemma4 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --max-model-len 268288