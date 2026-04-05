#!/bin/bash

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_USE_MODELSCOPE=false
export MODELSCOPE_DISABLE_HF_PATCH=1

# apt update && apt install -y git
# pip install git+https://github.com/huggingface/transformers.git

# pip install mooncake-transfer-engine

vllm serve /data/models/Qwen/Qwen3.5-9B \
    --served-model-name 'Qwen/Qwen3.5-35B' \
    --max-num-seqs 32 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000