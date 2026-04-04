#!/bin/bash

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_MOE_FP16=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=4
export VLLM_USE_MODELSCOPE=false
export MODELSCOPE_DISABLE_HF_PATCH=1
export NVIDIA_VISIBLE_DEVICES=all

apt update && apt install -y git
pip install git+https://github.com/huggingface/transformers.git

#pip install mooncake-transfer-engine
# pip install --upgrade transformers

vllm serve /data/models/GLM-4.7-Flash \
    --served-model-name 'zai-org/GLM-4.7-Flash' \
    --max-num-seqs 32 \
    --max-model-len 196608 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
    --trust-remote-code --host 0.0.0.0 --port 8000
