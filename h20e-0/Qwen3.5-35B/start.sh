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

vllm serve /data/models/Qwen3.5-35B-A3B-FP8 \
    --served-model-name 'Qwen/Qwen3.5-35B' \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder  \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder  \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
    --max-model-len 1010000