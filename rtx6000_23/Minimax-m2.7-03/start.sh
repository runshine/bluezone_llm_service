#!/bin/bash

pip install pytest
export TORCH_ALLOW_TF32=1
export TORCH_BLAS_PREFER_HIPBLASLT=0   # 非必需，但有助于 NVIDIA 后端

vllm serve /data/models/nv-community/MiniMax-M2.7-NVFP4 \
    --served-model-name 'MiniMax/MiniMax-M2.5' \
    --max-num-seqs 32 \
    --max-model-len 163804  \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice 
    
        # --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
