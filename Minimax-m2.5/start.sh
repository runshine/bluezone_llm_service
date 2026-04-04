#!/bin/bash

vllm serve /data/models/MiniMax/MiniMax-M2.5/ \
    --served-model-name 'MiniMax/MiniMax-M2.5' \
    --max-num-seqs 32 \
    --max-model-len 163804  \
    --gpu-memory-utilization 0.95 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    --enable-auto-tool-choice 
