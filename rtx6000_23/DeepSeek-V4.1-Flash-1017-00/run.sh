#!/bin/bash

source /opt/venv/bin/activate
export VIRTUAL_ENV=/opt/venv
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export FLASHINFER_CUDA_ARCH_LIST=12.0
export VLLM_USE_V2_MODEL_RUNNER=1
export TRITON_PTXAS_BLACKWELL_PATH="$(
  "$VIRTUAL_ENV/bin/python" -c \
    'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas-blackwell")'
)"
"$TRITON_PTXAS_BLACKWELL_PATH" --version

vllm serve /data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/ \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --distributed-executor-backend mp \
  --enable-expert-parallel \
  --moe-backend auto \
  --kv-cache-dtype fp8 \
  --block-size 128 \
  --max-model-len auto \
  --max-num-seqs 20 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --engram-config '{"cpu_offload":true}' \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v41 \
  --hf-overrides '{"ced_prefill":true}' \
  --speculative-config \
  '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --compilation-config '{"cudagraph_mode":"FULL"}'