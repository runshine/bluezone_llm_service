#!/bin/bash

apt update && apt install -y git
#pip install git+https://github.com/huggingface/transformers.git
pip install transformers -U
#pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
python3 /data/glm4_mtp_fix.py
# vllm serve /data/models/tclf90/GLM-5-AWQ \
#     --served-model-name 'zai-org/GLM-5' \
#     --max-num-seqs 16 \
#     --max-model-len 196608 \
#     --gpu-memory-utilization 0.95 \
#     --max-num-batched-tokens 1024 \
#     --enable-chunked-prefill \
#     --tensor-parallel-size 4 \
#     --enable-expert-parallel \
#     --enable-auto-tool-choice \
#     --tool-call-parser glm47 \
#     --reasoning-parser glm45 \
#     --trust-remote-code \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --kv-cache-dtype fp8_ds_mla

#vllm serve /data/models/tclf90/GLM-5-AWQ \
#    --served-model-name 'zai-org/GLM-5' \
#    --max-num-seqs 64 \
#    --max-model-len 196608 \
#    --gpu-memory-utilization 0.95 \
#    --tensor-parallel-size 4 \
#    --enable-expert-parallel \
#    --enable-auto-tool-choice \
#    --tool-call-parser glm47 \
#    --reasoning-parser glm45 \
#    --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
#    --trust-remote-code --host 0.0.0.0 --port 8000

export PYTHORCH_ALLOC_CONF=expandable_segments:True

vllm serve /data/models/tclf90/GLM-5-AWQ \
    --served-model-name 'zai-org/GLM-5' \
    --max-num-seqs 16 \
    --max-model-len 196608 \
    --enable-chunked-prefill \
    --enable-expert-parallel \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens":1}'

#-cc.cudagraph_mode=PIECEWISE \
