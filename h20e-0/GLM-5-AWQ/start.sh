#!/bin/bash

apt update && apt install -y git
# pip install mooncake-transfer-engine
pip install transformers -U
#pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
python3 /data/glm4_mtp_fix.py

export PYTHORCH_ALLOC_CONF=expandable_segments:True,max_non_split_rounding_mb:256
export VLLM_USE_V1=1


# mooncake_http_metadata_server &
# mooncake_master &

vllm serve /data/models/tclf90/GLM-5-AWQ \
    --served-model-name 'zai-org/GLM-5' \
    --max-num-seqs 16 \
    --max-model-len 202752 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.80 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --tensor-parallel-size 8 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens":1}' 
    #--kv-transfer-config \
    #'{"kv_connector":"MooncakeConnector", "kv_role":"kv_both"}'

#-cc.cudagraph_mode=PIECEWISE \
