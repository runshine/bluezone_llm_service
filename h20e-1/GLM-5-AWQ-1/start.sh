#!/bin/bash

#apt update && apt install -y git
# pip install mooncake-transfer-engine
#pip install transformers -U
#pip install --upgrade transformers
#pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
python3 /data/glm4_mtp_fix.py
#pip install -i https://test.pypi.org/simple/ lmcache==0.4.4.dev5
export PYTHORCH_ALLOC_CONF=expandable_segments:True,max_non_split_rounding_mb:256
export VLLM_USE_V1=1
#export LMCACHE_CONFIG_FILE=/data/lmcache_config.yaml
#export PYTHONHASHSEED=0
echo $CUDA_VISIBLE_DEVICES

vllm serve /data/models/tclf90/GLM-5-AWQ \
    --served-model-name 'zai-org/GLM-5' \
    --max-num-seqs 64 \
    --max-model-len 202752 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --enable-prefix-caching \
    --enable-expert-parallel \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --speculative-config '{"method":"mtp", "num_speculative_tokens":1}'  
#    --kv-transfer-config \
#    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both" }'

