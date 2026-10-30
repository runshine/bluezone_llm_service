#!/usr/bin/env bash


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export GLOO_SOCKET_IFNAME="ens48f0"
export TP_SOCKET_IFNAME="ens48f0"
export FLASHINFER_CUDA_ARCH_LIST=12.0
export VLLM_USE_V2_MODEL_RUNNER=1
export CUDA_VISIBLE_DEVICES=$1
export MOONCAKE_CONFIG_PATH=/opt/workspace/mooncake_config.json 
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

pip install mooncake-transfer-engine-cuda13 
#pip install nixl

vllm serve /data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017/ \
    --served-model-name deepseek-v4.1-flash \
    --host 0.0.0.0 \
    --port $2 \
    --trust-remote-code \
    --tensor-parallel-size $7 \
    --distributed-executor-backend mp \
    --enable-expert-parallel \
    --moe-backend auto \
    --kv-cache-dtype fp8 \
    --block-size 128 \
    --prefix-match-unit 8 \
    --seed 1024 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --disable-hybrid-kv-cache-manager \
    --load-format safetensors \
    --speculative-config \
     '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
    --trust-remote-code \
    --enable-prefix-caching \
    --engram-config '{"cpu_offload":true}' \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --gpu-memory-utilization 0.9 \
    --hf-overrides '{"ced_prefill":false}' \
    --compilation-config '{"cudagraph_mode":"FULL"}'\
    --kv-transfer-config '{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_producer",
    "kv_connector_extra_config": {
      "device_name": "mlx5_4,mlx5_5",
      "connectors": [
        {
          "kv_connector": "MooncakeConnector",
          "kv_role": "kv_producer"
        },
        {
          "kv_connector": "MooncakeStoreConnector",
          "kv_role": "kv_both"
        }
      ]
    }
  }'
