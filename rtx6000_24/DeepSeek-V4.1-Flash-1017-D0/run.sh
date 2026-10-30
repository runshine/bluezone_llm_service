#!/bin/bash
cd /opt/workspace

source /opt/venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export VIRTUAL_ENV=/opt/venv

# The bundled vLLM build has a mixed-KV-cache compatibility mismatch: the
# hybrid cache path passes allow_prefix_caching when promoting MLA specs.
VLLM_KV_SPEC=/opt/venv/lib/python3.12/site-packages/vllm/v1/kv_cache_interface.py
if ! grep -q '# bluezone mixed-KV compatibility' "$VLLM_KV_SPEC"; then
  sed -i '/^class MLAAttentionSpec(FullAttentionSpec):$/a\    # bluezone mixed-KV compatibility\n    allow_prefix_caching: bool = True' "$VLLM_KV_SPEC"
fi

VLLM_KV_UTILS=/opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py
sed -i 's/kv_cache_spec.update(_promote_local_kv_cache_specs(kv_cache_spec))/pass  # bluezone: keep mixed KV cache specs separate/' "$VLLM_KV_UTILS"

VLLM_WORKER_UTILS=/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/utils.py
sed -i 's/raise ValueError(f"No common block size for {kv_manager_block_size}. ")/return kv_manager_block_size  # bluezone: Mooncake mixed-backend fallback/' "$VLLM_WORKER_UTILS"

export TRITON_PTXAS_BLACKWELL_PATH="$(
  "$VIRTUAL_ENV/bin/python" -c \
    'from pathlib import Path; import triton; print(Path(triton.__file__).parent / "backends/nvidia/bin/ptxas-blackwell")'
)"
"$TRITON_PTXAS_BLACKWELL_PATH" --version
PREFILL_NODE_IP="172.31.29.23"

python3 launch_online_dp.py \
    --dp-size 1 \
    --tp-size 4 \
    --dp-size-local 1 \
    --dp-rank-start 0 \
    --dp-address "$PREFILL_NODE_IP" \
    --dp-rpc-port 12321 \
    --vllm-start-port 7300
