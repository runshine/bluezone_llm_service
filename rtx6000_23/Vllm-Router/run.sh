#!/bin/bash

cd /opt/workspace
source /opt/venv/bin/activate
pip install vllm-router  

vllm-router \
    --policy consistent_hash \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill http://172.31.29.23:7100 9100 \
    --prefill http://172.31.29.23:7101 9101 \
    --decode  http://172.31.29.24:7300 \
    --decode  http://172.31.29.24:7301 \
    --host 0.0.0.0 \
    --port 8077 \
    --intra-node-data-parallel-size 16
