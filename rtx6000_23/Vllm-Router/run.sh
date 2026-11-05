#!/bin/bash

cd /opt/workspace
source /opt/venv/bin/activate
pip install vllm-router  

# P0/P1 each launch with --dp-size 1; do not synthesize phantom @rank workers.
vllm-router \
    --policy round_robin \
    --vllm-pd-disaggregation \
    --kv-connector mooncake \
    --prefill http://172.31.29.23:7100 9100 \
    --prefill http://172.31.29.23:7101 9101 \
    --prefill http://172.31.29.31:7100 9100 \
    --prefill http://172.31.29.31:7101 9101 \
    --decode  http://172.31.29.24:7300 \
    --decode  http://172.31.29.24:7301 \
    --decode  http://172.31.29.32:7300 \
    --decode  http://172.31.29.32:7301 \
    --decode  http://172.31.29.34:7300 \
    --decode  http://172.31.29.34:7301 \
    --decode  http://172.31.29.35:7300 \
    --decode  http://172.31.29.35:7301 \
    --host 0.0.0.0 \
    --port 8077 \
    --intra-node-data-parallel-size 1
