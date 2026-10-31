#!/bin/bash

cd /opt/workspace
source /opt/venv/bin/activate
pip install mooncake-transfer-engine-cuda13 
python mooncake_connector_proxy.py \
    --prefill http://172.31.29.23:7100 9100 \
    --prefill http://172.31.29.23:7101 9101 \
    --decode http://172.31.29.24:7300 \
    --decode http://172.31.29.24:7301 \
    --host 0.0.0.0 \
    --port 8077