#!/bin/bash

cd /opt/workspace
source /opt/venv/bin/activate
pip install mooncake-transfer-engine-cuda13 
mooncake_master --port 50063 --enable_http_metadata_server=true --http_metadata_server_port=8092
