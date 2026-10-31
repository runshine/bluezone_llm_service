#!/bin/bash

cd /opt/workspace
source /opt/venv/bin/activate
pip install agentic-api
agentic-api serve --host 0.0.0.0 --port 8078 --vllm-base-url http://172.31.29.23:8077
