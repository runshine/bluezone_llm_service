#!/bin/bash

# Only the proxy-owned external web client reads these variables. PD backend
# and Redis clients explicitly remain direct and are not affected by them.
export CODEX_HTTP_PROXY=""
export CODEX_HTTPS_PROXY=""
export CODEX_WEB_FETCH_TIMEOUT_SECONDS="${CODEX_WEB_FETCH_TIMEOUT_SECONDS:-20}"
export CODEX_WEB_MAX_RESULTS="${CODEX_WEB_MAX_RESULTS:-5}"
export CODEX_WEB_MAX_REDIRECTS="${CODEX_WEB_MAX_REDIRECTS:-5}"
export CODEX_MAX_TOOL_ROUNDS="${CODEX_MAX_TOOL_ROUNDS:-4}"
# Responses state is persisted by default so a returned resp_* can be used as
# previous_response_id on the next request. Clients may still set store=false.
export RESPONSES_DEFAULT_STORE="${RESPONSES_DEFAULT_STORE:-true}"
# Set INSTANCE_API_KEY in the environment to enable instance administration.

python3 load_balance_proxy_server.py \
    --port 8077 \
    --host 0.0.0.0 \
    --model-path /nvme/Model/weights \
    --backend-model-name deepseek-v4.1-flash \
    --multimodal-tokenize-timeout-seconds 30 \
    --multimodal-tokenize-max-concurrency 8 \
    --prefiller-max-model-len 1048000 \
    --decoder-max-model-len 1048576 \
    --max-decoder-start-protocol-recompute-attempts 1 \
    --enable-decoder-start-protocol-retry-prompt \
    --healthcheck-success-ttl-seconds 60 \
    --responses-sse-heartbeat-seconds 10 \
    --decoder-first-response-observe-seconds 30 \
    --decoder-first-response-log-interval-seconds 60 \
    --pd-first-output-timeout-seconds 480 \
    --decoder-stream-idle-timeout-seconds 480 \
    --prefiller-hosts \
       172.31.29.23 \
       172.31.29.23 \
    --prefiller-ports \
       7100 7101\
    --decoder-hosts \
      172.31.29.24 \
      172.31.29.24 \
    --decoder-ports \
      7300 7301
