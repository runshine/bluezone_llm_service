# Adapted from https://github.com/vllm-project/vllm/tests/v1/kv_connector/nixl_integration/toy_proxy_server.py

# SPDX-License-Identifier: Apache-2.0
#
# Tutorial: Using the Load Balance Proxy Server Example
#
# This proxy server is designed to distribute requests between multiple
# "prefiller" and "decoder" backend servers for large language model inference.
# It is useful for scaling out inference workloads and balancing load across
# multiple backend instances.
#
# Features:
# - Load balances requests to multiple prefiller and decoder servers.
# - Supports OpenAI-compatible /v1/completions and /v1/chat/completions endpoints.
# - Streams responses from backend servers to clients.
#
# Prerequisites:
# - Python 3.10+
# - Install dependencies:
#     pip install fastapi<0.124.0 httpx uvicorn vllm transformers jsonschema
#
# Step 1: Start Your Backend Servers
# ----------------------------------
# You need to have at least one prefiller and one decoder backend running.
# These can be mock servers or actual vLLM servers.
#
# For testing, you can use the provided mock server:
#
#   vllm serve --host 0.0.0.0 --port 8100 ... # Prefiller 1
#   vllm serve --host 0.0.0.0 --port 8101 ... # Prefiller 2
#   vllm serve --host 0.0.0.0 --port 8200 ... # Decoder 1
#   vllm serve --host 0.0.0.0 --port 8201 ... # Decoder 2
#
# Step 2: Start the Proxy Server
# ------------------------------
# Run the proxy server, specifying the host/port for each prefiller and decoder:
#
#   python load_balance_proxy_server_example.py \
#     --host 0.0.0.0 --port 9000 \
#     --prefiller-hosts 127.0.0.1 127.0.0.1 \
#     --prefiller-ports 8100 8101 \
#     --decoder-hosts 127.0.0.1 127.0.0.1 \
#     --decoder-ports 8200 8201
#
# This will start the proxy on port 9000, load balancing between two prefiller
# and two decoder servers.
#
# Step 3: Send a Request to the Proxy
# -----------------------------------
# You can now send OpenAI-compatible requests to the proxy. For example:
#
#   curl -X POST http://localhost:9000/v1/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "prompt": "The quick brown fox jumps over the lazy dog",
#           "max_tokens": 16
#         }'
#
# Or for chat completions:
#
#   curl -X POST http://localhost:9000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#           "model": "your-model",
#           "messages": [{"role": "user", "content": "Hello!"}],
#           "max_tokens": 16
#         }'
#
# Step 4: Health Check
# --------------------
# To check if the proxy is running and see how many backend instances are
# connected, use:
#
#   curl http://localhost:9000/healthcheck
#
# This will return a JSON object with the status and the number of prefiller
# and decoder instances.
#
# Step 5: Add or Remove Prefiller or Decoder Instances (Optional)
# ---------------------------------------------------------------
# You can add or remove prefiller or decoder instances after the proxy is started.
# For example, add 2 prefiller instances:
#
#   curl -X POST http://localhost:9000/instances/add \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "prefill",
#           "instances": ["127.0.0.1:8102", "127.0.0.1:8103"]
#         }'
#
# or remove 1 decoder instance:
#
#   curl -X POST http://localhost:9000/instances/remove \
#     -H "Content-Type: application/json" \
#     -d '{
#           "type": "decode",
#           "instances": "127.0.0.1:8201"
#         }'
#
# This will return a JSON object with the adding or removing info
# and the current prefiller and decoder instances.
#
# When adding instances, if the instances are not started,
# the proxy will wait and try until the instances to be started
# or exceeding the number of attempts
#
# Notes:
# - You can scale the number of prefiller and decoder servers as needed.
# - The proxy will round-robin requests to balance load.
# - For production, ensure your backend servers are robust and secure.
#
# For more details, see the code and comments in this file.

import argparse
import asyncio
import base64
import copy
import contextvars
import hashlib
import heapq
import inspect
import ipaddress
import io
import json
import os
import re
import fcntl
import shutil
import socket
import sys
import threading
import time
import uuid
import secrets
from datetime import datetime, timedelta, timezone
from collections import Counter, OrderedDict, deque
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable
from urllib.parse import unquote, urlparse
from zoneinfo import ZoneInfo

import httpx
from codex_exec_normalization import (
    ExecNormalizationError, normalize_exec, validate_parsers,
)
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

ENCODING_DIR = os.path.join(os.path.dirname(__file__), "encoding")
if ENCODING_DIR not in sys.path:
    sys.path.insert(0, ENCODING_DIR)

try:
    from encoding_dsv41 import (
        encode_messages as encode_dsv4_messages,
    )
except ImportError:
    encode_dsv4_messages = None

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

# Add uvloop for faster event loop if available
try:
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


@dataclass
class InstanceType:
    PREFILL: str = "prefill"
    DECODE: str = "decode"


TAINT_PRIORITY = 1e15
ACTIVE_REQUEST_WEIGHT = 1.0
DECODER_OUTPUT_TOKEN_WEIGHT = 4.0
DECODER_REQUEST_WEIGHT = 4096.0
SAFE_CONTEXT_BUFFER_TOKENS = 4096
DEFAULT_CERTAIN_KV_LEAK_LOG_PATH = "certain_kv_leaks.jsonl"
DEFAULT_SUSPECTED_KV_LEAK_LOG_PATH = "suspected_kv_leaks.jsonl"
RESPONSES_REDIS_URL = os.environ.get("RESPONSES_REDIS_URL", "").strip()
RESPONSES_STATE_TTL_SECONDS = int(os.environ.get("RESPONSES_STATE_TTL_SECONDS", "86400"))
RESPONSES_KEY_PREFIX = os.environ.get("RESPONSES_KEY_PREFIX", "codex:responses:")
RESPONSES_WIRE_TRACE = os.environ.get("RESPONSES_WIRE_TRACE", "").lower() in {"1", "true", "yes"}
RESPONSES_SSE_TRACE = os.environ.get("RESPONSES_SSE_TRACE", "").lower() in {"1", "true", "yes"}
RESPONSES_WIRE_TRACE_DIR = os.environ.get(
    "RESPONSES_WIRE_TRACE_DIR",
    os.path.join(os.path.dirname(__file__), "trace"),
)
# Request audit is deliberately independent from the Responses adapter.  The
# latter is reached only after a request has been parsed and converted, which
# made malformed and rejected requests invisible during incident analysis.
REQUEST_AUDIT_TRACE = os.environ.get(
    "REQUEST_AUDIT_TRACE", "1" if RESPONSES_WIRE_TRACE else ""
).lower() in {"1", "true", "yes"}
REQUEST_AUDIT_TRACE_DIR = os.environ.get(
    "REQUEST_AUDIT_TRACE_DIR", RESPONSES_WIRE_TRACE_DIR
)
REQUEST_AUDIT_RETENTION_DAYS = max(1, int(os.environ.get("REQUEST_AUDIT_RETENTION_DAYS", "7")))
REQUEST_AUDIT_MAX_BYTES = max(0, int(os.environ.get("REQUEST_AUDIT_MAX_BYTES", str(200 * 1024**3))))
REQUEST_AUDIT_PAYLOAD_MODE = os.environ.get("REQUEST_AUDIT_PAYLOAD_MODE", "all").strip().lower()
REQUEST_AUDIT_SERVICE_ID = os.environ.get("REQUEST_AUDIT_SERVICE_ID", "loadbalance-unknown").strip()
REQUEST_AUDIT_LISTENER_PORT = os.environ.get("REQUEST_AUDIT_LISTENER_PORT", "unknown").strip()
REQUEST_AUDIT_TIMEZONE = os.environ.get("REQUEST_AUDIT_TIMEZONE", "Asia/Shanghai").strip()
CHAT_STREAM_GUARD_ENABLED = os.environ.get("CHAT_STREAM_GUARD_ENABLED", "0").lower() in {"1", "true", "yes"}
CHAT_STREAM_GUARD_TRACE_ENABLED = os.environ.get("CHAT_STREAM_GUARD_TRACE_ENABLED", "1").lower() in {"1", "true", "yes"}
CHAT_STREAM_GUARD_TRACE_MAX_BYTES = max(1024, int(os.environ.get("CHAT_STREAM_GUARD_TRACE_MAX_BYTES", str(50 * 1024 * 1024))))
DECODER_TAINT_ENABLED = os.environ.get("DECODER_TAINT_ENABLED", "1").lower() in {"1", "true", "yes"}
DECODER_TAINT_FAILURE_THRESHOLD = max(1, int(os.environ.get("DECODER_TAINT_FAILURE_THRESHOLD", "3")))
DECODER_TAINT_FAILURE_WINDOW_SECONDS = max(1.0, float(os.environ.get("DECODER_TAINT_FAILURE_WINDOW_SECONDS", "60")))
DECODER_TAINT_HTTP_5XX_THRESHOLD = max(1, int(os.environ.get("DECODER_TAINT_HTTP_5XX_THRESHOLD", "2")))
DECODER_TAINT_RESET_ON_SUCCESS = os.environ.get("DECODER_TAINT_RESET_ON_SUCCESS", "1").lower() in {"1", "true", "yes"}
try:
    _REQUEST_AUDIT_TZ = ZoneInfo(REQUEST_AUDIT_TIMEZONE)
except Exception:
    # Some inference images omit both /usr/share/zoneinfo and the optional
    # Python tzdata package.  Shanghai has no DST, so UTC+08:00 is an exact,
    # dependency-free fallback and must never prevent the proxy from starting.
    _REQUEST_AUDIT_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")
PREFILL_FAILURE_CAPTURE = os.environ.get("PREFILL_FAILURE_CAPTURE", "").lower() in {
    "1", "true", "yes"
}
PREFILL_FAILURE_CAPTURE_DIR = os.environ.get(
    "PREFILL_FAILURE_CAPTURE_DIR",
    os.path.join(os.path.dirname(__file__), "failure-captures"),
)
CODEX_HTTP_PROXY = os.environ.get("CODEX_HTTP_PROXY", "").strip() or None
CODEX_HTTPS_PROXY = os.environ.get("CODEX_HTTPS_PROXY", "").strip() or CODEX_HTTP_PROXY
CODEX_WEB_SEARCH_URL = os.environ.get("CODEX_WEB_SEARCH_URL", "").strip() or None
CODEX_WEB_SEARCH_API_KEY = os.environ.get("CODEX_WEB_SEARCH_API_KEY", "").strip() or None
CODEX_WEB_FETCH_TIMEOUT_SECONDS = float(os.environ.get("CODEX_WEB_FETCH_TIMEOUT_SECONDS", "20"))
CODEX_WEB_MAX_RESULTS = max(1, min(int(os.environ.get("CODEX_WEB_MAX_RESULTS", "5")), 20))
CODEX_WEB_MAX_REDIRECTS = max(0, min(int(os.environ.get("CODEX_WEB_MAX_REDIRECTS", "5")), 10))
CODEX_WEB_MAX_BODY_BYTES = max(1024, min(int(os.environ.get("CODEX_WEB_MAX_BODY_BYTES", "1048576")), 10 * 1024 * 1024))
CODEX_MAX_TOOL_ROUNDS = max(1, min(int(os.environ.get("CODEX_MAX_TOOL_ROUNDS", "4")), 16))
RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS = max(
    64, min(int(os.environ.get("RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS", "16384")), 32768)
)
RESPONSES_MIN_OUTPUT_TOKENS = 16384
NONSTREAM_DECODER_TIMEOUT_SECONDS = 1200.0
INSTANCE_API_KEY = os.environ.get("INSTANCE_API_KEY", "").strip()
PREFILL_CONNECTION_MODE = os.environ.get("PREFILL_CONNECTION_MODE", "close").strip().lower()
if PREFILL_CONNECTION_MODE not in {"close", "keepalive"}:
    PREFILL_CONNECTION_MODE = "close"
PREFILL_MAX_CONNECTIONS = max(1, int(os.environ.get("PREFILL_MAX_CONNECTIONS", "32")))
PREFILL_MAX_KEEPALIVE_CONNECTIONS = max(
    0, int(os.environ.get("PREFILL_MAX_KEEPALIVE_CONNECTIONS", "0"))
)

_active_wire_trace = contextvars.ContextVar("active_wire_trace", default=None)
_active_request_audit = contextvars.ContextVar("active_request_audit", default=None)
_active_client_request_id = contextvars.ContextVar("active_client_request_id", default=None)
_active_failure_capture_source = contextvars.ContextVar(
    "active_failure_capture_source", default=None
)
_active_failure_capture_context = contextvars.ContextVar(
    "active_failure_capture_context", default=None
)


class LeakLevel:
    NONE = "none"
    SUSPECTED = "suspected"
    CERTAIN = "certain"


class ClientDisconnectedError(RuntimeError):
    pass


class DecoderStreamIncompleteError(RuntimeError):
    """Decoder closed its SSE response before emitting the required terminator."""


class DecoderStartProtocolError(RuntimeError):
    """Decoder emitted its internal start-of-sentence marker as reasoning output."""


class PdFirstOutputTimeoutError(RuntimeError):
    """A PD attempt exceeded its first-valid-output deadline."""


class DecoderStreamIdleTimeoutError(RuntimeError):
    """A Decoder stopped producing meaningful output before its stream completed."""


DECODER_START_PROTOCOL_MARKERS = frozenset({
    "begin\u2581of\u2581sentence",
    "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
})
DECODER_START_PROTOCOL_RETRY_PROMPT = "\n\u8bf7\u91cd\u8bd5"


def decoder_start_protocol_marker_variant(marker_text: str) -> str:
    return "wrapped" if marker_text.startswith("<") else "unwrapped"


class DetachedTaskManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._tasks: set[asyncio.Task] = set()

    def track(self, task: asyncio.Task) -> None:
        with self._lock:
            self._tasks.add(task)
        task.add_done_callback(self._discard)

    def _discard(self, task: asyncio.Task) -> None:
        with self._lock:
            self._tasks.discard(task)

    async def shutdown(self, timeout: float = 5.0) -> None:
        with self._lock:
            tasks = list(self._tasks)
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/v1"
        try:
            ip = ipaddress.ip_address(self.host)
            if isinstance(ip, ipaddress.IPv6Address):
                self.url = f"http://[{host}]:{port}/v1"
        except Exception:
            pass
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(max_connections=100000, max_keepalive_connections=100000),
        )
        self.prefill_client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(
                max_connections=PREFILL_MAX_CONNECTIONS,
                max_keepalive_connections=PREFILL_MAX_KEEPALIVE_CONNECTIONS,
            ),
        )
        self.active_tokens = 0
        self.active_kv_cache = 0  # Only for prefiller
        self.active_requests = 0  # Number of active requests
        self.period_assigned_requests = 0  # Decoder assignments in the current stats period
        self.last_success_at = 0.0
        self.last_success_kind = ""
        # Removed individual server lock - will use global locks instead

    def __eq__(self, other):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        other_host = other.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return self_host == other_host and str(self.port) == str(other.port)

    def __hash__(self):
        self_host = self.host.replace("localhost", "0.0.0.0").replace("127.0.0.1", "0.0.0.0")
        return hash((self_host, str(self.port)))

    def __repr__(self):
        return f"{self.host}:{self.port}"


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        # Fix #2: Thread-safe lock for all shared state mutations
        # RLock allows reentrant acquisition from the same thread
        self._lock = threading.RLock()
        self._heap_counter = 0  # Monotonic counter for heap tiebreaking
        self._tokenize_round_robin = {"prefill": 0, "decode": 0}
        self.request_num = 0
        self.tainted_prefillers: list[ServerState] = []
        self.tainted_decoders: list[ServerState] = []
        self.pending_remove_prefillers: set[ServerState] = set()
        self.pending_remove_decoders: set[ServerState] = set()
        self.decoder_failure_state: dict[ServerState, dict[str, Any]] = {}
        self.decoder_failure_total: Counter[tuple[str, str]] = Counter()
        self.decoder_taint_suppressed_total: Counter[tuple[str, str]] = Counter()
        self.decoder_taint_total: Counter[tuple[str, str]] = Counter()
        self.decoder_recovery_total: Counter[str] = Counter()
        self.decoder_recovery_probe_total: Counter[tuple[str, str]] = Counter()

        self.prefillers: list[ServerState] = [ServerState(h, p) for h, p in prefiller_instances]
        self.decoders: list[ServerState] = [ServerState(h, p) for h, p in decoder_instances]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()

        # Initialize priority queues for efficient server selection
        # Each entry is (priority_score, counter, server_reference)
        # counter: monotonically increasing tiebreaker (avoids comparing ServerState objects)
        # Lower priority score = higher priority (less loaded)
        self.prefiller_heap = []
        self.decoder_heap = []
        for server in self.prefillers:
            self._heap_counter += 1
            self.prefiller_heap.append((0.0, self._heap_counter, server))
        for server in self.decoders:
            self._heap_counter += 1
            self.decoder_heap.append(((0, 0), self._heap_counter, server))
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)
        self.node_listener = NodeListener(self)

    def _next_counter(self) -> int:
        """Return next monotonic counter value. Caller must hold self._lock."""
        self._heap_counter += 1
        return self._heap_counter

    def _calc_prefiller_priority(self, server: ServerState) -> float:
        """Calculate priority score for a prefiller. Caller must hold self._lock."""
        if server in self.tainted_prefillers:
            return TAINT_PRIORITY
        return (
            server.active_tokens
            + server.active_kv_cache * 0.1
            + server.active_requests * ACTIVE_REQUEST_WEIGHT
        )

    def _calc_decoder_priority(self, server: ServerState) -> tuple[float, float]:
        """Use active connections first, then assignments in the current stats period."""
        if server in self.tainted_decoders:
            return (TAINT_PRIORITY, TAINT_PRIORITY)
        return (float(server.active_requests), float(server.period_assigned_requests))

    def _update_prefiller_priority(self, server: ServerState):
        """Update the priority of a prefiller server in the heap. Caller must hold self._lock."""
        priority = self._calc_prefiller_priority(server)
        # Remove old entry and add new one (match by server identity, not index)
        self.prefiller_heap = [(p, c, s) for p, c, s in self.prefiller_heap if s is not server]
        heapq.heappush(self.prefiller_heap, (priority, self._next_counter(), server))

    def _update_decoder_priority(self, server: ServerState):
        """Update the priority of a decoder server in the heap. Caller must hold self._lock."""
        priority = self._calc_decoder_priority(server)
        # Remove old entry and add new one (match by server identity, not index)
        self.decoder_heap = [(p, c, s) for p, c, s in self.decoder_heap if s is not server]
        heapq.heappush(self.decoder_heap, (priority, self._next_counter(), server))

    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())

    def select_prefiller(self, token_count) -> ServerState:
        with self._lock:
            if not self.prefiller_heap:
                raise RuntimeError("No prefiller servers available")

            # Skip tainted entries — they must never be selected for new requests
            skipped = []
            while self.prefiller_heap:
                priority, counter, server = heapq.heappop(self.prefiller_heap)
                if server in self.tainted_prefillers:
                    skipped.append((priority, counter, server))
                    continue
                # Found a healthy prefiller, push skipped entries back
                for entry in skipped:
                    heapq.heappush(self.prefiller_heap, entry)
                break
            else:
                # No healthy prefillers found, push skipped back and raise error
                for entry in skipped:
                    heapq.heappush(self.prefiller_heap, entry)
                raise RuntimeError("No healthy prefiller servers available")

            # Update the chosen server
            server.active_tokens += token_count
            server.active_kv_cache += token_count
            server.active_requests += 1

            # Update priority and re-add to heap
            self._update_prefiller_priority(server)

            return server

    def release_prefiller(self, server: ServerState, token_count):
        with self._lock:
            server.active_tokens = max(0, server.active_tokens - token_count)
            server.active_requests = max(0, server.active_requests - 1)
            # Update priority queue after releasing
            if server in self.prefillers:
                self._update_prefiller_priority(server)

    def release_prefiller_kv(self, server: ServerState, token_count):
        with self._lock:
            # Fix #5: prevent active_kv_cache from going negative
            server.active_kv_cache = max(0, server.active_kv_cache - token_count)
            # Update priority queue after releasing
            if server in self.prefillers:
                self._update_prefiller_priority(server)

    def select_decoder(
        self,
        token_count=None,
        excluded_decoders: set[ServerState] | None = None,
    ) -> ServerState:
        with self._lock:
            if not self.decoder_heap:
                raise RuntimeError("No decoder servers available")

            # Keep a request-local fallback for a protocol retry. Exclusion is
            # deliberately not a health judgment and must not affect other requests.
            skipped = []
            fallback = None
            while self.decoder_heap:
                priority, counter, server = heapq.heappop(self.decoder_heap)
                if server in self.tainted_decoders:
                    skipped.append((priority, counter, server))
                    continue
                if excluded_decoders and server in excluded_decoders:
                    if fallback is None:
                        fallback = (priority, counter, server)
                    skipped.append((priority, counter, server))
                    continue
                selected = server
                break
            else:
                if fallback is None:
                    for entry in skipped:
                        heapq.heappush(self.decoder_heap, entry)
                    raise RuntimeError("No healthy decoder servers available")
                selected = fallback[2]
                skipped.remove(fallback)

            for entry in skipped:
                heapq.heappush(self.decoder_heap, entry)

            # Active connections are the primary key. For equal active counts,
            # prefer the node assigned fewer requests in this stats period.
            server.active_requests += 1
            server.period_assigned_requests += 1

            # Update priority and re-add to heap
            self._update_decoder_priority(server)

            return server

    def release_decoder(self, server: ServerState, token_count=None):
        with self._lock:
            server.active_requests = max(0, server.active_requests - 1)
            # Update priority queue after releasing
            if server in self.decoders:
                self._update_decoder_priority(server)

    def reset_decoder_period_assignments(self) -> None:
        """Start a new stats period without changing active-request accounting."""
        with self._lock:
            counters = {server: counter for _, counter, server in self.decoder_heap}
            for server in self.decoders:
                server.period_assigned_requests = 0
            self.decoder_heap = []
            for server in self.decoders:
                counter = counters.get(server)
                if counter is None:
                    counter = self._next_counter()
                self.decoder_heap.append((self._calc_decoder_priority(server), counter, server))
            heapq.heapify(self.decoder_heap)

    def calculate_prefill_scores(self, token_count: int) -> float:
        return token_count

    def calculate_decode_scores(self, input_tokens: int, max_output_tokens: int) -> float:
        """Compatibility shim; decoder scheduling no longer uses token scores."""
        return 1.0

    def inc_request_num(self):
        with self._lock:
            self.request_num += 1

    def dec_request_num(self):
        with self._lock:
            self.request_num = max(0, self.request_num - 1)
            if self.request_num == 0:
                self._finalize_pending_removals_locked()

    def get_request_num(self):
        with self._lock:
            return self.request_num

    def select_tokenize_node(self, role: str) -> ServerState:
        """Select independently of inference load and without reserving KV."""
        with self._lock:
            nodes, tainted, pending = (
                (self.prefillers, self.tainted_prefillers, self.pending_remove_prefillers)
                if role == "prefill" else
                (self.decoders, self.tainted_decoders, self.pending_remove_decoders)
            )
            available = [node for node in nodes if node not in tainted and node not in pending]
            if not available:
                raise ResponsesRequestError(f"Image tokenize {role}: no healthy node.", "multimodal_tokenize_unavailable", 502)
            index = self._tokenize_round_robin[role]
            self._tokenize_round_robin[role] = index + 1
            return available[index % len(available)]

    async def add_instances(self, instance_type: str, instances: list[ServerState]) -> tuple[list[str], list[str]]:
        added_nodes, waiting_nodes = [], []
        for server in instances:
            is_valid = await self.node_listener.check_instance_status(server.client)
            if is_valid and instance_type == InstanceType.PREFILL:
                self.add_prefillers([server])
                added_nodes.append(str(server))
            elif is_valid and instance_type == InstanceType.DECODE:
                self.add_decoders([server])
                added_nodes.append(str(server))
            else:
                node = str(server)
                self.node_listener.add_waiting_node(node, instance_type, server)
                waiting_nodes.append(node)
        return added_nodes, waiting_nodes

    def add_prefillers(self, instances: list[ServerState]) -> None:
        with self._lock:
            for server in instances:
                self.pending_remove_prefillers.discard(server)
                if server in self.tainted_prefillers:
                    self.tainted_prefillers.remove(server)
                    # Fix #4: Calculate actual priority from current load instead of hardcoding 0
                    self.prefiller_heap = [(p, c, s) for p, c, s in self.prefiller_heap if s is not server]
                    heapq.heappush(self.prefiller_heap,
                                   (self._calc_prefiller_priority(server), self._next_counter(), server))
                elif server not in self.prefillers:
                    self.prefillers.append(server)
                    heapq.heappush(self.prefiller_heap,
                                   (self._calc_prefiller_priority(server), self._next_counter(), server))
            self.print_status(f"Add prefiller instances: {instances}.")

    def add_decoders(self, instances: list[ServerState], *, reason: str = "admin") -> None:
        with self._lock:
            for server in instances:
                self.pending_remove_decoders.discard(server)
                if server in self.tainted_decoders:
                    self.tainted_decoders.remove(server)
                    # Fix #4: Calculate actual priority from current load instead of hardcoding 0
                    self.decoder_heap = [(p, c, s) for p, c, s in self.decoder_heap if s is not server]
                    heapq.heappush(self.decoder_heap,
                                   (self._calc_decoder_priority(server), self._next_counter(), server))
                elif server not in self.decoders:
                    self.decoders.append(server)
                    heapq.heappush(self.decoder_heap,
                                   (self._calc_decoder_priority(server), self._next_counter(), server))
            event = "decoder_recovered_after_taint" if reason == "recovered_after_taint" else "decoder_admin_added"
            self.print_status(f"{event}: {instances}.")

    def remove_prefillers(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        with self._lock:
            if self.request_num > 0:
                logger.warning(f"Start to taint prefill instances {instances}.")
                self.pending_remove_prefillers.update(instances)
                self._taint_prefillers_locked(instances)
                return True

            self._remove_prefillers_locked(set(instances))
            self.print_status(f"Remove prefiller instances: {instances}.")
            return False

    def remove_decoders(self, instances: list[ServerState]) -> bool:
        if not instances:
            return False

        with self._lock:
            if self.request_num > 0:
                logger.warning(f"Start to taint decode instances {instances}.")
                self.pending_remove_decoders.update(instances)
                self._taint_decoders_locked(instances)
                return True

            self._remove_decoders_locked(set(instances))
            self.print_status(f"Remove decoder instances: {instances}.")
            return False

    def _remove_prefillers_locked(self, instances_to_remove: set[ServerState]) -> None:
        """Remove prefillers from active, tainted, pending and heap state. Caller holds lock."""
        self.prefillers = [server for server in self.prefillers if server not in instances_to_remove]
        self.tainted_prefillers = [server for server in self.tainted_prefillers if server not in instances_to_remove]
        self.pending_remove_prefillers.difference_update(instances_to_remove)
        self.prefiller_heap = [(p, c, s) for p, c, s in self.prefiller_heap if s not in instances_to_remove]
        heapq.heapify(self.prefiller_heap)

    def _remove_decoders_locked(self, instances_to_remove: set[ServerState]) -> None:
        """Remove decoders from active, tainted, pending and heap state. Caller holds lock."""
        self.decoders = [server for server in self.decoders if server not in instances_to_remove]
        self.tainted_decoders = [server for server in self.tainted_decoders if server not in instances_to_remove]
        self.pending_remove_decoders.difference_update(instances_to_remove)
        self.decoder_heap = [(p, c, s) for p, c, s in self.decoder_heap if s not in instances_to_remove]
        heapq.heapify(self.decoder_heap)

    def _finalize_pending_removals_locked(self) -> None:
        """Physically remove tainted nodes once all in-flight requests finish. Caller holds lock."""
        if self.pending_remove_prefillers:
            instances = set(self.pending_remove_prefillers)
            self._remove_prefillers_locked(instances)
            self.print_status(f"Remove pending prefiller instances: {list(instances)}.")
        if self.pending_remove_decoders:
            instances = set(self.pending_remove_decoders)
            self._remove_decoders_locked(instances)
            self.print_status(f"Remove pending decoder instances: {list(instances)}.")

    def _taint_prefillers_locked(self, instances: list[ServerState]) -> None:
        """Internal: taint prefillers. Caller must hold self._lock."""
        instances_to_taint = set(instances)
        for server in self.prefillers:
            if server in instances_to_taint and server not in self.tainted_prefillers:
                self.tainted_prefillers.append(server)

        self.prefiller_heap = [
            (TAINT_PRIORITY, c, srv) if srv in instances_to_taint else (priority, c, srv)
            for priority, c, srv in self.prefiller_heap
        ]
        heapq.heapify(self.prefiller_heap)

    def _taint_decoders_locked(self, instances: list[ServerState]) -> None:
        """Internal: taint decoders. Caller must hold self._lock."""
        instances_to_taint = set(instances)
        for server in self.decoders:
            if server in instances_to_taint and server not in self.tainted_decoders:
                self.tainted_decoders.append(server)

        self.decoder_heap = [
            (self._calc_decoder_priority(srv), c, srv)
            if srv in instances_to_taint else (priority, c, srv)
            for priority, c, srv in self.decoder_heap
        ]
        heapq.heapify(self.decoder_heap)

    def _taint_prefillers(self, instances: list[ServerState]) -> None:
        with self._lock:
            self._taint_prefillers_locked(instances)

    def _taint_decoders(self, instances: list[ServerState]) -> None:
        with self._lock:
            self._taint_decoders_locked(instances)

    def taint_decoder(self, server: ServerState, reason: str = "unknown") -> None:
        """Mark a decoder as unhealthy so it is not selected for new requests."""
        with self._lock:
            was_tainted = server in self.tainted_decoders
            self._taint_decoders_locked([server])
            if not was_tainted:
                self.decoder_taint_total[(str(server), reason)] += 1
        logger.warning(
            "decoder_tainted decoder=%s reason=%s failure_category=%s",
            server, reason, reason,
        )

    def taint_prefillers(self, instances: list[ServerState]) -> None:
        """Mark prefillers as unhealthy so they are not selected for new requests."""
        self._taint_prefillers(instances)
        for s in instances:
            logger.warning(f"Prefiller {s.url} marked as tainted due to failure.")

    def snapshot_for_health_check(self) -> tuple[list[ServerState], list[ServerState],
                                                   list[ServerState], list[ServerState],
                                                   set[ServerState], set[ServerState]]:
        """Return snapshots of all server lists for safe iteration in health check thread.
        Returns active, tainted and pending-removal snapshots.
        """
        with self._lock:
            return (list(self.prefillers), list(self.decoders),
                    list(self.tainted_prefillers), list(self.tainted_decoders),
                    set(self.pending_remove_prefillers), set(self.pending_remove_decoders))

    def snapshot_for_stats(self) -> tuple[list[ServerState], list[ServerState], dict[str, int], dict[str, int], int]:
        """Return node snapshots plus current active request counts at one moment."""
        with self._lock:
            prefillers = list(self.prefillers)
            decoders = list(self.decoders)
            prefiller_active = {str(server): server.active_requests for server in prefillers}
            decoder_active = {str(server): server.active_requests for server in decoders}
            return prefillers, decoders, prefiller_active, decoder_active, self.request_num

    def mark_backend_success(self, server: ServerState, kind: str) -> None:
        """Record a successful real backend operation for adaptive health checks."""
        with self._lock:
            server.last_success_at = time.time()
            server.last_success_kind = kind
            if getattr(globals().get("global_args"), "decoder_taint_reset_on_success", DECODER_TAINT_RESET_ON_SUCCESS):
                getattr(self, "decoder_failure_state", {}).pop(server, None)
        self.node_listener.record_business_success(str(server), kind)

    def record_decoder_failure_observation(self, server: ServerState, category: str) -> None:
        with self._lock:
            self.decoder_failure_total[(str(server), category)] += 1

    def record_decoder_taint_suppressed(self, server: ServerState, category: str) -> None:
        with self._lock:
            self.decoder_taint_suppressed_total[(str(server), category)] += 1

    def record_decoder_recovery_probe(self, server: ServerState, ok: bool) -> None:
        with self._lock:
            self.decoder_recovery_probe_total[(str(server), "success" if ok else "failure")] += 1

    def record_decoder_recovery(self, server: ServerState) -> None:
        with self._lock:
            self.decoder_recovery_total[str(server)] += 1

    def snapshot_decoder_taint_metrics(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            failure_windows = []
            for server, state in self.decoder_failure_state.items():
                failure_windows.append({
                    "decoder": str(server),
                    "window_age_seconds": round(max(0.0, now - state["window_started_at"]), 3),
                    "consecutive_failures": state["consecutive_failures"],
                    "transport_failures": state["transport_failures"],
                    "http_5xx_failures": state["http_5xx_failures"],
                    "last_failure_at": state.get("last_failure_at"),
                    "last_success_at": server.last_success_at or None,
                })
            return {
                "decoder_failure_total": [
                    {"decoder": decoder, "category": category, "value": value}
                    for (decoder, category), value in sorted(self.decoder_failure_total.items())
                ],
                "decoder_taint_suppressed_total": [
                    {"decoder": decoder, "category": category, "value": value}
                    for (decoder, category), value in sorted(self.decoder_taint_suppressed_total.items())
                ],
                "decoder_taint_total": [
                    {"decoder": decoder, "reason": reason, "value": value}
                    for (decoder, reason), value in sorted(self.decoder_taint_total.items())
                ],
                "decoder_recovery_total": [
                    {"decoder": decoder, "value": value}
                    for decoder, value in sorted(self.decoder_recovery_total.items())
                ],
                "decoder_recovery_probe_total": [
                    {"decoder": decoder, "result": result, "value": value}
                    for (decoder, result), value in sorted(self.decoder_recovery_probe_total.items())
                ],
                "decoder_failure_window_current": sorted(
                    failure_windows, key=lambda item: item["decoder"]
                ),
            }

    def record_decoder_failure(self, server: ServerState, category: str) -> tuple[int, bool]:
        now = time.time()
        window = getattr(global_args, "decoder_taint_failure_window_seconds", DECODER_TAINT_FAILURE_WINDOW_SECONDS)
        threshold = getattr(global_args, "decoder_taint_failure_threshold", DECODER_TAINT_FAILURE_THRESHOLD)
        http_threshold = getattr(global_args, "decoder_taint_http_5xx_threshold", DECODER_TAINT_HTTP_5XX_THRESHOLD)
        with self._lock:
            state = self.decoder_failure_state.get(server)
            if state is None or now - state["window_started_at"] > window:
                state = {"window_started_at": now, "consecutive_failures": 0, "transport_failures": 0, "http_5xx_failures": 0}
                self.decoder_failure_state[server] = state
            state["consecutive_failures"] += 1
            if category == "transport":
                state["transport_failures"] += 1
            if category == "http_5xx":
                state["http_5xx_failures"] += 1
            state["last_failure_at"] = now
            count = state["consecutive_failures"]
            should_taint = (category == "http_5xx" and state["http_5xx_failures"] >= http_threshold) or count >= threshold
            return count, should_taint

    def snapshot_backend_success(self) -> dict[str, dict[str, Any]]:
        """Return business-health timestamps without exposing mutable ServerState."""
        now = time.time()
        with self._lock:
            return {
                str(server): {
                    "last_success_age_seconds": (
                        round(max(0.0, now - server.last_success_at), 3)
                        if server.last_success_at else None
                    ),
                    "last_success_kind": server.last_success_kind or None,
                }
                for server in [*self.prefillers, *self.decoders]
            }

    def print_status(self, msg: str) -> None:
        status = {
            "prefill_instances": [str(server) for server in self.prefillers],
            "decode_instances": [str(server) for server in self.decoders],
        }
        print(f"{msg} Status: {status}")


proxy_state = None
detached_task_manager = DetachedTaskManager()
proxy_shutdown_started = False


class NodeListener:
    def __init__(self, proxy):
        self.proxy_state = proxy
        self.waiting_nodes: dict[str, tuple[str, Any, int]] = {}
        self._waiting_lock = threading.Lock()
        # Health check stats: key = "host:port", value = probe results and skip counts.
        self.health_stats: dict[str, dict] = {}
        self.recovery_successes: dict[str, int] = {}
        self.recovery_last_probe: dict[str, float] = {}
        # Lock for health_stats (read by main event loop for stats, written by this thread)
        self._stats_lock = threading.Lock()
        # Separate sync client for health checks — AsyncClient cannot be shared
        # across the main event loop and this sync thread's event loops
        self._sync_client = httpx.Client(timeout=5.0)
        self.listening_thread = threading.Thread(target=self._node_listener, daemon=True)
        self.listening_thread.start()

    def _node_listener(self) -> None:
        while True:
            # Phase 1: Check nodes queued by /instances/add.
            for node, (instance_type, server, check_times) in self.get_waiting_nodes_snapshot():
                is_valid = self.check_instance_status_sync(server)
                self._record_health_probe(node, "waiting", is_valid)
                print(f"Checking instance {node}...")
                check_times += 1
                if is_valid:
                    if instance_type == InstanceType.PREFILL:
                        self.proxy_state.add_prefillers([server])
                    else:
                        self.proxy_state.add_decoders([server], reason="recovered_after_taint")
                    self.remove_waiting_node(node)
                elif check_times == getattr(globals().get("global_args"), "max_waiting_retries", 3):
                    print(f"Instance {node} was not added to the proxy.")
                    self.remove_waiting_node(node)
                else:
                    self.update_waiting_node(node, instance_type, server, check_times)

            # Phase 2: Snapshot shared state under lock, then check health outside lock.
            # Recovery probes for tainted nodes are always enabled. The periodic flag
            # controls only proactive probes of nodes that are currently active.
            prefillers, decoders, tainted_pfs, tainted_dcs, pending_pfs, pending_dcs = \
                self.proxy_state.snapshot_for_health_check()
            periodic_enabled = self._periodic_healthcheck_enabled()
            if not self._should_run_backend_checks(
                periodic_enabled, tainted_pfs, tainted_dcs, pending_pfs, pending_dcs
            ):
                time.sleep(getattr(globals().get("global_args"), "waiting_retry_interval", 10.0))
                continue

            # Check if tainted backends have recovered — restore them to active pool
            for server in tainted_pfs:
                if server in pending_pfs:
                    continue
                key = f"{server.host}:{server.port}"
                is_valid = self.check_instance_status_sync(server)
                self._record_health_probe(key, "tainted", is_valid)
                if is_valid:
                    self.proxy_state.add_prefillers([server])
                    print(f"Prefiller {server.url} recovered and restored to active pool.")

            for server in tainted_dcs:
                if server in pending_dcs:
                    continue
                key = f"{server.host}:{server.port}"
                is_valid = self.check_instance_status_sync(server)
                self._record_health_probe(key, "tainted", is_valid)
                self.proxy_state.record_decoder_recovery_probe(server, is_valid)
                if is_valid:
                    now = time.time()
                    if now - self.recovery_last_probe.get(key, 0.0) < 10.0:
                        continue
                    self.recovery_last_probe[key] = now
                    successes = self.recovery_successes.get(key, 0) + 1
                    self.recovery_successes[key] = successes
                    if successes >= 2:
                        self.proxy_state.add_decoders([server], reason="recovered_after_taint")
                        self.proxy_state.record_decoder_recovery(server)
                        self.recovery_successes.pop(key, None)
                        self.recovery_last_probe.pop(key, None)
                        print(f"decoder_recovered_after_taint decoder={server} recovery_probe_count={successes}")
                else:
                    self.recovery_successes[key] = 0

            # Phase 3: Optionally health-check active pool backends and auto-taint
            # failures. This is deliberately separate from Phase 2 recovery.
            if periodic_enabled:
                # Use the snapshot taken above (safe even if list changed since snapshot)
                for server in prefillers:
                    if server in tainted_pfs or server in pending_pfs:
                        continue
                    key = f"{server.host}:{server.port}"
                    if self._has_recent_business_success(server):
                        self._record_health_skip(key)
                        continue
                    is_valid = self.check_instance_status_sync(server)
                    self._record_health_probe(key, "idle", is_valid)
                    # Re-check tainted status under ProxyState lock via taint method
                    if not is_valid:
                        self.proxy_state.taint_prefillers([server])

                for server in decoders:
                    if server in tainted_dcs or server in pending_dcs:
                        continue
                    key = f"{server.host}:{server.port}"
                    if self._has_recent_business_success(server):
                        self._record_health_skip(key)
                        continue
                    is_valid = self.check_instance_status_sync(server)
                    self._record_health_probe(key, "idle", is_valid)
                    if not is_valid and getattr(global_args, "decoder_taint_enabled", DECODER_TAINT_ENABLED):
                        self.proxy_state.record_decoder_failure_observation(server, "healthcheck")
                        self.proxy_state.taint_decoder(server, reason="healthcheck")

            time.sleep(getattr(globals().get("global_args"), "waiting_retry_interval", 10.0))

    def add_waiting_node(self, node: str, instance_type: str, server: ServerState) -> None:
        with self._waiting_lock:
            self.waiting_nodes[node] = (instance_type, server, 0)

    def get_waiting_nodes_snapshot(self) -> list[tuple[str, tuple[str, Any, int]]]:
        with self._waiting_lock:
            return list(self.waiting_nodes.items())

    def update_waiting_node(self, node: str, instance_type: str, server: ServerState, check_times: int) -> None:
        with self._waiting_lock:
            if node in self.waiting_nodes:
                self.waiting_nodes[node] = (instance_type, server, check_times)

    def remove_waiting_node(self, node: str) -> None:
        with self._waiting_lock:
            self.waiting_nodes.pop(node, None)

    def _has_recent_business_success(self, server: ServerState) -> bool:
        args = globals().get("global_args")
        ttl = getattr(args, "healthcheck_success_ttl_seconds", 60.0)
        if ttl <= 0:
            return False
        with self.proxy_state._lock:
            return bool(server.last_success_at and time.time() - server.last_success_at <= ttl)

    @staticmethod
    def _periodic_healthcheck_enabled() -> bool:
        return bool(getattr(globals().get("global_args"), "enable_periodic_backend_healthcheck", False))

    @staticmethod
    def _should_run_backend_checks(
        periodic_enabled: bool,
        tainted_prefillers: list[ServerState],
        tainted_decoders: list[ServerState],
        pending_prefillers: set[ServerState],
        pending_decoders: set[ServerState],
    ) -> bool:
        """Run checks when active probing is enabled or recovery targets exist."""
        if periodic_enabled:
            return True
        return any(server not in pending_prefillers for server in tainted_prefillers) or any(
            server not in pending_decoders for server in tainted_decoders
        )

    def _record_health_probe(self, key: str, probe_kind: str, ok: bool) -> None:
        with self._stats_lock:
            self._ensure_health_stats_locked(key)
            self.health_stats[key][f"probed_{probe_kind}"] += 1
            if ok:
                self.health_stats[key]["ok"] += 1
            else:
                self.health_stats[key]["fail"] += 1

    def _record_health_skip(self, key: str) -> None:
        with self._stats_lock:
            self._ensure_health_stats_locked(key)
            self.health_stats[key]["skipped_recent_success"] += 1

    def record_business_success(self, key: str, kind: str) -> None:
        with self._stats_lock:
            self._ensure_health_stats_locked(key)
            self.health_stats[key]["business_success"] += 1
            by_kind = self.health_stats[key]["business_success_by_kind"]
            by_kind[kind] = by_kind.get(kind, 0) + 1

    def _ensure_health_stats_locked(self, key: str) -> None:
        if key not in self.health_stats:
            self.health_stats[key] = {
                "ok": 0,
                "fail": 0,
                "probed_idle": 0,
                "probed_tainted": 0,
                "probed_waiting": 0,
                "skipped_recent_success": 0,
                "business_success": 0,
                "business_success_by_kind": {},
            }

    def get_health_stats(self) -> dict:
        """Return a snapshot of health_stats for safe reading from other threads."""
        with self._stats_lock:
            return {k: dict(v) for k, v in self.health_stats.items()}

    def get_healthcheck_summary(self) -> dict[str, Any]:
        with self._stats_lock:
            totals = {
                "probed_idle": 0,
                "probed_tainted": 0,
                "probed_waiting": 0,
                "skipped_recent_success": 0,
                "ok": 0,
                "fail": 0,
                "business_success": 0,
            }
            for stats in self.health_stats.values():
                for key in totals:
                    totals[key] += stats.get(key, 0)
        args = globals().get("global_args")
        totals["periodic_enabled"] = bool(
            getattr(args, "enable_periodic_backend_healthcheck", False)
        )
        totals["success_ttl_seconds"] = getattr(args, "healthcheck_success_ttl_seconds", 60.0)
        totals["business_success_by_kind"] = {}
        with self._stats_lock:
            for stats in self.health_stats.values():
                for kind, count in stats.get("business_success_by_kind", {}).items():
                    totals["business_success_by_kind"][kind] = (
                        totals["business_success_by_kind"].get(kind, 0) + count
                    )
        return totals

    @staticmethod
    def fmt_health(ok: int, fail: int, tainted: bool = False) -> str:
        total = ok + fail
        if total == 0:
            return "(unkn)"
        pct = ok / total * 100
        prefix = "(taint) " if tainted else ""
        return f"{prefix}{pct:.2f}%({ok}/{fail})"

    def check_instance_status_sync(self, server: ServerState) -> bool:
        """Synchronous health check — safe to call from the _node_listener thread."""
        url = server.url.replace("/v1", "/health")
        try:
            response = self._sync_client.get(url)
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False

    @staticmethod
    async def check_instance_status(client: httpx.AsyncClient) -> bool:
        endpoint = "/health"
        try:
            response = await client.get(endpoint)
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPStatusError):
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--multimodal-tokenize-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--multimodal-tokenize-max-concurrency", type=int, default=8)
    parser.add_argument("--multimodal-max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument("--decoder-taint-enabled", action=argparse.BooleanOptionalAction, default=DECODER_TAINT_ENABLED)
    parser.add_argument("--decoder-taint-failure-threshold", type=int, default=DECODER_TAINT_FAILURE_THRESHOLD)
    parser.add_argument("--decoder-taint-failure-window-seconds", type=float, default=DECODER_TAINT_FAILURE_WINDOW_SECONDS)
    parser.add_argument("--decoder-taint-http-5xx-threshold", type=int, default=DECODER_TAINT_HTTP_5XX_THRESHOLD)
    parser.add_argument("--decoder-taint-reset-on-success", action=argparse.BooleanOptionalAction, default=DECODER_TAINT_RESET_ON_SUCCESS)
    parser.add_argument(
        "--enable-chat-stream-guard", action=argparse.BooleanOptionalAction,
        default=CHAT_STREAM_GUARD_ENABLED,
        help="Guard Chat streaming against role-only empty assistant events and empty completions.",
    )
    parser.add_argument(
        "--chat-stream-guard-trace", action=argparse.BooleanOptionalAction,
        default=CHAT_STREAM_GUARD_TRACE_ENABLED,
        help="Capture complete request/response evidence when Chat stream guard detects an anomaly.",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries"
    )
    parser.add_argument(
        "--max-waiting-retries", type=int, default=3, help="Maximum number of retries for waiting nodes to be started"
    )
    parser.add_argument(
        "--waiting-retry-interval",
        type=float,
        default=10,
        help="Check interval (seconds) for waiting nodes to be started",
    )
    parser.add_argument(
        "--enable-periodic-backend-healthcheck",
        action="store_true",
        default=False,
        help="Enable periodic /health probes for active and tainted backend nodes. Disabled by default.",
    )
    parser.add_argument(
        "--healthcheck-success-ttl-seconds",
        type=float,
        default=60.0,
        help="Skip active-node /health checks after a successful real request for this many seconds; 0 disables skipping.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model directory for tokenizer. If provided, enables token limit "
        "validation against prefiller/decoder max_model_len.",
    )
    parser.add_argument(
        "--backend-model-name",
        type=str,
        default=None,
        help="Force every backend request to use this model ID. Requests keep their original "
        "model value when this option is unset.",
    )
    parser.add_argument(
        "--prefiller-max-model-len",
        type=int,
        default=None,
        help="Maximum model length (max_model_len) for prefiller instances. "
        "If not provided, falls back to max_position_embeddings from model config.json.",
    )
    parser.add_argument(
        "--decoder-max-model-len",
        type=int,
        default=None,
        help="Maximum model length (max_model_len) for decoder instances. "
        "If not provided, falls back to max_position_embeddings from model config.json.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=0,
        help="Reject requests whose JSON string payload roughly exceeds this many characters before tokenization. "
        "Set to 0 to disable.",
    )
    parser.add_argument(
        "--stats-enabled",
        action="store_true",
        default=True,
        help="Enable periodic statistics printing. Default: True.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_false",
        dest="stats_enabled",
        help="Disable periodic statistics printing.",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=120,
        help="Statistics printing interval in seconds. Default: 120.",
    )
    parser.add_argument(
        "--certain-kv-leak-log-path",
        type=str,
        default=DEFAULT_CERTAIN_KV_LEAK_LOG_PATH,
        help="JSONL file used to append certain KV leak records.",
    )
    parser.add_argument(
        "--suspected-kv-leak-log-path",
        type=str,
        default=DEFAULT_SUSPECTED_KV_LEAK_LOG_PATH,
        help="JSONL file used to append suspected KV leak records.",
    )
    parser.add_argument(
        "--kv-leak-recent-limit",
        type=int,
        default=20,
        help="Maximum number of recent KV leak summaries kept in memory for stats printing.",
    )
    parser.add_argument(
        "--max-recomputed-attempts",
        type=int,
        default=8,
        help="Maximum recomputed attempts allowed before background drain fails.",
    )
    parser.add_argument(
        "--max-total-decode-attempts",
        type=int,
        default=16,
        help="Maximum total decode attempts allowed before background drain fails.",
    )
    parser.add_argument(
        "--max-decoder-start-protocol-recompute-attempts",
        type=int,
        default=1,
        help="Maximum full Prefill and Decode recomputes after a Decoder start-protocol leak.",
    )
    parser.add_argument(
        "--enable-decoder-start-protocol-retry-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append an internal retry user turn after a Decoder start-protocol leak.",
    )
    parser.add_argument(
        "--responses-sse-heartbeat-seconds",
        type=float,
        default=10.0,
        help="Emit response.in_progress while a Responses stream waits for its first decoder event; 0 disables.",
    )
    parser.add_argument(
        "--decoder-first-response-observe-seconds",
        type=float,
        default=30.0,
        help="Warn when a Decoder stream has not produced its first SSE chunk after this many seconds; 0 disables warnings.",
    )
    parser.add_argument(
        "--decoder-first-response-log-interval-seconds",
        type=float,
        default=60.0,
        help="Repeat Decoder first-response waiting warnings at this interval.",
    )
    parser.add_argument(
        "--pd-first-output-timeout-seconds",
        type=float,
        default=480.0,
        help=(
            "Maximum seconds from sending a Prefill request until the first Decoder "
            "semantic output. 0 disables the recovery deadline."
        ),
    )
    parser.add_argument(
        "--decoder-stream-idle-timeout-seconds",
        type=float,
        default=480.0,
        help=(
            "Maximum seconds without Decoder semantic output after the first valid "
            "output. 0 disables the idle deadline."
        ),
    )
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")
    if args.backend_model_name is not None:
        args.backend_model_name = args.backend_model_name.strip()
        if not args.backend_model_name:
            raise ValueError("--backend-model-name must not be empty.")
    if args.multimodal_tokenize_timeout_seconds <= 0 or args.multimodal_tokenize_max_concurrency <= 0:
        raise ValueError("Multimodal tokenize timeout and concurrency must be positive.")
    if args.multimodal_max_image_bytes <= 0:
        raise ValueError("--multimodal-max-image-bytes must be positive.")
    if args.responses_sse_heartbeat_seconds < 0:
        raise ValueError("--responses-sse-heartbeat-seconds must be >= 0.")
    if args.max_decoder_start_protocol_recompute_attempts < 0:
        raise ValueError("--max-decoder-start-protocol-recompute-attempts must be >= 0.")
    if args.decoder_first_response_observe_seconds < 0:
        raise ValueError("--decoder-first-response-observe-seconds must be >= 0.")
    if args.decoder_first_response_log_interval_seconds <= 0:
        raise ValueError("--decoder-first-response-log-interval-seconds must be > 0.")
    if args.pd_first_output_timeout_seconds < 0:
        raise ValueError("--pd-first-output-timeout-seconds must be >= 0.")
    if args.decoder_stream_idle_timeout_seconds < 0:
        raise ValueError("--decoder-stream-idle-timeout-seconds must be >= 0.")
    if args.decoder_taint_failure_threshold <= 0 or args.decoder_taint_http_5xx_threshold <= 0:
        raise ValueError("Decoder taint thresholds must be positive.")
    if args.decoder_taint_failure_window_seconds <= 0:
        raise ValueError("--decoder-taint-failure-window-seconds must be > 0.")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    return args


tokenizer = None
prefiller_max_model_len = None
decoder_max_model_len = None
multimodal_tokenize_client = None
multimodal_tokenize_semaphore = None


def _has_multimodal_content(value: Any) -> bool:
    if isinstance(value, dict):
        value_type = value.get("type")
        if isinstance(value_type, str) and value_type in {"input_image", "image_url"}:
            return True
        return any(_has_multimodal_content(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_multimodal_content(item) for item in value)
    return False


def _redact_multimodal(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_multimodal(item) for item in value]
    if isinstance(value, dict):
        result = {key: _redact_multimodal(item) for key, item in value.items()}
        if result.get("type") == "input_image":
            if value.get("image_url") == "<redacted-image-url>":
                return result
            if isinstance(result.get("image_url"), str):
                result["image_url"] = "<redacted-image-url>"
            result["image_sha256"] = hashlib.sha256(
                str(value.get("image_url", "")).encode()
            ).hexdigest()
        if result.get("type") == "image_url" and isinstance(result.get("image_url"), dict):
            url = result["image_url"].get("url")
            if isinstance(url, str) and url != "<redacted-image-url>":
                result["image_url"] = {
                    **result["image_url"],
                    "url": "<redacted-image-url>",
                    "url_sha256": hashlib.sha256(url.encode()).hexdigest(),
                }
        return result
    return value


def _validate_image_part(part: dict[str, Any]) -> None:
    image_url = part.get("image_url")
    if not isinstance(image_url, str):
        raise ResponsesRequestError(
            "Image content requires an image_url string; file IDs are unsupported.",
            "unsupported_multimodal_content",
        )
    try:
        parsed = urlparse(image_url)
    except ValueError:
        raise ResponsesRequestError("Invalid image URL.", "multimodal_invalid_image") from None
    if parsed.scheme not in {"data", "http", "https"}:
        raise ResponsesRequestError("Image URL must use data, http, or https.", "multimodal_invalid_image")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ResponsesRequestError("Image URL requires a host.", "multimodal_invalid_image")
    if parsed.scheme == "data":
        try:
            header = image_url.split(",", 1)[0]
            if not header.startswith("data:image/") or not header.endswith(";base64"):
                raise ValueError("Expected a base64 image data URL.")
            encoded = image_url.split(",", 1)[1]
            image_bytes = base64.b64decode(encoded, validate=True)
        except (IndexError, ValueError, TypeError) as exc:
            raise ResponsesRequestError("Invalid data URL image.", "multimodal_invalid_image") from exc
        if len(image_bytes) > getattr(globals().get("global_args"), "multimodal_max_image_bytes", 20 * 1024 * 1024):
            raise ResponsesRequestError("Image exceeds the configured size limit.", "multimodal_image_too_large")


def _responses_multimodal_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesRequestError("Message content must be a string or an array.", "invalid_input_content")
    output = []
    for part in content:
        if isinstance(part, str):
            output.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise ResponsesRequestError("Invalid input content part.", "invalid_input_content")
        part_type = part.get("type")
        if not isinstance(part_type, str):
            raise ResponsesRequestError("Input content type must be a string.", "invalid_input_content")
        if part_type in {"input_text", "text", "output_text"}:
            if not isinstance(part.get("text"), str):
                raise ResponsesRequestError("Text content requires string text.", "invalid_input_content")
            output.append({"type": "text", "text": part["text"]})
        elif part_type == "input_image":
            _validate_image_part(part)
            if part.get("detail") is not None and part["detail"] not in ("auto", "low", "high", "original"):
                raise ResponsesRequestError("Invalid image detail.", "multimodal_invalid_image")
            image = {"url": part["image_url"]}
            if part.get("detail") is not None:
                image["detail"] = part["detail"]
            output.append({"type": "image_url", "image_url": image})
        elif part_type == "image_url":
            image = part.get("image_url")
            if not isinstance(image, dict):
                raise ResponsesRequestError("image_url must be an object.", "invalid_input_content")
            _validate_image_part({"image_url": image.get("url")})
            if image.get("detail") is not None and image["detail"] not in ("auto", "low", "high", "original"):
                raise ResponsesRequestError("Invalid image detail.", "multimodal_invalid_image")
            output.append(copy.deepcopy(part))
        else:
            raise ResponsesRequestError(
                "Only text and image content parts are supported.",
                "unsupported_multimodal_content",
            )
    return output


@dataclass
class RequestLeakState:
    request_id: str = ""
    client_addr: str = "unknown"
    api: str = ""
    stream: bool = False
    request_kind: str = ""
    prefill_committed: bool = False
    kv_transfer_params_received: bool = False
    decoder_finished_normally: bool = False
    client_disconnected: bool = False
    disconnect_phase: str = ""
    pipeline_phase: str = "prefill_pending"
    drain_after_prefill: bool = False
    prefill_disconnect_draining: bool = False
    prefill_disconnect_drain_completed: bool = False
    prefill_disconnect_drain_failed: bool = False
    first_decoder_output_received: bool = False
    first_decoder_output_at: float = 0.0
    prefill_completed_at: float = 0.0
    decoder_started_at: float = 0.0
    decoder_stream_closed_at: float = 0.0
    worker_finished_at: float = 0.0
    drain_stop_reason: str = ""
    drain_stop_at: float = 0.0
    local_prefiller_kv_released_at: float = 0.0
    local_decoder_released_at: float = 0.0
    trace: Any = None
    recomputed_seen: bool = False
    decoder_start_protocol_seen: bool = False
    leak_level: str = LeakLevel.NONE
    leak_reason: str = ""
    leak_recorded: bool = False
    input_tokens: int = 0
    decoder_input_tokens: int = 0
    requested_max_tokens: int = 0
    completion_tokens: int = 0
    prefiller: str = ""
    decoder: str = ""
    final_request_state: str = ""
    generated_text: str = ""
    origin_max_tokens: int = 0
    completion_tokens_before_retry: int = 0
    recomputed_attempts: int = 0
    decoder_start_protocol_recompute_attempts: int = 0
    decode_attempts: int = 0
    final_decoder_finish_observed: bool = False
    prefill_disconnect_drain_started_at: float = 0.0
    prefill_disconnect_drain_ended_at: float = 0.0
    disconnect_detected_at: float = 0.0
    last_decoder_chunk_at: float = 0.0
    decoder_done_observed: bool = False


@dataclass
class StreamOutputAttempt:
    """Output observed from one decoder attempt, before any recomputed retry."""

    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_argument_parts: dict[str, list[str]] = field(default_factory=dict)
    backend_completion_tokens: int | None = None
    first_output_at: float | None = None
    last_output_at: float | None = None
    token_source: str = "unavailable"
    total_tokens: int | None = None
    text_tokens: int | None = None
    reasoning_tokens: int | None = None
    tool_argument_tokens: int | None = None


@dataclass
class StreamOutputAccounting:
    """Request-level decoder output accounting, kept separate across retries."""

    attempts: list[StreamOutputAttempt] = field(default_factory=list)

    def start_attempt(self) -> StreamOutputAttempt:
        attempt = StreamOutputAttempt()
        self.attempts.append(attempt)
        return attempt

    @staticmethod
    def _encode_parts(parts: list[str]) -> int | None:
        if not parts:
            return 0
        if tokenizer is None:
            return None
        return len(tokenizer.encode("".join(parts), add_special_tokens=False))

    def observe(self, attempt: StreamOutputAttempt, info: "StreamChoiceInfo", observed_at: float) -> None:
        if info.text:
            attempt.text_parts.append(info.text)
        if info.reasoning:
            attempt.reasoning_parts.append(info.reasoning)
        for key, arguments in info.tool_arguments.items():
            if arguments:
                attempt.tool_argument_parts.setdefault(key, []).append(arguments)
        if info.backend_completion_tokens is not None:
            attempt.backend_completion_tokens = info.backend_completion_tokens
        if info.has_output:
            if attempt.first_output_at is None:
                attempt.first_output_at = observed_at
            attempt.last_output_at = observed_at

    def finalize_attempt(self, attempt: StreamOutputAttempt) -> StreamOutputAttempt:
        if attempt.total_tokens is not None:
            return attempt
        try:
            attempt.text_tokens = self._encode_parts(attempt.text_parts)
            attempt.reasoning_tokens = self._encode_parts(attempt.reasoning_parts)
            tool_tokens = 0
            for parts in attempt.tool_argument_parts.values():
                count = self._encode_parts(parts)
                if count is None:
                    tool_tokens = None
                    break
                tool_tokens += count
            attempt.tool_argument_tokens = tool_tokens
        except Exception:
            logger.debug("Unable to tokenize decoder stream output.", exc_info=True)
            attempt.text_tokens = attempt.reasoning_tokens = attempt.tool_argument_tokens = None

        if isinstance(attempt.backend_completion_tokens, int) and attempt.backend_completion_tokens >= 0:
            attempt.total_tokens = attempt.backend_completion_tokens
            attempt.token_source = "backend_usage"
        elif None not in (attempt.text_tokens, attempt.reasoning_tokens, attempt.tool_argument_tokens):
            attempt.total_tokens = (
                attempt.text_tokens + attempt.reasoning_tokens + attempt.tool_argument_tokens
            )
            attempt.token_source = "tokenizer_estimate"
        else:
            attempt.total_tokens = None
            attempt.token_source = "unavailable"
        return attempt

    def finalize_all(self) -> None:
        for attempt in self.attempts:
            self.finalize_attempt(attempt)

    @property
    def total_tokens(self) -> int | None:
        self.finalize_all()
        if any(attempt.total_tokens is None for attempt in self.attempts):
            return None
        return sum(attempt.total_tokens or 0 for attempt in self.attempts)

    @property
    def token_source(self) -> str:
        self.finalize_all()
        sources = {attempt.token_source for attempt in self.attempts}
        if "unavailable" in sources:
            return "unavailable"
        return "backend_usage" if sources == {"backend_usage"} else "tokenizer_estimate"

    def category_tokens(self) -> tuple[int | None, int | None, int | None]:
        self.finalize_all()
        categories = (
            [attempt.text_tokens for attempt in self.attempts],
            [attempt.reasoning_tokens for attempt in self.attempts],
            [attempt.tool_argument_tokens for attempt in self.attempts],
        )
        return tuple(None if any(value is None for value in values) else sum(values) for values in categories)

    def output_window(self) -> tuple[float | None, float | None]:
        timestamps = [
            timestamp
            for attempt in self.attempts
            for timestamp in (attempt.first_output_at, attempt.last_output_at)
            if timestamp is not None
        ]
        return (min(timestamps), max(timestamps)) if timestamps else (None, None)


@dataclass
class StreamChoiceInfo:
    text: str = ""
    reasoning: str = ""
    tool_arguments: dict[str, str] = field(default_factory=dict)
    backend_completion_tokens: int | None = None
    stop_reason: str | None = None

    @property
    def has_output(self) -> bool:
        return bool(self.text or self.reasoning or any(self.tool_arguments.values()))


@dataclass
class DecoderStreamProgress:
    """Semantic stream progress observed after Chat SSE decoding."""

    first_valid_output_at: float | None = None
    last_valid_output_at: float | None = None
    first_valid_output_delay: float | None = None

    def observe_valid_output(self, observed_at: float | None = None) -> None:
        observed_at = time.monotonic() if observed_at is None else observed_at
        if self.first_valid_output_at is None:
            self.first_valid_output_at = observed_at
        self.last_valid_output_at = observed_at


class ChatStreamGuard:
    """Fail-closed evidence and semantic state for guarded Chat SSE streams."""

    def __init__(self, enabled: bool, trace_enabled: bool, max_bytes: int):
        self.enabled = enabled
        self.trace_enabled = trace_enabled
        self.max_bytes = max_bytes
        self.raw = bytearray()
        self.events: list[dict[str, Any]] = []
        self.has_semantic_output = False
        self.finish_reason: str | None = None
        self.done_seen = False
        self.empty_events_dropped = 0
        self.anomaly: str | None = None
        self.truncated = False

    def record(self, event: str, payload: dict[str, Any] | None = None) -> None:
        record = {"seq": len(self.events) + 1, "event": event}
        if payload is not None:
            record["payload"] = payload
        self.events.append(record)

    def record_raw(self, chunk: bytes) -> None:
        if self.truncated:
            return
        if len(self.raw) + len(chunk) <= self.max_bytes:
            self.raw.extend(chunk)
        else:
            self.truncated = True

    def inspect(self, payload: dict[str, Any], info: "StreamChoiceInfo") -> bool:
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason") or choice.get("stop_reason")
        if info.has_output:
            self.has_semantic_output = True
            self.record("semantic_output", payload)
            if info.tool_arguments:
                self.record("tool_call", payload)
        elif finish:
            self.finish_reason = str(finish)
            self.record("finish", payload)
            if not self.has_semantic_output and self.finish_reason != "tool_calls":
                self.anomaly = "empty_model_completion"
        elif delta.get("role") == "assistant" and not delta.get("tool_calls"):
            self.empty_events_dropped += 1
            self.record("role_only_empty_delta", payload)
            return False
        else:
            self.record("protocol_empty_delta", payload)
        return True

    def done(self) -> None:
        self.done_seen = True
        self.record("done")
        if not self.has_semantic_output and self.finish_reason is None:
            self.anomaly = self.anomaly or "empty_model_completion"


class ChatEmptyCompletionError(RuntimeError):
    """Decoder ended a guarded Chat turn without semantic output."""


def promote_chat_guard_trace(trace: "RequestAuditTrace | None", guard: ChatStreamGuard) -> None:
    if trace is None or not guard.trace_enabled or guard.anomaly is None:
        return
    trace.write_bytes("chat.response.raw.sse", bytes(guard.raw))
    for event in guard.events:
        trace.write_json("chat.response.events.jsonl", event, append=True)
    trace.write_json("chat.guard.json", {
        "anomaly_class": guard.anomaly,
        "has_semantic_output": guard.has_semantic_output,
        "finish_reason": guard.finish_reason,
        "done_seen": guard.done_seen,
        "empty_events_dropped": guard.empty_events_dropped,
        "trace_truncated": guard.truncated,
    })
    trace.event("chat_guard_anomaly", anomaly_class=guard.anomaly,
                empty_events_dropped=guard.empty_events_dropped,
                trace_truncated=guard.truncated)


def is_decoder_start_protocol_leak(
    stream_info: StreamChoiceInfo,
    attempt_has_valid_output: bool,
) -> bool:
    """Recognize only the Decoder's reserved marker before any real output."""
    return (
        not attempt_has_valid_output
        and stream_info.reasoning in DECODER_START_PROTOCOL_MARKERS
        and not stream_info.text
        and not stream_info.tool_arguments
    )


def decoder_start_protocol_marker_state(
    buffered_reasoning: str,
    stream_info: StreamChoiceInfo,
    attempt_has_valid_output: bool,
) -> Any:
    """Classify an initial reasoning fragment against the reserved marker."""
    if (
        attempt_has_valid_output
        or not stream_info.reasoning
        or stream_info.text
        or stream_info.tool_arguments
    ):
        return "not_marker"
    candidate = buffered_reasoning + stream_info.reasoning
    if not any(marker.startswith(candidate) for marker in DECODER_START_PROTOCOL_MARKERS):
        return "not_marker"
    return "marker" if candidate in DECODER_START_PROTOCOL_MARKERS else "prefix"


def append_decoder_start_protocol_retry_prompt(req_data: dict[str, Any]) -> bool:
    """Append an internal retry turn without rewriting the original request history."""
    messages = req_data.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            messages.append({"role": "user", "content": DECODER_START_PROTOCOL_RETRY_PROMPT})
            return True
        if isinstance(content, list):
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": DECODER_START_PROTOCOL_RETRY_PROMPT}],
            })
            return True
        return False
    return False


class LeakTracker:
    def __init__(self, certain_log_path: str, suspected_log_path: str, recent_limit: int):
        self._lock = threading.Lock()
        self._states: dict[str, str] = {}
        self._certain_log_path = certain_log_path
        self._suspected_log_path = suspected_log_path
        self._recent_limit = recent_limit

    def configure(self, certain_log_path: str, suspected_log_path: str, recent_limit: int) -> None:
        with self._lock:
            self._certain_log_path = certain_log_path
            self._suspected_log_path = suspected_log_path
            self._recent_limit = recent_limit

    def record(self, leak_state: RequestLeakState, leak_level: str, reason: str) -> None:
        if not leak_state.prefill_committed or leak_state.decoder_finished_normally or not leak_state.request_id:
            return
        with self._lock:
            previous_level = self._states.get(leak_state.request_id)
            if previous_level == LeakLevel.CERTAIN:
                return
            if previous_level == leak_level:
                return

            summary = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "request_id": leak_state.request_id,
                "leak_level": leak_level,
                "leak_reason": reason,
                "client_addr": leak_state.client_addr,
                "api": leak_state.api,
                "stream": leak_state.stream,
                "prefiller": leak_state.prefiller or "-",
                "decoder": leak_state.decoder or "-",
                "prefill_tokens": leak_state.input_tokens,
                "decoder_input_tokens": leak_state.decoder_input_tokens,
                "requested_max_tokens": leak_state.requested_max_tokens,
                "completion_tokens": leak_state.completion_tokens,
                "kv_transfer_params_received": leak_state.kv_transfer_params_received,
                "recomputed_seen": leak_state.recomputed_seen,
                "client_disconnected": leak_state.client_disconnected,
                "disconnect_phase": leak_state.disconnect_phase or None,
                "pipeline_phase": leak_state.pipeline_phase,
                "drain_after_prefill": leak_state.drain_after_prefill,
                "prefill_disconnect_draining": leak_state.prefill_disconnect_draining,
                "first_decoder_output_received": leak_state.first_decoder_output_received,
                "first_decoder_output_at": leak_state.first_decoder_output_at,
                "prefill_completed_at": leak_state.prefill_completed_at,
                "decoder_started_at": leak_state.decoder_started_at,
                "decoder_stream_closed_at": leak_state.decoder_stream_closed_at,
                "worker_finished_at": leak_state.worker_finished_at,
                "drain_stop_reason": leak_state.drain_stop_reason or None,
                "drain_stop_at": leak_state.drain_stop_at,
                "final_request_state": leak_state.final_request_state or "unknown",
                "prefill_disconnect_drain_started_at": leak_state.prefill_disconnect_drain_started_at,
                "prefill_disconnect_drain_ended_at": leak_state.prefill_disconnect_drain_ended_at,
                "drain_duration_seconds": round(
                    max(0.0, leak_state.prefill_disconnect_drain_ended_at - leak_state.prefill_disconnect_drain_started_at),
                    2,
                ) if leak_state.prefill_disconnect_drain_started_at and leak_state.prefill_disconnect_drain_ended_at else 0.0,
                "disconnect_detected_at": leak_state.disconnect_detected_at,
                "local_prefiller_kv_released_at": leak_state.local_prefiller_kv_released_at,
                "local_decoder_released_at": leak_state.local_decoder_released_at,
                "last_decoder_chunk_at": leak_state.last_decoder_chunk_at,
                "decoder_done_observed": leak_state.decoder_done_observed,
                "recomputed_attempts": leak_state.recomputed_attempts,
                "decode_attempts": leak_state.decode_attempts,
            }
            if previous_level == LeakLevel.SUSPECTED and leak_level == LeakLevel.CERTAIN:
                summary["upgraded_from"] = LeakLevel.SUSPECTED
            self._write_record(summary, leak_level)
            self._states[leak_state.request_id] = leak_level
            stats_collector.record_kv_leak(
                leak_level=leak_level,
                reason=reason,
                prefiller=leak_state.prefiller or None,
                decoder=leak_state.decoder or None,
                summary=summary,
                previous_level=previous_level,
            )
            leak_state.leak_level = leak_level
            leak_state.leak_reason = reason
            leak_state.leak_recorded = True

    def _write_record(self, summary: dict[str, Any], leak_level: str) -> None:
        log_path = self._certain_log_path if leak_level == LeakLevel.CERTAIN else self._suspected_log_path
        directory = os.path.dirname(log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")


class StatsCollector:
    """Collects and periodically prints request statistics per node."""

    def __init__(self):
        self._lock = threading.Lock()
        # Per-prefiller: key = "host:port"
        self.pf_reqs = {}
        self.pf_reqs_total = {}
        # Per-node recomputations in the current statistics interval.
        self.pf_recomputes = {}
        self.pf_input_tokens = {}
        self.pf_prefill_time = {}
        # Per-decoder: key = "host:port"
        self.dc_reqs = {}
        self.dc_recomputes = {}
        self.dc_output_tokens = {}
        self.dc_decode_time = {}
        self.dc_output_tokens_backend_usage = {}
        self.dc_output_tokens_estimated = {}
        self.dc_output_tokens_unavailable = {}
        self.dc_text_tokens = {}
        self.dc_reasoning_tokens = {}
        self.dc_tool_argument_tokens = {}
        self.dc_tpot_samples = {}
        self.dc_tpot_total_seconds = {}
        # Per-node rcvd within this stats period (reset each period)
        self.pf_rcvd_period = {}   # key = "host:port", prefiller rcvd this period
        self.dc_rcvd_period = {}   # key = "host:port", decoder rcvd this period
        # Global
        self.completed_reqs = 0   # requests that finished streaming this period
        self.rcvd_reqs = 0        # system-wide requests received this period
        self.rejected_reqs = 0
        self.error_total = 0
        self.error_by_category = {}
        self.error_by_status = {}
        self.error_by_role = {}
        self.error_by_node = {}
        self.recent_errors = deque(maxlen=20)
        self.certain_kv_leak_total_period = 0
        self.suspected_kv_leak_total_period = 0
        self.certain_kv_leak_open_total = 0
        self.suspected_kv_leak_open_total = 0
        self.certain_kv_leak_by_reason = {}
        self.suspected_kv_leak_by_reason = {}
        self.certain_kv_leak_by_prefiller = {}
        self.suspected_kv_leak_by_prefiller = {}
        self.certain_kv_leak_by_decoder = {}
        self.suspected_kv_leak_by_decoder = {}
        self.recent_certain_kv_leaks = deque(maxlen=20)
        self.recent_suspected_kv_leaks = deque(maxlen=20)
        self.prefill_disconnect_drain_started_total = 0
        self.prefill_disconnect_drain_active = 0
        self.prefill_disconnect_drain_finished_total = 0
        self.prefill_disconnect_drain_failed_total = 0
        self.prefill_disconnect_drain_active_requests = {}
        self.prefill_disconnect_drain_success_by_prefiller = {}
        self.prefill_disconnect_drain_success_by_decoder = {}
        self.ttft_requests = 0
        self.ttft_total_time = 0.0
        self.ttft_preprocess_time = 0.0
        self.ttft_dispatch_time = 0.0
        self.ttft_prefill_time = 0.0
        self.ttft_decode_wait_time = 0.0
        self.prefill_body_bytes = 0
        self.prefill_requests = 0
        self.prefill_400_invalid_json = 0
        self.prefill_transport_errors = 0
        self.prefill_retries = 0
        self.decode_transport_errors = 0
        self.failure_capture_saved_total = 0
        self.failure_capture_write_failed_total = 0
        self.failure_capture_by_kind = {}
        self.responses_sse_heartbeats_total = 0
        self.responses_sse_heartbeats_prefill = 0
        self.responses_sse_heartbeats_decoder_first_sse = 0
        # Cumulative counters for the narrow Responses DSML tool-tail workaround.
        self.responses_dsml_filter_requests_total = 0
        self.responses_dsml_filter_affected_requests_total = 0
        self.decoder_attempts_started = 0
        self.decoder_response_headers_received = 0
        self.decoder_first_sse_received = 0
        self.decoder_first_valid_output_received = 0
        self.pd_first_valid_output_timeout_total = 0
        self.decoder_stream_idle_timeout_total = 0
        self.decoder_first_response_waiting_total = 0
        self.decoder_stream_eof_without_done_total = 0
        self.decoder_first_response_waiting_active = {}
        self.disabled_abort_cleanup = {"enabled": 0}
        self.last_completed_period = None
        self.completed_period_id = 0
        # Track last reset time for interval calculation in snapshot
        self.last_reset_time = time.time()

    def record_prefill(self, host_port: str, input_tokens: int, prefill_time: float):
        with self._lock:
            self.pf_reqs[host_port] = self.pf_reqs.get(host_port, 0) + 1
            self.pf_reqs_total[host_port] = self.pf_reqs_total.get(host_port, 0) + 1
            self.pf_input_tokens[host_port] = self.pf_input_tokens.get(host_port, 0) + input_tokens
            self.pf_prefill_time[host_port] = self.pf_prefill_time.get(host_port, 0.0) + prefill_time

    def record_prefill_recompute(self, host_port: str) -> None:
        with self._lock:
            self.pf_recomputes[host_port] = self.pf_recomputes.get(host_port, 0) + 1

    def record_decode_recompute(self, host_port: str) -> None:
        with self._lock:
            self.dc_recomputes[host_port] = self.dc_recomputes.get(host_port, 0) + 1

    def record_decode(
        self,
        host_port: str,
        output_tokens: int | None,
        decode_time: float,
        *,
        token_source: str = "backend_usage",
        text_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        tool_argument_tokens: int | None = None,
        tpot_seconds: float | None = None,
    ):
        with self._lock:
            self.dc_reqs[host_port] = self.dc_reqs.get(host_port, 0) + 1
            if output_tokens is not None:
                self.dc_output_tokens[host_port] = self.dc_output_tokens.get(host_port, 0) + output_tokens
            if token_source == "backend_usage" and output_tokens is not None:
                self.dc_output_tokens_backend_usage[host_port] = (
                    self.dc_output_tokens_backend_usage.get(host_port, 0) + output_tokens
                )
            elif token_source == "tokenizer_estimate" and output_tokens is not None:
                self.dc_output_tokens_estimated[host_port] = (
                    self.dc_output_tokens_estimated.get(host_port, 0) + output_tokens
                )
            elif token_source == "unavailable":
                self.dc_output_tokens_unavailable[host_port] = (
                    self.dc_output_tokens_unavailable.get(host_port, 0) + 1
                )
            for target, value in (
                (self.dc_text_tokens, text_tokens),
                (self.dc_reasoning_tokens, reasoning_tokens),
                (self.dc_tool_argument_tokens, tool_argument_tokens),
            ):
                if value is not None:
                    target[host_port] = target.get(host_port, 0) + value
            if tpot_seconds is not None:
                self.dc_tpot_samples[host_port] = self.dc_tpot_samples.get(host_port, 0) + 1
                self.dc_tpot_total_seconds[host_port] = (
                    self.dc_tpot_total_seconds.get(host_port, 0.0) + tpot_seconds
                )
            self.dc_decode_time[host_port] = self.dc_decode_time.get(host_port, 0.0) + decode_time

    def record_pf_rcvd(self, host_port: str):
        with self._lock:
            self.pf_rcvd_period[host_port] = self.pf_rcvd_period.get(host_port, 0) + 1

    def record_dc_rcvd(self, host_port: str):
        with self._lock:
            self.dc_rcvd_period[host_port] = self.dc_rcvd_period.get(host_port, 0) + 1

    def record_rcvd(self):
        with self._lock:
            self.rcvd_reqs += 1

    def record_prefill_wire(self, body_bytes: int) -> None:
        with self._lock:
            self.prefill_requests += 1
            self.prefill_body_bytes += body_bytes

    def record_prefill_invalid_json(self) -> None:
        with self._lock:
            self.prefill_400_invalid_json += 1

    def record_prefill_transport_error(self) -> None:
        with self._lock:
            self.prefill_transport_errors += 1

    def record_prefill_retry(self) -> None:
        with self._lock:
            self.prefill_retries += 1

    def record_decode_transport_error(self) -> None:
        with self._lock:
            self.decode_transport_errors += 1

    def record_failure_capture_saved(self, failure_kind: str) -> None:
        with self._lock:
            self.failure_capture_saved_total += 1
            self.failure_capture_by_kind[failure_kind] = (
                self.failure_capture_by_kind.get(failure_kind, 0) + 1
            )

    def record_failure_capture_write_failed(self) -> None:
        with self._lock:
            self.failure_capture_write_failed_total += 1

    def record_responses_sse_heartbeat(self, phase: str) -> None:
        with self._lock:
            self.responses_sse_heartbeats_total += 1
            if phase == "prefill":
                self.responses_sse_heartbeats_prefill += 1
            elif phase == "decoder_first_sse":
                self.responses_sse_heartbeats_decoder_first_sse += 1

    def record_decoder_attempt_started(self) -> None:
        with self._lock:
            self.decoder_attempts_started += 1

    def record_decoder_response_headers(self) -> None:
        with self._lock:
            self.decoder_response_headers_received += 1

    def record_decoder_first_sse(self, attempt_id: str) -> None:
        with self._lock:
            self.decoder_first_sse_received += 1

    def set_decoder_first_response_phase(self, attempt_id: str, phase: str) -> None:
        with self._lock:
            item = self.decoder_first_response_waiting_active.get(attempt_id)
            if item is not None:
                item["phase"] = phase

    def record_decoder_first_valid_output(self, attempt_id: str) -> None:
        with self._lock:
            self.decoder_first_valid_output_received += 1
            self.decoder_first_response_waiting_active.pop(attempt_id, None)

    def record_pd_first_valid_output_timeout(self) -> None:
        with self._lock:
            self.pd_first_valid_output_timeout_total += 1

    def record_decoder_stream_idle_timeout(self) -> None:
        with self._lock:
            self.decoder_stream_idle_timeout_total += 1

    def start_decoder_first_response_waiting(
        self,
        attempt_id: str,
        node: str,
        phase: str,
        started_at: float,
    ) -> None:
        with self._lock:
            self.decoder_first_response_waiting_active[attempt_id] = {
                "node": node,
                "phase": phase,
                "started_at": started_at,
                "warning_recorded": False,
            }

    def record_decoder_first_response_waiting(
        self,
        attempt_id: str,
        node: str,
        phase: str,
        started_at: float,
    ) -> None:
        with self._lock:
            item = self.decoder_first_response_waiting_active.get(attempt_id)
            if item is None:
                item = {
                    "node": node,
                    "phase": phase,
                    "started_at": started_at,
                    "warning_recorded": False,
                }
                self.decoder_first_response_waiting_active[attempt_id] = item
            else:
                item["node"] = node
                item["phase"] = phase
            if not item["warning_recorded"]:
                self.decoder_first_response_waiting_total += 1
                item["warning_recorded"] = True

    def finish_decoder_first_response_waiting(self, attempt_id: str) -> None:
        with self._lock:
            self.decoder_first_response_waiting_active.pop(attempt_id, None)

    def record_decoder_stream_eof_without_done(self) -> None:
        with self._lock:
            self.decoder_stream_eof_without_done_total += 1

    def _decoder_observability_snapshot_locked(self) -> dict:
        now = time.time()
        active = [
            {
                "attempt_id": attempt_id,
                "node": item["node"],
                "phase": item["phase"],
                "age_seconds": round(max(0.0, now - item["started_at"]), 3),
            }
            for attempt_id, item in self.decoder_first_response_waiting_active.items()
        ]
        active.sort(key=lambda item: item["age_seconds"], reverse=True)
        return {
            "attempts_started": self.decoder_attempts_started,
            "response_headers_received": self.decoder_response_headers_received,
            "first_sse_received": self.decoder_first_sse_received,
            "first_valid_output_received": self.decoder_first_valid_output_received,
            "first_response_waiting_total": self.decoder_first_response_waiting_total,
            "first_valid_output_timeout_total": self.pd_first_valid_output_timeout_total,
            "stream_idle_timeout_total": self.decoder_stream_idle_timeout_total,
            "stream_eof_without_done_total": self.decoder_stream_eof_without_done_total,
            "first_response_waiting_active": len(active),
            "waiting_by_node": dict(Counter(item["node"] for item in active)),
            "longest_waiting_seconds": active[0]["age_seconds"] if active else 0.0,
            "longest_waiting_node": active[0]["node"] if active else None,
            "active_attempts": active,
        }

    def record_completed(self):
        with self._lock:
            self.completed_reqs += 1

    def record_responses_dsml_filter_request(self, *, affected: bool = False) -> None:
        """Count Responses requests inspected and requests with DSML text removed."""
        with self._lock:
            self.responses_dsml_filter_requests_total += 1
            if affected:
                self.responses_dsml_filter_affected_requests_total += 1

    def record_responses_dsml_filter_affected(self) -> None:
        with self._lock:
            self.responses_dsml_filter_affected_requests_total += 1

    def record_rejected(self):
        with self._lock:
            self.rejected_reqs += 1

    def record_ttft(
        self,
        ttft_total: float,
        preprocess_time: float,
        dispatch_time: float,
        prefill_time: float,
        decode_wait_time: float,
    ) -> None:
        with self._lock:
            self.ttft_requests += 1
            self.ttft_total_time += ttft_total
            self.ttft_preprocess_time += preprocess_time
            self.ttft_dispatch_time += dispatch_time
            self.ttft_prefill_time += prefill_time
            self.ttft_decode_wait_time += decode_wait_time

    def record_error(
        self,
        category: str,
        role: str = "proxy",
        node: str | None = None,
        status_code: int | None = None,
        endpoint: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            self.error_total += 1
            self.error_by_category[category] = self.error_by_category.get(category, 0) + 1
            self.error_by_role[role] = self.error_by_role.get(role, 0) + 1
            if status_code is not None:
                status_key = str(status_code)
                self.error_by_status[status_key] = self.error_by_status.get(status_key, 0) + 1
            if node is not None:
                node_stats = self.error_by_node.setdefault(
                    node,
                    {
                        "total": 0,
                        "by_category": {},
                        "by_status": {},
                        "by_role": {},
                    },
                )
                node_stats["total"] += 1
                node_stats["by_category"][category] = node_stats["by_category"].get(category, 0) + 1
                node_stats["by_role"][role] = node_stats["by_role"].get(role, 0) + 1
                if status_code is not None:
                    node_stats["by_status"][status_key] = node_stats["by_status"].get(status_key, 0) + 1

            if message:
                message = message.replace("\n", " ").strip()
                if len(message) > 240:
                    message = f"{message[:240]}..."
            self.recent_errors.append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "category": category,
                "role": role,
                "node": node,
                "status_code": status_code,
                "endpoint": endpoint,
                "message": message,
            })

    def record_disabled_abort_cleanup(self) -> None:
        with self._lock:
            self.disabled_abort_cleanup["enabled"] += 1

    def record_kv_leak(
        self,
        leak_level: str,
        reason: str,
        prefiller: str | None,
        decoder: str | None,
        summary: dict[str, Any],
        previous_level: str | None = None,
    ) -> None:
        with self._lock:
            if leak_level == LeakLevel.CERTAIN:
                self.certain_kv_leak_total_period += 1
                if previous_level == LeakLevel.SUSPECTED:
                    self.suspected_kv_leak_open_total = max(0, self.suspected_kv_leak_open_total - 1)
                self.certain_kv_leak_open_total += 1
                self.certain_kv_leak_by_reason[reason] = self.certain_kv_leak_by_reason.get(reason, 0) + 1
                if prefiller:
                    self.certain_kv_leak_by_prefiller[prefiller] = self.certain_kv_leak_by_prefiller.get(prefiller, 0) + 1
                if decoder:
                    self.certain_kv_leak_by_decoder[decoder] = self.certain_kv_leak_by_decoder.get(decoder, 0) + 1
                self.recent_certain_kv_leaks.append(summary)
                return

            if previous_level is None:
                self.suspected_kv_leak_total_period += 1
                self.suspected_kv_leak_open_total += 1
                self.suspected_kv_leak_by_reason[reason] = self.suspected_kv_leak_by_reason.get(reason, 0) + 1
                if prefiller:
                    self.suspected_kv_leak_by_prefiller[prefiller] = self.suspected_kv_leak_by_prefiller.get(prefiller, 0) + 1
                if decoder:
                    self.suspected_kv_leak_by_decoder[decoder] = self.suspected_kv_leak_by_decoder.get(decoder, 0) + 1
            self.recent_suspected_kv_leaks.append(summary)

    def record_prefill_disconnect_drain_started(self, leak_state: RequestLeakState) -> None:
        with self._lock:
            self.prefill_disconnect_drain_started_total += 1
            self.prefill_disconnect_drain_active += 1
            if leak_state.request_id:
                self.prefill_disconnect_drain_active_requests[leak_state.request_id] = {
                    "decoder": leak_state.decoder or "unknown",
                    "started_at": leak_state.prefill_disconnect_drain_started_at,
                }

    def record_prefill_disconnect_drain_finished(
        self,
        prefiller: str | None,
        decoder: str | None,
        request_id: str,
    ) -> None:
        with self._lock:
            self.prefill_disconnect_drain_active = max(0, self.prefill_disconnect_drain_active - 1)
            self.prefill_disconnect_drain_finished_total += 1
            self.prefill_disconnect_drain_active_requests.pop(request_id, None)
            if prefiller:
                self.prefill_disconnect_drain_success_by_prefiller[prefiller] = (
                    self.prefill_disconnect_drain_success_by_prefiller.get(prefiller, 0) + 1
                )
            if decoder:
                self.prefill_disconnect_drain_success_by_decoder[decoder] = (
                    self.prefill_disconnect_drain_success_by_decoder.get(decoder, 0) + 1
                )

    def record_prefill_disconnect_drain_failed(
        self,
        request_id: str,
    ) -> None:
        with self._lock:
            self.prefill_disconnect_drain_active = max(0, self.prefill_disconnect_drain_active - 1)
            self.prefill_disconnect_drain_failed_total += 1
            self.prefill_disconnect_drain_active_requests.pop(request_id, None)

    def _error_snapshot_locked(self) -> dict:
        return {
            "total": self.error_total,
            "by_category": dict(self.error_by_category),
            "by_status": dict(self.error_by_status),
            "by_role": dict(self.error_by_role),
            "by_node": {
                node: {
                    "total": stats["total"],
                    "by_category": dict(stats["by_category"]),
                    "by_status": dict(stats["by_status"]),
                    "by_role": dict(stats["by_role"]),
                }
                for node, stats in self.error_by_node.items()
            },
            "recent": list(self.recent_errors),
        }

    def _transport_snapshot_locked(self) -> dict:
        return {
            "prefill_requests": self.prefill_requests,
            "prefill_body_bytes": self.prefill_body_bytes,
            "prefill_400_invalid_json": self.prefill_400_invalid_json,
            "prefill_transport_errors": self.prefill_transport_errors,
            "prefill_retries": self.prefill_retries,
            "decode_transport_errors": self.decode_transport_errors,
            "failure_capture_saved_total": self.failure_capture_saved_total,
            "failure_capture_write_failed_total": self.failure_capture_write_failed_total,
            "failure_capture_by_kind": dict(self.failure_capture_by_kind),
        }

    def _responses_sse_heartbeat_snapshot_locked(self) -> dict:
        return {
            "total": self.responses_sse_heartbeats_total,
            "prefill": self.responses_sse_heartbeats_prefill,
            "decoder_first_sse": self.responses_sse_heartbeats_decoder_first_sse,
        }

    def _kv_leak_snapshot_locked(self) -> dict:
        active_by_decoder = {}
        oldest_started_at = 0.0
        for item in self.prefill_disconnect_drain_active_requests.values():
            decoder = item["decoder"]
            active_by_decoder[decoder] = active_by_decoder.get(decoder, 0) + 1
            started_at = item["started_at"]
            if started_at and (not oldest_started_at or started_at < oldest_started_at):
                oldest_started_at = started_at
        return {
            "certain": {
                "period": self.certain_kv_leak_total_period,
                "open_total": self.certain_kv_leak_open_total,
                "by_reason": dict(self.certain_kv_leak_by_reason),
                "by_prefiller": dict(self.certain_kv_leak_by_prefiller),
                "by_decoder": dict(self.certain_kv_leak_by_decoder),
                "recent": list(self.recent_certain_kv_leaks),
            },
            "suspected": {
                "period": self.suspected_kv_leak_total_period,
                "open_total": self.suspected_kv_leak_open_total,
                "by_reason": dict(self.suspected_kv_leak_by_reason),
                "by_prefiller": dict(self.suspected_kv_leak_by_prefiller),
                "by_decoder": dict(self.suspected_kv_leak_by_decoder),
                "recent": list(self.recent_suspected_kv_leaks),
            },
            "abort_cleanup_disabled": dict(self.disabled_abort_cleanup),
            "prefill_disconnect_drain": {
                "started_total": self.prefill_disconnect_drain_started_total,
                "active": self.prefill_disconnect_drain_active,
                "finished_total": self.prefill_disconnect_drain_finished_total,
                "failed_total": self.prefill_disconnect_drain_failed_total,
                "success_by_prefiller": dict(self.prefill_disconnect_drain_success_by_prefiller),
                "success_by_decoder": dict(self.prefill_disconnect_drain_success_by_decoder),
                "active_oldest_seconds": round(max(0.0, time.time() - oldest_started_at), 2)
                if oldest_started_at else 0.0,
                "active_by_decoder": active_by_decoder,
            },
        }

    def _fmt_row(self, role, host, reqs, tokens, total_time, interval, is_prefill=True,
                 busy_override=None, pressure_override=None, health=None,
                 rcvd=None, completed=None, inflight=None,
                 recomputes=None, req_total=None, leak_certain_total=None,
                 leak_suspected_total=None, drain_success_total=None):
        """Format one table row with aligned columns.

        busy_override: if provided, use this for Busy% (avg of per-node capped values for total row).
        pressure_override: if provided, use this for Pressure% (avg of per-node raw values for total row).
        health: optional health string (e.g. "100.00%(5/0)", "(taint)50.00%(2/2)", "(unkn)").
        rcvd/completed/inflight: optional integers for new stats columns; inflight shown as "-" for per-node rows.
        """
        avg_time = f"{total_time / reqs:.2f}" if reqs > 0 else "-"
        # Busy%: capped at 100% per node (wall-clock utilization)
        if busy_override is not None:
            busy = f"{busy_override:.2f}%" if reqs > 0 else "-"
        else:
            raw_busy = total_time / interval * 100 if interval > 0 and reqs > 0 else 0
            busy = f"{min(raw_busy, 100.0):.2f}%" if reqs > 0 else "-"
        # Pressure%: uncapped (total_time / interval * 100), shows scheduling pressure.
        if pressure_override is not None:
            pressure = f"{pressure_override:.2f}%" if reqs > 0 else "-"
        else:
            raw_pressure = total_time / interval * 100 if interval > 0 and reqs > 0 else 0
            pressure = f"{raw_pressure:.2f}%" if reqs > 0 else "-"
        speed = f"{tokens / total_time:.2f}" if total_time > 0 else "-"
        avg_speed = f"{tokens / interval:.2f}" if interval > 0 and reqs > 0 else "-"
        tok_label = "input" if is_prefill else "output"
        health_str = f"{health:>19}" if health else "-".ljust(19)
        rcvd_str = f"{rcvd:>5}" if rcvd is not None else "-".ljust(5)
        completed_str = f"{completed:>9}" if completed is not None else "-".ljust(9)
        inflight_str = f"{inflight:>8}" if inflight is not None else "-".ljust(8)
        recomputes_str = f"{recomputes:>9}" if recomputes is not None else "-".ljust(9)
        req_total_str = f"{req_total:>9}" if req_total is not None else "-".ljust(9)
        leak_certain_str = f"{leak_certain_total:>9}" if leak_certain_total is not None else "-".ljust(9)
        leak_suspected_str = f"{leak_suspected_total:>9}" if leak_suspected_total is not None else "-".ljust(9)
        drain_success_str = f"{drain_success_total:>9}" if drain_success_total is not None else "-".ljust(9)
        return (f"| {role:<8} | {host:<22} | {inflight_str} | {reqs:>5} | {recomputes_str} | "
                f"{tok_label}={tokens:<10} | {avg_time:>9} | {busy:>7} | {pressure:>9} | "
                f"{speed:>10} tok/s | {avg_speed:>10} tok/s | {health_str} | {rcvd_str} | "
                f"{completed_str} | {req_total_str} | {leak_certain_str} | "
                f"{leak_suspected_str} | {drain_success_str} |")

    @staticmethod
    def _format_count_pairs(counts: dict[str, int]) -> str:
        if not counts:
            return "-"
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))

    def print_and_reset(self, interval: int, prefiller_nodes: list = None, decoder_nodes: list = None,
                        proxy_state=None):
        if proxy_state is not None:
            proxy_state.reset_decoder_period_assignments()
        with self._lock:
            pf_reqs = dict(self.pf_reqs)
            pf_reqs_total = dict(self.pf_reqs_total)
            pf_recomputes = dict(self.pf_recomputes)
            pf_input_tokens = dict(self.pf_input_tokens)
            pf_prefill_time = dict(self.pf_prefill_time)
            dc_reqs = dict(self.dc_reqs)
            dc_recomputes = dict(self.dc_recomputes)
            dc_output_tokens = dict(self.dc_output_tokens)
            dc_decode_time = dict(self.dc_decode_time)
            dc_output_tokens_backend_usage = dict(self.dc_output_tokens_backend_usage)
            dc_output_tokens_estimated = dict(self.dc_output_tokens_estimated)
            dc_output_tokens_unavailable = dict(self.dc_output_tokens_unavailable)
            dc_text_tokens = dict(self.dc_text_tokens)
            dc_reasoning_tokens = dict(self.dc_reasoning_tokens)
            dc_tool_argument_tokens = dict(self.dc_tool_argument_tokens)
            dc_tpot_samples = dict(self.dc_tpot_samples)
            dc_tpot_total_seconds = dict(self.dc_tpot_total_seconds)
            pf_rcvd_period = dict(self.pf_rcvd_period)
            dc_rcvd_period = dict(self.dc_rcvd_period)
            completed_reqs = self.completed_reqs
            rcvd_reqs = self.rcvd_reqs
            rejected_reqs = self.rejected_reqs
            errors = self._error_snapshot_locked()
            transport = self._transport_snapshot_locked()
            responses_sse_heartbeats = self._responses_sse_heartbeat_snapshot_locked()
            responses_dsml_filter = {
                "total_requests": self.responses_dsml_filter_requests_total,
                "affected_requests": self.responses_dsml_filter_affected_requests_total,
            }
            decoder_observability = self._decoder_observability_snapshot_locked()
            kv_leaks = self._kv_leak_snapshot_locked()
            ttft_requests = self.ttft_requests
            ttft_total_time = self.ttft_total_time
            ttft_preprocess_time = self.ttft_preprocess_time
            ttft_dispatch_time = self.ttft_dispatch_time
            ttft_prefill_time = self.ttft_prefill_time
            ttft_decode_wait_time = self.ttft_decode_wait_time

            self.completed_period_id += 1
            self.last_completed_period = {
                "period_id": self.completed_period_id,
                "ended_at": time.time(),
                "interval_seconds": float(interval),
                "total_requests": completed_reqs,
            }

            self.pf_reqs.clear()
            self.pf_recomputes.clear()
            self.pf_input_tokens.clear()
            self.pf_prefill_time.clear()
            self.dc_reqs.clear()
            self.dc_recomputes.clear()
            self.dc_output_tokens.clear()
            self.dc_decode_time.clear()
            self.dc_output_tokens_backend_usage.clear()
            self.dc_output_tokens_estimated.clear()
            self.dc_output_tokens_unavailable.clear()
            self.dc_text_tokens.clear()
            self.dc_reasoning_tokens.clear()
            self.dc_tool_argument_tokens.clear()
            self.dc_tpot_samples.clear()
            self.dc_tpot_total_seconds.clear()
            self.completed_reqs = 0
            self.rcvd_reqs = 0
            self.pf_rcvd_period.clear()
            self.dc_rcvd_period.clear()
            self.rejected_reqs = 0
            self.error_total = 0
            self.error_by_category.clear()
            self.error_by_status.clear()
            self.error_by_role.clear()
            self.error_by_node.clear()
            self.recent_errors.clear()
            self.ttft_requests = 0
            self.ttft_total_time = 0.0
            self.ttft_preprocess_time = 0.0
            self.ttft_dispatch_time = 0.0
            self.ttft_prefill_time = 0.0
            self.ttft_decode_wait_time = 0.0
            self.prefill_body_bytes = 0
            self.prefill_requests = 0
            self.prefill_400_invalid_json = 0
            self.prefill_transport_errors = 0
            self.prefill_retries = 0
            self.decode_transport_errors = 0
            self.failure_capture_saved_total = 0
            self.failure_capture_write_failed_total = 0
            self.failure_capture_by_kind.clear()
            self.responses_sse_heartbeats_total = 0
            self.responses_sse_heartbeats_prefill = 0
            self.responses_sse_heartbeats_decoder_first_sse = 0
            self.decoder_attempts_started = 0
            self.decoder_response_headers_received = 0
            self.decoder_first_sse_received = 0
            self.decoder_first_valid_output_received = 0
            self.pd_first_valid_output_timeout_total = 0
            self.decoder_stream_idle_timeout_total = 0
            self.decoder_first_response_waiting_total = 0
            self.decoder_stream_eof_without_done_total = 0
            self.last_reset_time = time.time()

        # Build full node key lists from registered nodes (not just those with requests)
        all_pf_keys = sorted(set(f"{s.host}:{s.port}" for s in (prefiller_nodes or [])))
        all_dc_keys = sorted(set(f"{s.host}:{s.port}" for s in (decoder_nodes or [])))
        prefiller_active = {}
        decoder_active = {}
        inflight = 0

        # Gather health stats from NodeListener
        health_stats = proxy_state.node_listener.get_health_stats() if proxy_state else {}
        healthcheck_summary = (
            proxy_state.node_listener.get_healthcheck_summary() if proxy_state else {}
        )
        if proxy_state:
            stats_pf_nodes, stats_dc_nodes, prefiller_active, decoder_active, inflight = proxy_state.snapshot_for_stats()
            all_pf_keys = sorted(set(f"{s.host}:{s.port}" for s in stats_pf_nodes))
            all_dc_keys = sorted(set(f"{s.host}:{s.port}" for s in stats_dc_nodes))
            _, _, tainted_pf_nodes, tainted_dc_nodes, _, _ = proxy_state.snapshot_for_health_check()
            tainted_pfs = {str(s) for s in tainted_pf_nodes}
            tainted_dcs = {str(s) for s in tainted_dc_nodes}
        else:
            tainted_pfs = set()
            tainted_dcs = set()

        def fmt_node_health(key, is_tainted):
            stats = health_stats.get(key, {})
            return NodeListener.fmt_health(stats.get("ok", 0), stats.get("fail", 0), is_tainted)

        # Total health across all backends
        total_ok = sum(s.get("ok", 0) for s in health_stats.values())
        total_fail = sum(s.get("fail", 0) for s in health_stats.values())
        total_health = NodeListener.fmt_health(total_ok, total_fail, False)

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        sep = (f"+{'-'*10}+{'-'*24}+{'-'*10}+{'-'*7}+{'-'*11}+{'-'*18}+{'-'*11}+{'-'*9}"
               f"+{'-'*11}+{'-'*14}+{'-'*14}+{'-'*21}+{'-'*7}+{'-'*11}+{'-'*10}"
               f"+{'-'*11}+{'-'*11}+{'-'*11}+")
        header = (f"| {'Role':<8} | {'Host:Port':<22} | {'active_now':>8} | {'Reqs':>5} | {'Recompute':>9} | "
                  f"{'Tokens':<18} | {'AvgTime(s)':>11} | {'Busy%':>7} | {'Pressure%':>9} | "
                  f"{'Speed':>14} | {'AvgSpeed':>14} | {'Health':>21} | {'rcvd':>7} | "
                  f"{'completed':>11} | {'req_total':>9} | {'leak_c':>9} | "
                  f"{'leak_s':>9} | {'drain_ok':>9} |")

        print(f"--- Stats {now} (last {interval}s) ---", flush=True)
        ttft_summary = ""
        if ttft_requests > 0:
            ttft_summary = (
                " | --- TTFT Summary --- "
                f"requests={ttft_requests} avg_total={ttft_total_time / ttft_requests:.2f}s "
                f"avg_prep={ttft_preprocess_time / ttft_requests:.2f}s "
                f"avg_dispatch={ttft_dispatch_time / ttft_requests:.2f}s "
                f"avg_prefill={ttft_prefill_time / ttft_requests:.2f}s "
                f"avg_decode_wait={ttft_decode_wait_time / ttft_requests:.2f}s"
            )
        print(
            "period_requests: "
            f"received={rcvd_reqs} completed={completed_reqs}"
            f"{ttft_summary}",
            flush=True,
        )
        if healthcheck_summary:
            print(
                "health_probe "
                f"idle={healthcheck_summary['probed_idle']} "
                f"tainted={healthcheck_summary['probed_tainted']} "
                f"waiting={healthcheck_summary['probed_waiting']} "
                f"skipped_recent_success={healthcheck_summary['skipped_recent_success']} "
                f"business_success={healthcheck_summary['business_success']} "
                f"ttl={healthcheck_summary['success_ttl_seconds']}s",
                flush=True,
            )
        print(
            "decoder_first_response "
            f"waiting_active={decoder_observability['first_response_waiting_active']} "
            f"waiting_total={decoder_observability['first_response_waiting_total']} "
            f"longest={decoder_observability['longest_waiting_seconds']:.1f}s "
            f"node={decoder_observability['longest_waiting_node'] or '-'}",
            flush=True,
        )
        print(sep, flush=True)
        print(header, flush=True)
        print(sep, flush=True)

        # Per-request average (client perspective: one request = prefill + decode)
        total_input = sum(pf_input_tokens.values())
        total_output = sum(dc_output_tokens.values())
        total_pf_time = sum(pf_prefill_time.values())
        total_dc_time = sum(dc_decode_time.values())
        total_time = total_pf_time + total_dc_time
        total_tokens = total_input + total_output
        avg_time = total_time / completed_reqs if completed_reqs > 0 else 0
        speed = f"{total_tokens / total_time:.2f}" if total_time > 0 else "-"
        avg_speed = f"{total_tokens / interval:.2f}" if interval > 0 and completed_reqs > 0 else "-"
        tok_info = f"in={total_input}+out={total_output}"
        total_pf_recomputes = sum(pf_recomputes.values())
        total_dc_recomputes = sum(dc_recomputes.values())
        if completed_reqs > 0:
            print(f"| {'Request':<8} | {'avg/req':<22} | {inflight:>8} | {completed_reqs:>5} | "
                  f"{'pf=' + str(total_pf_recomputes) + '/dc=' + str(total_dc_recomputes):>9} | "
                  f"{tok_info:<18} | {avg_time:>9.2f} | {'-':>7} | {'-':>9} | {speed:>10} tok/s | "
                  f"{avg_speed:>10} tok/s | {total_health:>21} | {rcvd_reqs:>7} | "
                  f"{completed_reqs:>11} | {'-':>9} | {'-':>9} | {'-':>9} | {'-':>9} |", flush=True)
        # Total prefill — Busy% = avg of per-node capped, Pressure% = avg of per-node raw
        total_pf_reqs = sum(pf_reqs.values())
        total_pf_input = sum(pf_input_tokens.values())
        total_pf_time = sum(pf_prefill_time.values())
        total_pf_rcvd = sum(pf_rcvd_period.values())
        pf_node_count = len(all_pf_keys) if all_pf_keys else 1
        pf_avg_busy = sum(
            min(pf_prefill_time.get(k, 0.0) / interval * 100, 100.0)
            for k in all_pf_keys
        ) / pf_node_count if interval > 0 else 0
        pf_avg_load = sum(
            pf_prefill_time.get(k, 0.0) / interval * 100
            for k in all_pf_keys
        ) / pf_node_count if interval > 0 else 0
        total_pf_active = sum(prefiller_active.get(k, 0) for k in all_pf_keys)
        print(self._fmt_row("Prefill", "total", total_pf_reqs, total_pf_input, total_pf_time,
                            interval, True, pf_avg_busy, pf_avg_load, total_health,
                            rcvd=total_pf_rcvd, completed=total_pf_reqs, inflight=total_pf_active,
                            recomputes=total_pf_recomputes,
                            req_total=sum(pf_reqs_total.values()),
                            leak_certain_total=sum(kv_leaks["certain"]["by_prefiller"].values()),
                            leak_suspected_total=sum(kv_leaks["suspected"]["by_prefiller"].values()),
                            drain_success_total=sum(kv_leaks["prefill_disconnect_drain"]["success_by_prefiller"].values())), flush=True)

        # Total decode
        total_dc_reqs = sum(dc_reqs.values())
        total_dc_output = sum(dc_output_tokens.values())
        total_dc_time = sum(dc_decode_time.values())
        total_dc_rcvd = sum(dc_rcvd_period.values())
        dc_node_count = len(all_dc_keys) if all_dc_keys else 1
        dc_avg_busy = sum(
            min(dc_decode_time.get(k, 0.0) / interval * 100, 100.0)
            for k in all_dc_keys
        ) / dc_node_count if interval > 0 else 0
        dc_avg_load = sum(
            dc_decode_time.get(k, 0.0) / interval * 100
            for k in all_dc_keys
        ) / dc_node_count if interval > 0 else 0
        total_dc_active = sum(decoder_active.get(k, 0) for k in all_dc_keys)
        print(self._fmt_row("Decode", "total", total_dc_reqs, total_dc_output, total_dc_time,
                            interval, False, dc_avg_busy, dc_avg_load, total_health,
                            rcvd=total_dc_rcvd, completed=total_dc_reqs, inflight=total_dc_active,
                            recomputes=total_dc_recomputes), flush=True)

        # Rejected
        if rejected_reqs > 0:
            print(f"| {'Reject':<8} | {'total':<22} | {'-':>8} | {rejected_reqs:>5} | {'-':>9} | "
                  f"{'-':<18} | {'-':>9} | {'-':>7} | {'-':>9} | {'-':>14} | {'-':>14} | "
                  f"{'-':>21} | {'-':>7} | {'-':>11} | {'-':>9} | {'-':>9} | {'-':>9} | {'-':>9} |", flush=True)
        if errors["total"] > 0:
            error_tokens = ",".join(f"{k}={v}" for k, v in sorted(errors["by_category"].items()))
            if len(error_tokens) > 18:
                error_tokens = f"{error_tokens[:15]}..."
            print(f"| {'Error':<8} | {'upstream/proxy':<22} | {'-':>8} | {errors['total']:>5} | {'-':>9} | "
                  f"{error_tokens:<18} | {'-':>9} | {'-':>7} | {'-':>9} | {'-':>14} | {'-':>14} | "
                  f"{'-':>21} | {'-':>7} | {'-':>11} | {'-':>9} | {'-':>9} | {'-':>9} | {'-':>9} |", flush=True)

        print(sep, flush=True)

        # Per prefiller — iterate ALL registered nodes, show 0 for idle ones
        for key in all_pf_keys:
            reqs = pf_reqs.get(key, 0)
            tokens = pf_input_tokens.get(key, 0)
            elapsed = pf_prefill_time.get(key, 0.0)
            print(self._fmt_row("Prefill", key, reqs, tokens, elapsed, interval, True,
                                health=fmt_node_health(key, key in tainted_pfs),
                                rcvd=pf_rcvd_period.get(key, 0),
                                completed=pf_reqs.get(key, 0),
                                inflight=prefiller_active.get(key, 0),
                                recomputes=pf_recomputes.get(key, 0),
                                req_total=pf_reqs_total.get(key, 0),
                                leak_certain_total=kv_leaks["certain"]["by_prefiller"].get(key, 0),
                                leak_suspected_total=kv_leaks["suspected"]["by_prefiller"].get(key, 0),
                                drain_success_total=kv_leaks["prefill_disconnect_drain"]["success_by_prefiller"].get(key, 0)), flush=True)

        # Per decoder — iterate ALL registered nodes, show 0 for idle ones
        for key in all_dc_keys:
            reqs = dc_reqs.get(key, 0)
            tokens = dc_output_tokens.get(key, 0)
            elapsed = dc_decode_time.get(key, 0.0)
            print(self._fmt_row("Decode", key, reqs, tokens, elapsed, interval, False,
                                health=fmt_node_health(key, key in tainted_dcs),
                                rcvd=dc_rcvd_period.get(key, 0),
                                completed=dc_reqs.get(key, 0),
                                inflight=decoder_active.get(key, 0),
                                recomputes=dc_recomputes.get(key, 0)), flush=True)

        print(sep, flush=True)
        if errors["total"] > 0:
            print("--- Error Summary ---", flush=True)
            print(f"total={errors['total']}", flush=True)
            print(f"by_category: {self._format_count_pairs(errors['by_category'])}", flush=True)
            print(f"by_status: {self._format_count_pairs(errors['by_status'])}", flush=True)
            print(f"by_role: {self._format_count_pairs(errors['by_role'])}", flush=True)
            if errors["by_node"]:
                print("by_node:", flush=True)
                for node, node_stats in sorted(errors["by_node"].items()):
                    print(
                        f"  {node}: total={node_stats['total']}, "
                        f"category[{self._format_count_pairs(node_stats['by_category'])}], "
                        f"status[{self._format_count_pairs(node_stats['by_status'])}], "
                        f"role[{self._format_count_pairs(node_stats['by_role'])}]",
                        flush=True,
                    )
            if errors["recent"]:
                print("recent_errors:", flush=True)
                for item in errors["recent"]:
                    status = item["status_code"] if item["status_code"] is not None else "-"
                    node = item["node"] or "-"
                    endpoint = item["endpoint"] or "-"
                    message = item["message"] or "-"
                    print(
                        f"  {item['time']} | {item['category']} | role={item['role']} | "
                        f"node={node} | status={status} | endpoint={endpoint} | {message}",
                        flush=True,
                    )
        if kv_leaks["certain"]["period"] > 0 or kv_leaks["certain"]["open_total"] > 0:
            print("--- Certain KV Leaks ---", flush=True)
            print(
                f"total={kv_leaks['certain']['period']} open_total={kv_leaks['certain']['open_total']}",
                flush=True,
            )
            print(
                f"by_reason: {self._format_count_pairs(kv_leaks['certain']['by_reason'])}",
                flush=True,
            )
            print(
                f"by_prefiller: {self._format_count_pairs(kv_leaks['certain']['by_prefiller'])}",
                flush=True,
            )
            print(
                f"by_decoder: {self._format_count_pairs(kv_leaks['certain']['by_decoder'])}",
                flush=True,
            )
            if kv_leaks["certain"]["recent"]:
                print("recent_certain_kv_leaks:", flush=True)
                for item in kv_leaks["certain"]["recent"]:
                    print(
                        f"  {item['timestamp']} | request_id={item['request_id']} | "
                        f"reason={item['leak_reason']} | prefiller={item['prefiller']} | decoder={item['decoder']}",
                        flush=True,
                    )
        if kv_leaks["suspected"]["period"] > 0 or kv_leaks["suspected"]["open_total"] > 0:
            print("--- Suspected KV Leaks ---", flush=True)
            print(
                f"total={kv_leaks['suspected']['period']} open_total={kv_leaks['suspected']['open_total']}",
                flush=True,
            )
            print(
                f"by_reason: {self._format_count_pairs(kv_leaks['suspected']['by_reason'])}",
                flush=True,
            )
            print(
                f"by_prefiller: {self._format_count_pairs(kv_leaks['suspected']['by_prefiller'])}",
                flush=True,
            )
            print(
                f"by_decoder: {self._format_count_pairs(kv_leaks['suspected']['by_decoder'])}",
                flush=True,
            )
            if kv_leaks["suspected"]["recent"]:
                print("recent_suspected_kv_leaks:", flush=True)
                for item in kv_leaks["suspected"]["recent"]:
                    print(
                        f"  {item['timestamp']} | request_id={item['request_id']} | "
                        f"reason={item['leak_reason']} | prefiller={item['prefiller']} | decoder={item['decoder']}",
                        flush=True,
                    )
        if (
            kv_leaks["prefill_disconnect_drain"]["started_total"] > 0
            or kv_leaks["prefill_disconnect_drain"]["active"] > 0
            or kv_leaks["prefill_disconnect_drain"]["finished_total"] > 0
            or kv_leaks["prefill_disconnect_drain"]["failed_total"] > 0
        ):
            print("--- Prefill Disconnect Drain Summary ---", flush=True)
            print(
                "started_total={started_total} active={active} finished_total={finished_total} failed_total={failed_total}".format(
                    **kv_leaks["prefill_disconnect_drain"]
                ),
                flush=True,
            )
        if responses_sse_heartbeats["total"] > 0:
            print(
                "responses_sse_heartbeats total={total} prefill={prefill} decoder_first_sse={decoder_first_sse}".format(
                    **responses_sse_heartbeats
                ),
                flush=True,
            )
        print(f"--- End Stats ---", flush=True)


    def snapshot(self, prefiller_nodes: list = None, decoder_nodes: list = None) -> dict:
        """Return current stats as a dict for /metrics endpoint (does NOT reset)."""
        with self._lock:
            pf_reqs = dict(self.pf_reqs)
            pf_reqs_total = dict(self.pf_reqs_total)
            pf_input_tokens = dict(self.pf_input_tokens)
            pf_prefill_time = dict(self.pf_prefill_time)
            dc_reqs = dict(self.dc_reqs)
            dc_output_tokens = dict(self.dc_output_tokens)
            dc_decode_time = dict(self.dc_decode_time)
            dc_output_tokens_backend_usage = dict(self.dc_output_tokens_backend_usage)
            dc_output_tokens_estimated = dict(self.dc_output_tokens_estimated)
            dc_output_tokens_unavailable = dict(self.dc_output_tokens_unavailable)
            dc_text_tokens = dict(self.dc_text_tokens)
            dc_reasoning_tokens = dict(self.dc_reasoning_tokens)
            dc_tool_argument_tokens = dict(self.dc_tool_argument_tokens)
            dc_tpot_samples = dict(self.dc_tpot_samples)
            dc_tpot_total_seconds = dict(self.dc_tpot_total_seconds)
            pf_rcvd_period = dict(self.pf_rcvd_period)
            dc_rcvd_period = dict(self.dc_rcvd_period)
            completed_reqs = self.completed_reqs
            rcvd_reqs = self.rcvd_reqs
            rejected_reqs = self.rejected_reqs
            errors = self._error_snapshot_locked()
            transport = self._transport_snapshot_locked()
            responses_sse_heartbeats = self._responses_sse_heartbeat_snapshot_locked()
            responses_dsml_filter = {
                "total_requests": self.responses_dsml_filter_requests_total,
                "affected_requests": self.responses_dsml_filter_affected_requests_total,
            }
            decoder_observability = self._decoder_observability_snapshot_locked()
            kv_leaks = self._kv_leak_snapshot_locked()
            ttft_requests = self.ttft_requests
            ttft_total_time = self.ttft_total_time
            ttft_preprocess_time = self.ttft_preprocess_time
            ttft_dispatch_time = self.ttft_dispatch_time
            ttft_prefill_time = self.ttft_prefill_time
            ttft_decode_wait_time = self.ttft_decode_wait_time
            last_reset_time = self.last_reset_time
            last_completed_period = dict(self.last_completed_period) if self.last_completed_period else None

        interval = time.time() - last_reset_time if last_reset_time else 0
        all_pf_keys = sorted(set(f"{s.host}:{s.port}" for s in (prefiller_nodes or [])))
        all_dc_keys = sorted(set(f"{s.host}:{s.port}" for s in (decoder_nodes or [])))
        healthcheck_summary = {}
        backend_success = {}
        health_stats = {}
        proxy_requests = {"running": 0, "waiting": 0}
        if proxy_state is not None:
            healthcheck_summary = proxy_state.node_listener.get_healthcheck_summary()
            backend_success = proxy_state.snapshot_backend_success()
            health_stats = proxy_state.node_listener.get_health_stats()
            proxy_requests["running"] = proxy_state.get_request_num()

        # Per-request average
        total_input = sum(pf_input_tokens.values())
        total_output = sum(dc_output_tokens.values())
        total_pf_time = sum(pf_prefill_time.values())
        total_dc_time = sum(dc_decode_time.values())
        total_time = total_pf_time + total_dc_time
        total_tokens = total_input + total_output

        # Prefill totals
        pf_total_reqs = sum(pf_reqs.values())
        pf_total_tokens = sum(pf_input_tokens.values())
        pf_avg_busy = (sum(min(pf_prefill_time.get(k, 0.0) / interval * 100, 100.0)
                        for k in all_pf_keys) / len(all_pf_keys)) if interval > 0 and all_pf_keys else 0
        pf_avg_load = (sum(pf_prefill_time.get(k, 0.0) / interval * 100
                        for k in all_pf_keys) / len(all_pf_keys)) if interval > 0 and all_pf_keys else 0

        # Decode totals
        dc_total_reqs = sum(dc_reqs.values())
        dc_total_tokens = sum(dc_output_tokens.values())
        dc_avg_busy = (sum(min(dc_decode_time.get(k, 0.0) / interval * 100, 100.0)
                        for k in all_dc_keys) / len(all_dc_keys)) if interval > 0 and all_dc_keys else 0
        dc_avg_load = (sum(dc_decode_time.get(k, 0.0) / interval * 100
                        for k in all_dc_keys) / len(all_dc_keys)) if interval > 0 and all_dc_keys else 0

        result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interval_seconds": round(interval, 2),
            "config": {
                "prefiller_max_model_len": prefiller_max_model_len,
                "decoder_max_model_len": decoder_max_model_len,
                "stats_interval": global_args.stats_interval,
                "stats_enabled": global_args.stats_enabled,
                "certain_kv_leak_log_path": global_args.certain_kv_leak_log_path,
                "suspected_kv_leak_log_path": global_args.suspected_kv_leak_log_path,
                "num_prefillers": len(all_pf_keys),
                "num_decoders": len(all_dc_keys),
                "prefill_connection_mode": PREFILL_CONNECTION_MODE,
                "prefill_max_connections": PREFILL_MAX_CONNECTIONS,
                "prefill_max_keepalive_connections": PREFILL_MAX_KEEPALIVE_CONNECTIONS,
                "healthcheck_success_ttl_seconds": getattr(
                    global_args, "healthcheck_success_ttl_seconds", 60.0
                ),
                "periodic_backend_healthcheck_enabled": getattr(
                    global_args, "enable_periodic_backend_healthcheck", False
                ),
                "responses_sse_heartbeat_seconds": getattr(
                    global_args, "responses_sse_heartbeat_seconds", 10.0
                ),
                "decoder_first_response_observe_seconds": getattr(
                    global_args, "decoder_first_response_observe_seconds", 30.0
                ),
                "decoder_first_response_log_interval_seconds": getattr(
                    global_args, "decoder_first_response_log_interval_seconds", 60.0
                ),
                "pd_first_output_timeout_seconds": getattr(
                    global_args, "pd_first_output_timeout_seconds", 480.0
                ),
                "decoder_stream_idle_timeout_seconds": getattr(
                    global_args, "decoder_stream_idle_timeout_seconds", 480.0
                ),
            },
            "request_avg": {
                "received_requests": rcvd_reqs,
                "total_requests": completed_reqs,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "avg_time_per_request": round(total_time / completed_reqs, 2) if completed_reqs > 0 else 0,
                "speed_tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            },
            "last_completed_period": last_completed_period,
            "prefill_total": {
                "received": sum(pf_rcvd_period.values()),
                "requests": pf_total_reqs,
                "input_tokens": pf_total_tokens,
                "total_time": round(total_pf_time, 2),
                "avg_time": round(total_pf_time / pf_total_reqs, 2) if pf_total_reqs > 0 else 0,
                "busy_pct": round(pf_avg_busy, 2),
                "load_pct": round(pf_avg_load, 2),
                "pressure_pct": round(pf_avg_load, 2),
                "speed_tokens_per_second": round(pf_total_tokens / total_pf_time, 2) if total_pf_time > 0 else 0,
                "avg_speed_tokens_per_second": round(pf_total_tokens / interval, 2) if interval > 0 else 0,
            },
            "decode_total": {
                "received": sum(dc_rcvd_period.values()),
                "requests": dc_total_reqs,
                "output_tokens": dc_total_tokens,
                "total_time": round(total_dc_time, 2),
                "avg_time": round(total_dc_time / dc_total_reqs, 2) if dc_total_reqs > 0 else 0,
                "busy_pct": round(dc_avg_busy, 2),
                "load_pct": round(dc_avg_load, 2),
                "pressure_pct": round(dc_avg_load, 2),
                "speed_tokens_per_second": round(dc_total_tokens / total_dc_time, 2) if total_dc_time > 0 else 0,
                "avg_speed_tokens_per_second": round(dc_total_tokens / interval, 2) if interval > 0 else 0,
                "output_tokens_backend_usage": sum(dc_output_tokens_backend_usage.values()),
                "output_tokens_estimated": sum(dc_output_tokens_estimated.values()),
                "output_tokens_unavailable": sum(dc_output_tokens_unavailable.values()),
                "text_tokens": sum(dc_text_tokens.values()),
                "reasoning_tokens": sum(dc_reasoning_tokens.values()),
                "tool_argument_tokens": sum(dc_tool_argument_tokens.values()),
                "tpot_samples": sum(dc_tpot_samples.values()),
                "tpot_total_seconds": round(sum(dc_tpot_total_seconds.values()), 4),
            },
            "ttft": {
                "requests": ttft_requests,
                "avg_total": round(ttft_total_time / ttft_requests, 2) if ttft_requests > 0 else 0,
                "avg_preprocess": round(ttft_preprocess_time / ttft_requests, 2) if ttft_requests > 0 else 0,
                "avg_dispatch": round(ttft_dispatch_time / ttft_requests, 2) if ttft_requests > 0 else 0,
                "avg_prefill": round(ttft_prefill_time / ttft_requests, 2) if ttft_requests > 0 else 0,
                "avg_decode_wait": round(ttft_decode_wait_time / ttft_requests, 2) if ttft_requests > 0 else 0,
                "total_time": round(ttft_total_time, 2),
            },
            "rejected": rejected_reqs,
            "errors": errors,
            "certain_kv_leaks": kv_leaks["certain"],
                "suspected_kv_leaks": kv_leaks["suspected"],
            "transport": transport,
            "responses_sse_heartbeats": responses_sse_heartbeats,
            "responses_dsml_filter": responses_dsml_filter,
            "decoder_observability": decoder_observability,
            "decoder_taint": proxy_state.snapshot_decoder_taint_metrics() if proxy_state is not None else {},
            # Running is all active client requests in the proxy. Waiting is
            # the subset currently accepted by a Decoder but lacking its first
            # valid output; it is intentionally not a second queue length.
            "proxy_requests": {
                "running": proxy_requests["running"],
                "waiting": decoder_observability["first_response_waiting_active"],
            },
            "abort_cleanup_disabled": kv_leaks["abort_cleanup_disabled"],
            "prefill_disconnect_drain": kv_leaks["prefill_disconnect_drain"],
            "healthcheck": healthcheck_summary,
            "prefill_nodes": [],
            "decode_nodes": [],
        }

        for key in all_pf_keys:
            reqs = pf_reqs.get(key, 0)
            tokens = pf_input_tokens.get(key, 0)
            elapsed = pf_prefill_time.get(key, 0.0)
            busy = min(elapsed / interval * 100, 100.0) if interval > 0 and reqs > 0 else 0
            load = elapsed / interval * 100 if interval > 0 and reqs > 0 else 0
            host, port = key.rsplit(":", 1)
            result["prefill_nodes"].append({
                "host": host,
                "port": int(port),
                "received": pf_rcvd_period.get(key, 0),
                "completed": reqs,
                "requests": reqs,
                "request_total": pf_reqs_total.get(key, 0),
                "input_tokens": tokens,
                "avg_time": round(elapsed / reqs, 2) if reqs > 0 else 0,
                "busy_pct": round(busy, 2),
                "load_pct": round(load, 2),
                "pressure_pct": round(load, 2),
                "speed_tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
                "avg_speed_tokens_per_second": round(tokens / interval, 2) if interval > 0 else 0,
                "leak_certain_total": kv_leaks["certain"]["by_prefiller"].get(key, 0),
                "leak_suspected_total": kv_leaks["suspected"]["by_prefiller"].get(key, 0),
                "drain_success_total": kv_leaks["prefill_disconnect_drain"]["success_by_prefiller"].get(key, 0),
                "healthcheck": health_stats.get(key, {}),
                **backend_success.get(key, {}),
                "errors": errors["by_node"].get(key, {
                    "total": 0,
                    "by_category": {},
                    "by_status": {},
                    "by_role": {},
                }),
            })

        for key in all_dc_keys:
            reqs = dc_reqs.get(key, 0)
            tokens = dc_output_tokens.get(key, 0)
            elapsed = dc_decode_time.get(key, 0.0)
            busy = min(elapsed / interval * 100, 100.0) if interval > 0 and reqs > 0 else 0
            load = elapsed / interval * 100 if interval > 0 and reqs > 0 else 0
            host, port = key.rsplit(":", 1)
            result["decode_nodes"].append({
                "host": host,
                "port": int(port),
                "received": dc_rcvd_period.get(key, 0),
                "completed": reqs,
                "requests": reqs,
                "output_tokens": tokens,
                "output_tokens_backend_usage": dc_output_tokens_backend_usage.get(key, 0),
                "output_tokens_estimated": dc_output_tokens_estimated.get(key, 0),
                "output_tokens_unavailable": dc_output_tokens_unavailable.get(key, 0),
                "text_tokens": dc_text_tokens.get(key, 0),
                "reasoning_tokens": dc_reasoning_tokens.get(key, 0),
                "tool_argument_tokens": dc_tool_argument_tokens.get(key, 0),
                "tpot_samples": dc_tpot_samples.get(key, 0),
                "tpot_total_seconds": round(dc_tpot_total_seconds.get(key, 0.0), 4),
                "healthcheck": health_stats.get(key, {}),
                **backend_success.get(key, {}),
                "avg_time": round(elapsed / reqs, 2) if reqs > 0 else 0,
                "busy_pct": round(busy, 2),
                "load_pct": round(load, 2),
                "pressure_pct": round(load, 2),
                "speed_tokens_per_second": round(tokens / elapsed, 2) if elapsed > 0 else 0,
                "avg_speed_tokens_per_second": round(tokens / interval, 2) if interval > 0 else 0,
                "errors": errors["by_node"].get(key, {
                    "total": 0,
                    "by_category": {},
                    "by_status": {},
                    "by_role": {},
                }),
            })

        return result


stats_collector = StatsCollector()
leak_tracker = LeakTracker(
    DEFAULT_CERTAIN_KV_LEAK_LOG_PATH,
    DEFAULT_SUSPECTED_KV_LEAK_LOG_PATH,
    20,
)


async def validate_backend_model_name(proxy: ProxyState, backend_model_name: str | None) -> None:
    """Validate model IDs and quarantine nodes that are temporarily unreachable."""
    if not backend_model_name:
        return
    nodes = [*proxy.prefillers, *proxy.decoders]
    checks = await asyncio.gather(
        *(node.client.get("/models") for node in nodes),
        return_exceptions=True,
    )
    invalid_nodes = []
    unreachable_prefillers = []
    unreachable_decoders = []
    for node, result in zip(nodes, checks):
        if isinstance(result, httpx.RequestError):
            if node in proxy.prefillers:
                unreachable_prefillers.append(node)
            else:
                unreachable_decoders.append(node)
            logger.warning(
                "backend_unreachable_at_startup node=%s failure_category=healthcheck error=%s",
                node, type(result).__name__,
            )
            continue
        if isinstance(result, Exception):
            invalid_nodes.append(f"{node} (models request failed: {type(result).__name__})")
            continue
        try:
            result.raise_for_status()
            payload = result.json()
            model_ids = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            }
        except (httpx.HTTPError, ValueError, AttributeError):
            invalid_nodes.append(f"{node} (invalid /v1/models response)")
            continue
        if backend_model_name not in model_ids:
            available = ",".join(sorted(model_ids)) or "none"
            invalid_nodes.append(f"{node} (available={available})")
    if invalid_nodes:
        raise ValueError(
            f"Configured backend model {backend_model_name!r} is unavailable on: "
            + "; ".join(invalid_nodes)
        )
    if unreachable_prefillers:
        proxy._taint_prefillers(unreachable_prefillers)
    if unreachable_decoders:
        if getattr(global_args, "decoder_taint_enabled", DECODER_TAINT_ENABLED):
            for node in unreachable_decoders:
                proxy.record_decoder_failure_observation(node, "healthcheck")
                proxy.taint_decoder(node, reason="healthcheck")
        else:
            invalid_nodes.extend(f"{node} (unreachable and decoder taint disabled)" for node in unreachable_decoders)
    if invalid_nodes:
        raise ValueError(
            f"Configured backend model {backend_model_name!r} is unavailable on: "
            + "; ".join(invalid_nodes)
        )
    verified = len(nodes) - len(unreachable_prefillers) - len(unreachable_decoders)
    print(
        f"Verified forced backend model {backend_model_name!r} on {verified}/{len(nodes)} PD nodes; "
        f"startup_tainted={len(unreachable_prefillers) + len(unreachable_decoders)}."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state, tokenizer, prefiller_max_model_len, decoder_max_model_len, proxy_shutdown_started, multimodal_tokenize_client, multimodal_tokenize_semaphore
    validate_parsers()
    proxy_shutdown_started = False
    initialize_failure_capture_storage()
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    multimodal_tokenize_client = httpx.AsyncClient(trust_env=False)
    multimodal_tokenize_semaphore = asyncio.Semaphore(global_args.multimodal_tokenize_max_concurrency)
    try:
        await validate_backend_model_name(proxy_state, global_args.backend_model_name)
    except Exception:
        for node in [*proxy_state.prefillers, *proxy_state.decoders]:
            await node.client.aclose()
            await node.prefill_client.aclose()
        if multimodal_tokenize_client is not None:
            await multimodal_tokenize_client.aclose()
            multimodal_tokenize_client = None
        raise
    leak_tracker.configure(
        global_args.certain_kv_leak_log_path,
        global_args.suspected_kv_leak_log_path,
        global_args.kv_leak_recent_limit,
    )
    print(
        f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients. "
        "prefill_disconnect_drain=phase_aware "
        f"periodic_backend_healthcheck_enabled="
        f"{getattr(global_args, 'enable_periodic_backend_healthcheck', False)}"
    )

    # Start stats background task
    stats_task = None

    async def stats_loop():
        while True:
            await asyncio.sleep(global_args.stats_interval)
            stats_collector.print_and_reset(
                global_args.stats_interval,
                proxy_state.prefillers,
                proxy_state.decoders,
                proxy_state,
            )

    if global_args.stats_enabled:
        stats_task = asyncio.create_task(stats_loop())

    if not global_args.model_path:
        raise ValueError("--model-path is required because the proxy now requires a tokenizer for all requests.")

    # Try AutoTokenizer first; fall back to direct tokenizer.json loading
    # for models with non-standard tokenizer_class (e.g. "TokenizersBackend")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            global_args.model_path, trust_remote_code=True, local_files_only=True
        )
    except (ValueError, ImportError):
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        tokenizer_path = os.path.join(global_args.model_path, "tokenizer.json")
        fast_tokenizer = Tokenizer.from_file(tokenizer_path)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast_tokenizer)
        chat_template_path = os.path.join(global_args.model_path, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            with open(chat_template_path) as f:
                tokenizer.chat_template = f.read()
        config_path = os.path.join(global_args.model_path, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                token_config = json.load(f)
            if "eos_token" in token_config and tokenizer.eos_token is None:
                tokenizer.eos_token = token_config["eos_token"]
            if "pad_token" in token_config and tokenizer.pad_token is None:
                tokenizer.pad_token = token_config["pad_token"]

    validate_tokenizer_for_proxy(tokenizer)

    config_max_position_embeddings = None
    config_path = os.path.join(global_args.model_path, "config.json")
    with open(config_path) as f:
        config_max_position_embeddings = json.load(f).get("max_position_embeddings")
    if global_args.prefiller_max_model_len is not None:
        prefiller_max_model_len = global_args.prefiller_max_model_len
    elif config_max_position_embeddings is not None:
        prefiller_max_model_len = config_max_position_embeddings
    if global_args.decoder_max_model_len is not None:
        decoder_max_model_len = global_args.decoder_max_model_len
    elif config_max_position_embeddings is not None:
        decoder_max_model_len = config_max_position_embeddings
    print(f"Loaded tokenizer from {global_args.model_path}, "
          f"prefiller_max_model_len={prefiller_max_model_len}, "
          f"decoder_max_model_len={decoder_max_model_len}")
    if prefiller_max_model_len is not None and prefiller_max_model_len <= SAFE_CONTEXT_BUFFER_TOKENS:
        raise ValueError(
            f"prefiller_max_model_len ({prefiller_max_model_len}) must be greater than "
            f"SAFE_CONTEXT_BUFFER_TOKENS ({SAFE_CONTEXT_BUFFER_TOKENS})."
        )
    if decoder_max_model_len is not None and decoder_max_model_len <= SAFE_CONTEXT_BUFFER_TOKENS:
        raise ValueError(
            f"decoder_max_model_len ({decoder_max_model_len}) must be greater than "
            f"SAFE_CONTEXT_BUFFER_TOKENS ({SAFE_CONTEXT_BUFFER_TOKENS})."
        )
    if (prefiller_max_model_len is not None and decoder_max_model_len is not None
            and prefiller_max_model_len >= decoder_max_model_len):
        raise ValueError(
            f"prefiller_max_model_len ({prefiller_max_model_len}) must be less than "
            f"decoder_max_model_len ({decoder_max_model_len}). "
            f"In PD disaggregated deployment, the decoder must have a larger context window "
            f"than the prefiller."
        )
    yield
    # Shutdown: cancel stats task and close clients
    proxy_shutdown_started = True
    if stats_task is not None:
        stats_task.cancel()
    await detached_task_manager.shutdown(timeout=5.0)
    for p in proxy_state.prefillers:
        await p.client.aclose()
        await p.prefill_client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()
        await d.prefill_client.aclose()
    if multimodal_tokenize_client is not None:
        await multimodal_tokenize_client.aclose()
        multimodal_tokenize_client = None


def count_encoded_tokens(encoded) -> int:
    if isinstance(encoded, dict) and "input_ids" in encoded:
        input_ids = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif hasattr(encoded, "ids"):
        return len(encoded.ids)
    elif isinstance(encoded, list) and encoded and all(hasattr(item, "ids") for item in encoded):
        # TokenizersBackend chat templating in newer transformers can return
        # a list of tokenizers.Encoding objects instead of raw token ids.
        return sum(len(item.ids) for item in encoded)
    else:
        return len(encoded)

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _prompt_text_from_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_prompt_text_from_value(item) for item in value)
    if isinstance(value, dict):
        return visible_text_for_token_count(value)
    return ""


def visible_text_for_token_count(content: Any) -> str:
    """Mirror the model chat template's visible_text behavior safely.

    Unknown OpenAI content-part types such as image_url, file, audio, or custom
    mappings are not rendered as text by the template and must not inflate the
    local context-length guard.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts)
    if isinstance(content, dict) and content.get("type") == "text":
        text = content.get("text", "")
        return text if isinstance(text, str) else ""
    return ""


def normalize_chat_messages_for_token_count(messages: Any) -> list[dict]:
    """Normalize chat messages to the text that the model template can see."""
    if not isinstance(messages, list):
        return []

    normalized = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            continue

        message = {
            "role": raw_message.get("role"),
            "content": visible_text_for_token_count(raw_message.get("content", "")),
        }
        if isinstance(raw_message.get("reasoning_content"), str):
            message["reasoning_content"] = raw_message["reasoning_content"]

        # Tool calls are rendered by the chat template. Validation happens at
        # the forwarding boundary; use the same canonical representation here
        # so token accounting cannot silently diverge from the backend payload.
        if isinstance(raw_message.get("tool_calls"), list):
            tool_calls = []
            for raw_tool_call in raw_message["tool_calls"]:
                if not isinstance(raw_tool_call, dict):
                    continue
                tool_call = dict(raw_tool_call)
                if isinstance(tool_call.get("function"), dict):
                    function = dict(tool_call["function"])
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        function["arguments"] = json.loads(arguments)
                    elif not isinstance(arguments, dict):
                        function["arguments"] = arguments
                    tool_call["function"] = function
                tool_calls.append(tool_call)
            if tool_calls:
                message["tool_calls"] = tool_calls
        normalized.append(message)
    return normalized


def _infer_thinking_mode(req_data: dict) -> str:
    settings = resolve_reasoning_settings(req_data, source="chat")
    return "thinking" if settings["thinking_enabled"] else "chat"


def _encode_reasoning_effort(req_data: dict) -> str | None:
    return resolve_reasoning_settings(req_data, source="chat")["backend_effort"]


def _build_dsv4_messages_for_count(req_data: dict, for_prefill: bool) -> tuple[list[dict], str]:
    if _has_multimodal_content(req_data.get("messages", [])):
        raise ValueError("Image inputs require upstream V4.1 tokenization.")
    messages = copy.deepcopy(req_data.get("messages", []))
    if not messages and for_prefill:
        prompt_text = _prompt_text_from_value(req_data.get("prompt", ""))
        messages = [{"role": "user", "content": prompt_text}]
    # Match the serving tokenizer's normalization, including its text-part
    # separator and reasoning alias. The V4.1 encoder handles tool merging and
    # ordering itself; doing that before encoding can change the prompt.
    for msg in messages:
        if msg.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Invalid V4.1 role: {msg.get('role')!r}")
        if "reasoning" in msg:
            msg["reasoning_content"] = msg["reasoning"]
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = "\n\n".join(
                part.get("text", "") for part in content
                if part.get("type") in {"text", "input_text", "output_text"}
            )
    if req_data.get("tools"):
        system = next((m for m in messages if m.get("role") == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            messages.insert(0, system)
        system["tools"] = copy.deepcopy(req_data["tools"])
    thinking_mode = _infer_thinking_mode(req_data)
    return messages, thinking_mode


def _encode_dsv4_prompt_for_count(req_data: dict, for_prefill: bool) -> str:
    if encode_dsv4_messages is None:
        raise RuntimeError("DeepSeek-V4.1 encoder is unavailable.")

    messages, thinking_mode = _build_dsv4_messages_for_count(req_data, for_prefill=for_prefill)
    return encode_dsv4_messages(
        messages,
        thinking_mode=thinking_mode,
        reasoning_effort=_encode_reasoning_effort(req_data),
        drop_thinking=(req_data.get("chat_template_kwargs") or {}).get("drop_thinking", True),
    )


def normalize_chat_messages_for_backend(messages: Any) -> list[dict]:
    """Validate content without losing image parts or their order."""
    if not isinstance(messages, list):
        raise ResponsesRequestError("messages must be an array.", "invalid_input_content")
    normalized = []
    for raw_message in messages:
        if not isinstance(raw_message, dict):
            raise ResponsesRequestError("Invalid message.", "invalid_input_content")
        message = copy.deepcopy(raw_message)
        if "content" in message and message["content"] is not None:
            message["content"] = _responses_multimodal_content(message["content"])
        normalized.append(message)
    return normalized


def resolve_reasoning_settings(req_data: dict, *, source: str) -> dict[str, Any]:
    """Resolve client semantics once; template efforts always use native semantics.

    The returned metadata is kept in proxy state/trace, never in the wire body.
    Unknown strings deliberately use Codex Medium (native high). Missing effort
    uses the explicitly configured low default.
    """
    if source not in {"responses", "chat"}:
        raise ValueError(f"Unknown reasoning source: {source}")
    template = req_data.get("chat_template_kwargs")
    if template is None:
        template = {}
    if not isinstance(template, dict):
        raise ResponsesRequestError("chat_template_kwargs must be an object.", "invalid_thinking")

    switches = []
    if "thinking" in req_data:
        thinking = req_data["thinking"]
        if (not isinstance(thinking, dict) or set(thinking) != {"type"}
                or thinking.get("type") not in ("enabled", "disabled")):
            raise ResponsesRequestError(
                "thinking must be an object with type enabled or disabled.", "invalid_thinking"
            )
        switches.append(thinking["type"] == "enabled")
    for key in ("enable_thinking", "thinking"):
        if key in template:
            if type(template[key]) is not bool:
                raise ResponsesRequestError(f"chat_template_kwargs.{key} must be a boolean.", "invalid_thinking")
            switches.append(template[key])
    if switches and any(value != switches[0] for value in switches):
        raise ResponsesRequestError("Conflicting thinking switches.", "conflicting_reasoning_parameters")

    disabled_aliases = {"none", "off", "disabled", "disable"}
    codex_mapping = {
        "light": "low", "low": "low", "minimal": "low", "medium": "high",
        "high": "max", "xhigh": "max", "extra_high": "max", "extra high": "max", "max": "max",
    }

    def parse_effort(raw: Any, semantics: str) -> tuple[str | None, str | None, bool]:
        if raw is not None and not isinstance(raw, str):
            raise ResponsesRequestError("reasoning effort must be a string or null.", "invalid_reasoning_effort")
        value = raw.strip().lower() if isinstance(raw, str) else ""
        if not value:
            return "low", "missing_effort", False
        if value in disabled_aliases:
            return None, None, True
        mapping = codex_mapping if semantics == "responses" else {v: v for v in ("low", "high", "max")}
        return mapping.get(value, "high"), (None if value in mapping else "unknown_effort"), False

    raw = req_data.get("reasoning_effort")
    template_raw = template.get("reasoning_effort")
    primary = parse_effort(raw, source)
    secondary = parse_effort(template_raw, "chat")
    primary_present = raw is not None and (not isinstance(raw, str) or bool(raw.strip()))
    template_present = template_raw is not None and (not isinstance(template_raw, str) or bool(template_raw.strip()))
    selected = primary if primary_present or not template_present else secondary
    raw_selected = raw if primary_present or not template_present else template_raw
    enabled = switches[0] if switches else not (primary[2] or secondary[2])
    if switches and enabled and (primary[2] or secondary[2]):
        raise ResponsesRequestError("Thinking is enabled but effort requests disabling it.", "conflicting_reasoning_parameters")
    if enabled and primary_present and template_present and primary[0] != secondary[0]:
        raise ResponsesRequestError("Conflicting entry and template reasoning efforts.", "conflicting_reasoning_parameters")
    if not switches and primary_present and template_present and primary[2] != secondary[2]:
        raise ResponsesRequestError("Conflicting entry and template reasoning efforts.", "conflicting_reasoning_parameters")

    final_effort = selected[0] if enabled else None
    normalized_raw = raw_selected.strip().lower() if isinstance(raw_selected, str) else None
    if not enabled:
        effective = "none"
    elif selected[1] == "missing_effort":
        effective = "low"
    elif selected[1] is not None:
        effective = "medium"
    elif source == "responses" and primary_present:
        effective = {"light": "low", "minimal": "low", "extra_high": "xhigh", "extra high": "xhigh"}.get(normalized_raw, normalized_raw)
    else:
        effective = normalized_raw
    return {
        "source": source,
        "raw_effort": raw,
        "raw_template_effort": template_raw,
        "effective_client_effort": effective,
        "thinking_enabled": enabled,
        "backend_effort": final_effort,
        "default_reason": selected[1] if enabled and selected[1] == "missing_effort" else None,
        "fallback_reason": selected[1] if enabled and selected[1] == "unknown_effort" else None,
    }


def normalize_reasoning_effort_for_backend(
    req_data: dict, *, source: str = "chat", reasoning_settings: dict | None = None,
) -> dict:
    """Serialize resolved V4.1 settings without adding internal wire fields."""
    settings = reasoning_settings if reasoning_settings is not None else resolve_reasoning_settings(req_data, source=source)
    backend_req_data = copy.deepcopy(req_data)
    backend_req_data.pop("thinking", None)
    backend_req_data.pop("reasoning_effort", None)
    template = backend_req_data.get("chat_template_kwargs") or {}
    template.pop("thinking", None)
    template.pop("reasoning_effort", None)
    template["enable_thinking"] = settings["thinking_enabled"]
    backend_req_data["chat_template_kwargs"] = template
    if settings["thinking_enabled"]:
        backend_req_data["reasoning_effort"] = settings["backend_effort"]
    if reasoning_settings is None:
        logger.info("reasoning_normalized %s", json.dumps(settings, ensure_ascii=True))
    return backend_req_data


TOOL_IMAGE_FAILURE_TEXT = (
    "[图片读取失败：图片数据损坏或被截断，未提供给模型。"
    "如需识图，请重新生成或读取该图片；若这是损坏图片测试，请将其记录为读取失败。]"
)


def replace_corrupt_tool_images(messages: list[dict]) -> None:
    """Repair only the request's working copy; never discard user images."""
    checked: dict[str, bool] = {}
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, list):
            continue
        for content_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url = part["image_url"]["url"]
            if not url.startswith("data:"):
                continue
            digest = hashlib.sha256(url.encode()).hexdigest()
            if digest not in checked:
                # URL syntax, strict base64 and byte limits were checked by
                # normalize_chat_messages_for_backend before this step.
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.verify()
                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                except (OSError, SyntaxError, ValueError):
                    checked[digest] = False
                else:
                    checked[digest] = True
            if not checked[digest]:
                content[content_index] = {"type": "text", "text": TOOL_IMAGE_FAILURE_TEXT}
                logger.warning(
                    "corrupt_tool_image_replaced message_index=%d content_index=%d "
                    "image_sha256=%s reason=image_decode_failed",
                    message_index, content_index, digest,
                )


def prepare_backend_request_data(req_data: dict, *, reasoning_settings: dict | None = None) -> dict:
    """Build a backend-safe request while preserving structured image content."""
    backend_req_data = normalize_reasoning_effort_for_backend(req_data, reasoning_settings=reasoning_settings)
    backend_model_name = getattr(globals().get("global_args"), "backend_model_name", None)
    requested_model = backend_req_data.get("model")
    if backend_model_name and requested_model != backend_model_name:
        backend_req_data["model"] = backend_model_name
        logger.info(
            "model_rewritten client_request_id=%s requested_model=%r backend_model=%r",
            _active_client_request_id.get(),
            requested_model,
            backend_model_name,
        )
    if "messages" in backend_req_data:
        backend_req_data["messages"] = normalize_chat_messages_for_backend(backend_req_data["messages"])
        validate_chat_tool_calls(backend_req_data["messages"])
        replace_corrupt_tool_images(backend_req_data["messages"])
    return backend_req_data


def validate_tokenizer_for_proxy(_tokenizer: Any) -> None:
    if _tokenizer is None:
        raise ValueError("Tokenizer initialization failed.")
    if not hasattr(_tokenizer, "encode"):
        raise ValueError(
            "Tokenizer must implement encode because the proxy counts tokens from rendered DeepSeek-V4.1 prompts."
        )
    if encode_dsv4_messages is None:
        raise ValueError(
            "DeepSeek-V4.1 encoder is unavailable; cannot compute prompt lengths."
        )
    try:
        rendered = _encode_dsv4_prompt_for_count({"messages": [{"role": "user", "content": "ping"}]}, False)
        count_encoded_tokens(_tokenizer.encode(rendered, add_special_tokens=False))
    except Exception as exc:
        raise ValueError(
            "Tokenizer validation failed; local token counting would diverge from DeepSeek-V4.1 prompt formatting."
        ) from exc


def count_tokens(req_data: dict, _tokenizer, for_prefill: bool = False) -> int:
    """Count input tokens using the serving-compatible DeepSeek-V4.1 encoder."""

    if "messages" in req_data or for_prefill:
        try:
            rendered = _encode_dsv4_prompt_for_count(req_data, for_prefill=for_prefill)
            return count_encoded_tokens(_tokenizer.encode(rendered, add_special_tokens=False))
        except Exception as exc:
            raise ValueError("Unable to encode request with the DeepSeek-V4.1 format.") from exc

    prompt = _prompt_text_from_value(req_data.get("prompt", ""))
    return count_encoded_tokens(_tokenizer.encode(prompt))


def prepare_prefill_request_data(req_data: dict) -> dict:
    """Build the request shape that the prefiller actually tokenizes."""
    prefill_req_data = copy.deepcopy(req_data)
    if "prompt" in prefill_req_data and "messages" not in prefill_req_data:
        prompt_text = prefill_req_data.pop("prompt")
        if isinstance(prompt_text, list):
            prompt_text = "".join(p if isinstance(p, str) else p.get("text", "") for p in prompt_text)
        prefill_req_data["messages"] = [{"role": "user", "content": prompt_text}]
    return prefill_req_data


def count_prefill_and_decode_tokens(req_data: dict, _tokenizer) -> tuple[int, int]:
    """Return token counts for the actual prefiller and decoder request shapes."""
    prefill_tokens = count_tokens(req_data, _tokenizer, for_prefill=True)
    decoder_tokens = count_tokens(req_data, _tokenizer)
    return prefill_tokens, decoder_tokens


def image_input_count(value: Any) -> int:
    if isinstance(value, list):
        return sum(image_input_count(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in ("input_image", "image_url"):
            return 1
        return sum(image_input_count(item) for item in value.values())
    return 0


def build_multimodal_tokenize_payload(req_data: dict) -> dict:
    """Use only rendering/processor inputs, never generation or KV parameters."""
    keys = (
        "model", "messages", "tools", "chat_template",
        "chat_template_kwargs", "add_generation_prompt", "continue_final_message",
        "mm_processor_kwargs", "media_io_kwargs",
    )
    payload = {key: copy.deepcopy(req_data[key]) for key in keys if key in req_data}
    template = payload.setdefault("chat_template_kwargs", {})
    template.pop("thinking", None)
    template.pop("reasoning_effort", None)
    if req_data.get("reasoning_effort") is not None:
        template["reasoning_effort"] = req_data["reasoning_effort"]
    payload["add_special_tokens"] = False
    return payload


async def _tokenize_image_request(req_data: dict, role: str, node: ServerState) -> int:
    origin = node.url.rsplit("/v1", 1)[0]
    stage = f"Image tokenize {role} node={origin}"
    timeout = getattr(globals().get("global_args"), "multimodal_tokenize_timeout_seconds", 30.0)

    async def perform():
        if multimodal_tokenize_client is None or multimodal_tokenize_semaphore is None:
            raise ResponsesRequestError(f"{stage}: client unavailable.", "multimodal_tokenize_unavailable", 502)
        async with multimodal_tokenize_semaphore:
            response = await multimodal_tokenize_client.post(
                origin + "/tokenize", json=build_multimodal_tokenize_payload(req_data), timeout=timeout,
            )
        # Do not include upstream bodies: loaders may echo data URLs.
        if response.status_code in {400, 413, 415, 422}:
            raise ResponsesRequestError(f"{stage}: invalid image or rendering parameters (HTTP {response.status_code}).",
                                        "multimodal_tokenize_invalid_input", 400)
        if response.status_code != 200:
            raise ResponsesRequestError(f"{stage}: upstream HTTP {response.status_code}.", "multimodal_tokenize_failed", 502)
        try:
            count = response.json()["count"]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid count")
        except (ValueError, KeyError, TypeError):
            raise ResponsesRequestError(f"{stage}: invalid count response.", "multimodal_tokenize_invalid_response", 502) from None
        return count

    try:
        # The deadline includes queueing for the shared concurrency limit.
        return await asyncio.wait_for(perform(), timeout=timeout)
    except (asyncio.TimeoutError, httpx.TimeoutException):
        raise ResponsesRequestError(f"{stage}: timed out.", "multimodal_tokenize_timeout", 504) from None
    except httpx.RequestError:
        raise ResponsesRequestError(f"{stage}: connection failed.", "multimodal_tokenize_failed", 502) from None


async def count_request_tokens(req_data: dict, _tokenizer) -> tuple[int, int]:
    """Count the actual P/D contexts; image expansion belongs to the backend."""
    started = time.monotonic()
    images = image_input_count(req_data.get("messages", []))
    metadata = {"backend_route": "pd", "contains_images": bool(images), "image_count": images,
                "model": req_data.get("model")}
    if not images:
        counts = await asyncio.to_thread(count_prefill_and_decode_tokens, req_data, _tokenizer)
        metadata["count_source"] = "local_v41"
    else:
        p = proxy_state.select_tokenize_node("prefill")
        d = proxy_state.select_tokenize_node("decode")
        metadata.update(count_source="upstream_v41", prefill_count_node=p.url, decode_count_node=d.url)
        tasks = [
            asyncio.create_task(_tokenize_image_request(prepare_prefill_request_data(req_data), "prefill", p)),
            asyncio.create_task(_tokenize_image_request(req_data, "decode", d)),
        ]
        try:
            counts = tuple(await asyncio.gather(*tasks))
        except Exception as exc:
            metadata.update(count_error=backend_error_category(exc),
                            count_seconds=round(time.monotonic() - started, 6))
            _record_token_count_metadata(metadata)
            raise
        finally:
            # No orphan request/permit when the peer fails or the caller cancels.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    metadata.update(prefill_input_tokens=counts[0], decode_input_tokens=counts[1],
                    count_seconds=round(time.monotonic() - started, 6))
    _record_token_count_metadata(metadata)
    return counts


def _record_token_count_metadata(metadata: dict) -> None:
    trace = _active_wire_trace.get()
    audit = _active_request_audit.get()
    if trace:
        trace.write_json("token_count.json", metadata)
    if audit:
        audit.bind(**metadata)
        audit.write_json("token_count.json", metadata)
    logger.info("request_token_count %s", json.dumps(metadata, ensure_ascii=True))


def estimate_json_string_chars(value: Any) -> int:
    """Roughly estimate user-controlled text size before expensive tokenization."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, list):
        return sum(estimate_json_string_chars(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") in ("input_image", "image_url"):
            return 0
        return sum(estimate_json_string_chars(item) for item in value.values())
    return 0


class ResponsesRequestError(ValueError):
    def __init__(self, message: str, code: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class UnsupportedResponsesInputItemError(ResponsesRequestError):
    """A role-less Responses item that needs protocol-specific handling."""

    def __init__(self, input_item_index: int | None, item: dict[str, Any]):
        raw_type = item.get("type")
        raw_role = item.get("role")
        item_type = raw_type if isinstance(raw_type, str) else None
        role = raw_role if isinstance(raw_role, str) else None
        message = (
            f"Unsupported Responses input item type {item_type!r} without a message role."
            if item_type
            else "Unsupported Responses input item without a message role."
        )
        super().__init__(message, "unsupported_input_role")
        self.input_item_index = input_item_index
        self.item_type = item_type
        self.role = role
        self.item_keys = sorted(str(key)[:128] for key in item.keys())[:64]
        self.has_id = isinstance(item.get("id"), str)
        self.has_call_id = isinstance(item.get("call_id"), str)
        self.has_content = "content" in item
        self.has_encrypted_content = "encrypted_content" in item

    def diagnostic_payload(self) -> dict[str, Any]:
        return {
            "input_item_index": self.input_item_index,
            "item_type": self.item_type,
            "role": self.role,
            "item_keys": self.item_keys,
            "has_id": self.has_id,
            "has_call_id": self.has_call_id,
            "has_content": self.has_content,
            "has_encrypted_content": self.has_encrypted_content,
        }


class UnsupportedResponsesCompactionError(UnsupportedResponsesInputItemError):
    """A remote compaction request rejected before backend dispatch.

    The proxy does not implement Codex remote compaction v2.  Keeping this as
    an input-item error preserves the existing safe diagnostic/audit path while
    giving clients a stable, specific error code.
    """

    def __init__(self, input_item_index: int, item: dict[str, Any]):
        super().__init__(input_item_index, item)
        self.args = ("Remote compaction is not supported by this proxy.",)
        self.code = "unsupported_compaction"


def validate_tool_call_arguments(
    arguments: Any,
    *,
    request_id: str | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> str:
    """Return backend-compatible JSON arguments or reject malformed input."""
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            parsed = None
            valid = False
        else:
            valid = isinstance(parsed, (dict, list))
    else:
        parsed = arguments
        valid = isinstance(parsed, (dict, list))

    if not valid:
        digest = hashlib.sha256(
            arguments.encode("utf-8") if isinstance(arguments, str)
            else json.dumps(arguments, ensure_ascii=False, default=str).encode("utf-8")
        ).hexdigest()
        logger.warning(
            "invalid_function_arguments request_id=%s tool_call_id=%s "
            "tool_name=%s argument_bytes=%s argument_sha256=%s",
            request_id or "-",
            tool_call_id or "-",
            tool_name or "-",
            len(arguments.encode("utf-8")) if isinstance(arguments, str) else 0,
            digest,
        )
        raise ResponsesRequestError(
            "Function call arguments are not valid JSON.",
            "invalid_function_arguments",
        )
    if isinstance(arguments, str):
        return arguments
    return json.dumps(parsed, ensure_ascii=False)


def validate_chat_tool_calls(messages: Any, *, request_id: str | None = None) -> None:
    """Validate every assistant tool call before tokenization or forwarding."""
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if calls is None:
            continue
        if not isinstance(calls, list):
            raise ResponsesRequestError(
                "assistant tool_calls must be an array.", "invalid_function_arguments"
            )
        for call in calls:
            if not isinstance(call, dict):
                raise ResponsesRequestError(
                    "assistant tool_calls entries must be objects.",
                    "invalid_function_arguments",
                )
            function = call.get("function")
            if not isinstance(function, dict):
                raise ResponsesRequestError(
                    "assistant tool_calls function must be an object.",
                    "invalid_function_arguments",
                )
            call_id = call.get("id")
            name = function.get("name")
            if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
                raise ResponsesRequestError(
                    "Function calls require an id and name.",
                    "invalid_function_arguments",
                )
            function["arguments"] = validate_tool_call_arguments(
                function.get("arguments"),
                request_id=request_id,
                tool_call_id=call_id,
                tool_name=name,
            )


class WebToolError(RuntimeError):
    """A controlled failure from the proxy-owned web tool."""


class ResponsesBackendError(ResponsesRequestError):
    """A classified backend failure, safe to expose without its raw payload."""

    def __init__(self, message: str, code: str, status_code: int = 502):
        super().__init__(message, code, status_code)


def _backend_response_error(error: Any, status_code: int = 502) -> ResponsesBackendError:
    code = error.get("code") if isinstance(error, dict) else None
    if (not isinstance(code, str) or not code or code.lower() == "unknown"
            or len(code) > 128 or not all(c.isascii() and (c.isalnum() or c == "_") for c in code)):
        code = "decoder_stream_error"
    return ResponsesBackendError("Decoder request failed.", code, status_code)


def _web_round_failure(exc: Exception) -> tuple[str, str, int]:
    if isinstance(exc, ResponsesBackendError):
        return str(exc), exc.code, exc.status_code
    if isinstance(exc, ResponsesRequestError):
        return str(exc), exc.code, 502
    if isinstance(exc, WebToolError):
        return str(exc), "web_tool_error", 502
    if isinstance(exc, (httpx.HTTPError, OSError)):
        return "Decoder connection failed.", "decoder_stream_error", 502
    return "Proxy failed while processing the decoder response.", "proxy_internal_error", 500


class _SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._title_anchor_open = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            if self._current and self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
            self._current = {"title": "", "url": attributes.get("href", "")}
            self._capture = "title"
            self._title_anchor_open = True
        elif self._current and ("result__snippet" in classes or "result-snippet" in classes):
            self._capture = "snippet"

    def handle_data(self, data: str) -> None:
        if self._current is not None and self._capture:
            self._current[self._capture] = self._current.get(self._capture, "") + data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._current is not None and self._title_anchor_open:
            if self._current.get("title") and self._current.get("url"):
                self.results.append(self._current)
                if len(self.results) >= CODEX_WEB_MAX_RESULTS:
                    self._current = None
                    self._capture = None
                    return
            self._title_anchor_open = False
            self._capture = None


def _validate_external_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise WebToolError("Only absolute http:// and https:// URLs are supported.")
    if parsed.username or parsed.password:
        raise WebToolError(" URLs with embedded credentials are not supported.")
    host = parsed.hostname.rstrip(".")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in socket.getaddrinfo(host, parsed.port, type=socket.SOCK_STREAM)]
        except OSError as exc:
            raise WebToolError(f"Unable to resolve external host {host!r}.") from exc
    if not addresses:
        raise WebToolError(f"Unable to resolve external host {host!r}.")
    for address in addresses:
        if (address.is_private or address.is_loopback or address.is_link_local
                or address.is_multicast or address.is_unspecified or address.is_reserved):
            raise WebToolError("Access to private or local network addresses is blocked.")
    return url


def _external_http_client() -> httpx.AsyncClient:
    # This client is intentionally separate from the PD clients. Environment
    # proxy settings must never affect prefill, decode, or Redis traffic.
    proxy = CODEX_HTTPS_PROXY or CODEX_HTTP_PROXY
    return httpx.AsyncClient(
        proxy=proxy,
        trust_env=False,
        follow_redirects=False,
        max_redirects=CODEX_WEB_MAX_REDIRECTS,
        timeout=httpx.Timeout(CODEX_WEB_FETCH_TIMEOUT_SECONDS),
        headers={"User-Agent": "DSv4-Flash-Responses-WebTool/1.0"},
    )


async def _read_limited_response(response: httpx.Response) -> bytes:
    chunks = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > CODEX_WEB_MAX_BODY_BYTES:
            raise WebToolError("External response exceeded the configured size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


async def fetch_external_url(url: str) -> dict[str, Any]:
    current = _validate_external_url(url)
    async with _external_http_client() as client:
        for _ in range(CODEX_WEB_MAX_REDIRECTS + 1):
            response = await client.get(current)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise WebToolError("External server returned a redirect without a location.")
                current = _validate_external_url(str(httpx.URL(current).join(location)))
                continue
            body = await _read_limited_response(response)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type and content_type not in {"text/plain", "text/html", "application/json", "application/xml"}:
                raise WebToolError(f"Unsupported external content type {content_type!r}.")
            return {
                "url": current,
                "status_code": response.status_code,
                "content_type": content_type,
                "body": body.decode("utf-8", errors="replace"),
            }
    raise WebToolError("External URL exceeded the redirect limit.")


async def execute_web_search(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {"query": arguments}
    if not isinstance(arguments, dict):
        raise WebToolError("web_search arguments must be an object.")
    direct_url = arguments.get("url") or arguments.get("open_page")
    if isinstance(direct_url, dict):
        direct_url = direct_url.get("url")
    if isinstance(direct_url, str) and direct_url:
        page = await fetch_external_url(direct_url)
        page["body"] = page["body"][:CODEX_WEB_MAX_BODY_BYTES]
        return {"type": "open_page", "page": page}
    query = arguments.get("query") or arguments.get("q")
    if not isinstance(query, str) or not query.strip():
        raise WebToolError("web_search requires a non-empty query.")
    if not CODEX_WEB_SEARCH_URL:
        search_url = _validate_external_url("https://html.duckduckgo.com/html/")
        async with _external_http_client() as client:
            response = await client.get(search_url, params={"q": query})
            body = await _read_limited_response(response)
        if response.status_code >= 400:
            raise WebToolError(f"Default web search returned HTTP {response.status_code}.")
        parser = _SearchResultParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        return {"query": query, "results": parser.results[:CODEX_WEB_MAX_RESULTS]}
    search_url = _validate_external_url(CODEX_WEB_SEARCH_URL)
    headers = {"Accept": "application/json"}
    if CODEX_WEB_SEARCH_API_KEY:
        headers["Authorization"] = f"Bearer {CODEX_WEB_SEARCH_API_KEY}"
    payload = {"query": query, "max_results": CODEX_WEB_MAX_RESULTS}
    async with _external_http_client() as client:
        response = await client.post(search_url, json=payload, headers=headers)
        body = await _read_limited_response(response)
    if response.status_code >= 400:
        raise WebToolError(f"Configured web search returned HTTP {response.status_code}.")
    try:
        result = json.loads(body)
    except json.JSONDecodeError as exc:
        raise WebToolError("Configured web search did not return JSON.") from exc
    if not isinstance(result, dict):
        raise WebToolError("Configured web search returned an invalid result.")
    return {"query": query, "results": result.get("results", result)}


def _chat_tool_calls(chat_response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = chat_response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return []
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    return [call for call in calls if isinstance(call, dict)]


def _assistant_message_for_tool_calls(chat_response: dict[str, Any]) -> dict[str, Any] | None:
    choices = chat_response.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    result = {
        "role": "assistant",
        "content": message.get("content") or "",
        "tool_calls": copy.deepcopy(calls),
    }
    reasoning = _extract_reasoning_text(message)
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


async def _run_proxy_web_tool_rounds(
    initial_response: dict[str, Any],
    chat_request: dict[str, Any],
    request: Request,
    state: "ResponseRequestState",
    trace: "WireTrace",
    backend_runner: "ResponsesBackendRunner | None" = None,
) -> dict[str, Any]:
    """Execute proxy-owned web tools and continue through the selected Chat backend."""
    original_stream = chat_request.get("stream", False)
    chat_request["stream"] = False
    response = initial_response
    try:
        for round_number in range(CODEX_MAX_TOOL_ROUNDS):
            if not isinstance(response, dict):
                raise ResponsesBackendError("Invalid decoder response.", "invalid_decoder_response")
            if response.get("error"):
                raise _backend_response_error(response["error"])
            choices = response.get("choices")
            if (not isinstance(choices, list) or not choices or not isinstance(choices[0], dict)
                    or not isinstance(choices[0].get("message"), dict)):
                raise ResponsesBackendError("Invalid decoder response.", "invalid_decoder_response")
            raw_calls = choices[0]["message"].get("tool_calls", [])
            if raw_calls is None:
                raw_calls = []
            if (not isinstance(raw_calls, list)
                    or any(not isinstance(call, dict) for call in raw_calls)):
                raise ResponsesBackendError("Invalid decoder tool calls.", "incomplete_function_call")
            calls = _chat_tool_calls(response)
            if calls:
                # Validate every call before executing any proxy-owned tool. Rebuild
                # canonical backend names so existing guarded aliases work here too.
                for call in calls:
                    function = call.get("function")
                    if (not isinstance(call.get("id"), str) or not call["id"]
                            or not isinstance(function, dict)
                            or not isinstance(function.get("name"), str) or not function["name"]):
                        raise ResponsesBackendError("Incomplete decoder tool call.", "incomplete_function_call")
                if len({call["id"] for call in calls}) != len(calls):
                    raise ResponsesBackendError("Duplicate decoder tool call ID.", "incomplete_function_call")
                canonical_calls = []
                for call in calls:
                    function = call["function"]
                    spec, arguments, _ = _normalize_decoder_tool_call(
                        function["name"], function.get("arguments", "{}"), call["id"], state
                    )
                    arguments = validate_tool_call_arguments(
                        arguments, tool_call_id=call["id"], tool_name=function["name"]
                    )
                    if spec.get("source_type") == "tool_search":
                        arguments, _ = _tool_search_arguments_object(arguments, tool_call_id=call["id"])
                    canonical = copy.deepcopy(call)
                    canonical["function"] = {
                        "name": spec.get("backend_name", function["name"]), "arguments": arguments,
                    }
                    canonical_calls.append(canonical)
                response = copy.deepcopy(response)
                response["choices"][0]["message"]["tool_calls"] = canonical_calls
                calls = canonical_calls
            web_calls = []
            client_calls = []
            for call in calls:
                function = call.get("function") or {}
                spec = state.tool_registry.get(function.get("name", ""), {})
                if spec.get("source_type") == "web_search" and spec.get("execution") == "proxy":
                    web_calls.append(call)
                else:
                    client_calls.append(call)
            if not web_calls:
                return response
            assistant = _assistant_message_for_tool_calls(response)
            if assistant is None:
                return response
            if client_calls:
                # Split the batch into a completed server-side exchange followed
                # by pending client calls. Store/return the latter exactly once.
                assistant = {"role": "assistant", "content": "", "tool_calls": web_calls}
            chat_request["messages"].append(assistant)
            for call in web_calls:
                function = call.get("function") or {}
                try:
                    result = await execute_web_search(function.get("arguments", "{}"))
                except WebToolError as exc:
                    result = {"error": {"code": "web_tool_error", "message": str(exc)}}
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.get("id", f"call_{uuid.uuid4().hex}"),
                    "content": json.dumps(result, ensure_ascii=False),
                }
                chat_request["messages"].append(tool_message)
                if trace:
                    trace.write("web_search.results.jsonl", {
                        "round": round_number,
                        "tool_call_id": tool_message["tool_call_id"],
                        "result": result,
                    })
            if trace:
                trace.write_json("request.normalized.json", chat_request)
            if client_calls:
                response["choices"][0]["message"]["tool_calls"] = client_calls
                return response
            runner = backend_runner or _run_responses_backend
            result_response = await runner(
                request,
                chat_request,
                state,
                trace,
                json.dumps(chat_request, ensure_ascii=False).encode(),
            )
            if not isinstance(result_response, JSONResponse):
                raise ResponsesBackendError("Invalid decoder continuation response.", "invalid_decoder_response")
            try:
                body = json.loads(result_response.body)
            except (TypeError, ValueError) as exc:
                raise ResponsesBackendError("Invalid decoder response JSON.", "invalid_decoder_response") from exc
            if result_response.status_code >= 400:
                raise _backend_response_error(
                    body.get("error", body) if isinstance(body, dict) else None,
                    result_response.status_code,
                )
            response = body
        raise WebToolError("Maximum web tool rounds exceeded.")
    finally:
        chat_request["stream"] = original_stream


async def _collect_chat_stream_response(body_iterator) -> dict[str, Any]:
    """Materialize a Chat Completions stream for an internal web-tool round."""
    buffer = ""
    content = ""
    reasoning = ""
    tool_calls: dict[int, dict[str, Any]] = {}
    usage = {}
    finish_reason = None
    saw_done = False

    async def consume_event(event: str) -> None:
        nonlocal content, reasoning, usage, finish_reason, saw_done
        data = "\n".join(
            line.lstrip()[5:].strip()
            for line in event.splitlines()
            if line.lstrip().startswith("data:")
        )
        if not data:
            return
        if data == "[DONE]":
            saw_done = True
            return
        payload = json.loads(data)
        if payload.get("error"):
            raise _backend_response_error(payload["error"])
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            content += delta["content"]
        reasoning_delta = _extract_reasoning_text(delta)
        if reasoning_delta:
            reasoning += reasoning_delta
        for call_delta in delta.get("tool_calls") or []:
            index = int(call_delta.get("index", 0))
            call = tool_calls.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            if call_delta.get("id"):
                call["id"] = call_delta["id"]
            function = call_delta.get("function") or {}
            if function.get("name"):
                call["function"]["name"] = function["name"]
            if function.get("arguments"):
                call["function"]["arguments"] += function["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        if payload.get("usage"):
            usage = payload["usage"]

    try:
        async for raw_chunk in body_iterator:
            buffer += raw_chunk.decode("utf-8", errors="replace") if isinstance(raw_chunk, bytes) else str(raw_chunk)
            buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                await consume_event(event)
        if buffer.strip():
            await consume_event(buffer)
    except ResponsesBackendError:
        raise
    except Exception as exc:
        raise ResponsesBackendError("Invalid or interrupted decoder stream.", "decoder_stream_error") from exc
    finally:
        with suppress(Exception):
            await _aclose_async_iterator(body_iterator)
    if not saw_done or not isinstance(finish_reason, str) or not finish_reason:
        raise ResponsesBackendError("Decoder stream ended without terminal signals.", "decoder_stream_error")
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


@dataclass
class ResponseRequestState:
    response_id: str
    model: str
    created_at: int
    store: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_response_id: str | None = None
    conversation_id: str | None = None
    reasoning_effort: str | None = None
    reasoning_settings: dict[str, Any] = field(default_factory=dict)
    # Maps backend-safe names back to the tool names exposed by Responses.
    tool_registry: dict[str, dict[str, Any]] = field(default_factory=dict)
    backend_route: str = "pd"
    contains_images: bool = False
    backend_model: str | None = None
    pipeline_leak_state: Any = None
    exec_normalization_cache: dict = field(default_factory=dict, repr=False)


@dataclass
class FailureCaptureSource:
    """Preserves the original API body when Responses creates a synthetic request."""

    client_request_body: bytes
    api: str
    response_id: str | None = None


@dataclass
class FailureCaptureContext:
    client_request_id: str
    response_id: str | None
    api: str
    request_kind: str
    client_request_body: bytes
    normalized_backend_request: dict[str, Any]
    received_at: str


class RequestAuditTrace:
    """Fail-open per-request audit index and payload archive.

    Payload starts in ``in_progress`` and is atomically moved to its terminal
    category.  This makes a directory's category trustworthy even if the
    process is interrupted while a stream is active.
    """

    _retention_lock = threading.Lock()
    _retention_checked_dates: set[str] = set()

    def __init__(self, request: Request, api_kind: str):
        self.trace_id = f"req_{uuid.uuid4().hex}"
        self.started_at = time.time()
        self.date = datetime.now(_REQUEST_AUDIT_TZ).date().isoformat()
        self.api_kind = api_kind
        self.endpoint = request.url.path
        self.method = request.method
        self.client_addr = (
            f"{request.client.host}:{request.client.port}" if request.client else "unknown"
        )
        self.fields: dict[str, Any] = {
            "trace_id": self.trace_id, "listener_port": REQUEST_AUDIT_LISTENER_PORT,
            "service_identity": REQUEST_AUDIT_SERVICE_ID,
            "deployment_identity": f"pid:{os.getpid()}", "endpoint": self.endpoint,
            "api_kind": api_kind, "method": self.method, "client_addr": self.client_addr,
            "stream": None, "response_id": None, "client_request_id": None,
            "upstream_request_id": None, "session_id": None, "thread_id": None,
            "payload_state": "pending", "audit_detail_dropped": False,
        }
        self._finished = False
        self._write_failed = False
        self._payload_enabled = REQUEST_AUDIT_TRACE and REQUEST_AUDIT_PAYLOAD_MODE == "all"
        self.audit_dir = os.path.join(REQUEST_AUDIT_TRACE_DIR, "audit", self.date, REQUEST_AUDIT_LISTENER_PORT)
        self.staging_dir = os.path.join(
            REQUEST_AUDIT_TRACE_DIR, "payload", self.date, REQUEST_AUDIT_LISTENER_PORT,
            "in_progress", self.trace_id,
        )
        self.directory = self.staging_dir
        if REQUEST_AUDIT_TRACE:
            self._run_retention_once()
            self._safe_makedirs(self.audit_dir)
            if self._payload_enabled:
                self._safe_makedirs(self.directory)
        self.event("received")

    def _safe_makedirs(self, path: str) -> bool:
        try:
            os.makedirs(path, mode=0o700, exist_ok=True)
            return True
        except OSError:
            self._trace_failure("mkdir", path)
            return False

    def _trace_failure(self, operation: str, path: str) -> None:
        if not self._write_failed:
            self._write_failed = True
            logger.warning("trace_write_failure trace_id=%s operation=%s path=%s", self.trace_id, operation, path)

    def _run_retention_once(self) -> None:
        with self._retention_lock:
            if self.date in self._retention_checked_dates:
                return
            self._retention_checked_dates.add(self.date)
            try:
                cutoff = time.strftime(
                    "%Y-%m-%d", time.localtime(self.started_at - REQUEST_AUDIT_RETENTION_DAYS * 86400)
                )
                for branch in ("audit", "payload"):
                    root = os.path.join(REQUEST_AUDIT_TRACE_DIR, branch)
                    if not os.path.isdir(root):
                        continue
                    for child in os.listdir(root):
                        if len(child) == 10 and child[4] == "-" and child[7] == "-" and child < cutoff:
                            shutil.rmtree(os.path.join(root, child), ignore_errors=True)
                if REQUEST_AUDIT_MAX_BYTES:
                    def trace_size() -> int:
                        return sum(os.path.getsize(os.path.join(base, name))
                                   for base, _, names in os.walk(REQUEST_AUDIT_TRACE_DIR) for name in names)
                    total = trace_size()
                    # Capacity protection removes whole oldest dates, keeping a
                    # date's audit index and payload evidence consistent.
                    dates = set()
                    for branch in ("audit", "payload"):
                        root = os.path.join(REQUEST_AUDIT_TRACE_DIR, branch)
                        if os.path.isdir(root):
                            dates.update(child for child in os.listdir(root)
                                         if len(child) == 10 and child[4] == "-" and child[7] == "-")
                    for child in sorted(dates):
                        if total < REQUEST_AUDIT_MAX_BYTES or child >= self.date:
                            break
                        for branch in ("audit", "payload"):
                            shutil.rmtree(os.path.join(REQUEST_AUDIT_TRACE_DIR, branch, child), ignore_errors=True)
                        total = trace_size()
                    if total >= REQUEST_AUDIT_MAX_BYTES:
                        self._payload_enabled = False
                        logger.error("trace_storage_degraded total_bytes=%d limit_bytes=%d", total, REQUEST_AUDIT_MAX_BYTES)
            except Exception:
                self._trace_failure("retention", REQUEST_AUDIT_TRACE_DIR)

    @staticmethod
    def _safe_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

    def event(self, event: str, **fields: Any) -> None:
        if not REQUEST_AUDIT_TRACE:
            return
        record = {**self.fields, **fields, "event": event,
                  "timestamp": datetime.now(_REQUEST_AUDIT_TZ).isoformat(timespec="milliseconds")}
        path = os.path.join(self.audit_dir, "events.jsonl")
        try:
            encoded = (self._safe_json(record) + "\n").encode("utf-8")
            self._safe_makedirs(self.audit_dir)
            with open(path, "ab", buffering=0) as output:
                fcntl.flock(output.fileno(), fcntl.LOCK_EX)
                try:
                    output.write(encoded)
                finally:
                    fcntl.flock(output.fileno(), fcntl.LOCK_UN)
            os.chmod(path, 0o600)
        except Exception:
            self._trace_failure("event", path)

    def write_json(self, name: str, payload: Any, *, append: bool = False) -> None:
        if not self._payload_enabled:
            return
        if name not in {"request.raw.json", "request.parsed.json"}:
            payload = _redact_multimodal(payload)
        path = os.path.join(self.directory, name)
        try:
            self._safe_makedirs(os.path.dirname(path))
            if append:
                with open(path, "a", encoding="utf-8") as output:
                    output.write(self._safe_json(payload) + "\n")
            else:
                tmp = f"{path}.tmp.{uuid.uuid4().hex}"
                with open(tmp, "w", encoding="utf-8") as output:
                    json.dump(payload, output, ensure_ascii=False, indent=2, default=str)
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
        except Exception:
            self._trace_failure("write_json", path)

    def write_bytes(self, name: str, payload: bytes) -> None:
        if not self._payload_enabled:
            return
        path = os.path.join(self.directory, name)
        try:
            self._safe_makedirs(os.path.dirname(path))
            tmp = f"{path}.tmp.{uuid.uuid4().hex}"
            with open(tmp, "wb") as output:
                output.write(payload)
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except Exception:
            self._trace_failure("write_bytes", path)

    def capture_body(self, raw_body: bytes) -> None:
        digest = hashlib.sha256(raw_body).hexdigest()
        self.fields.update(body_bytes=len(raw_body), body_sha256=digest)
        self.write_bytes("request.raw.bin", raw_body)
        self.write_bytes("request.raw.sha256", (digest + "\n").encode())
        self.event("body_read")

    def capture_parsed(self, payload: Any, *, normalized: bool = False) -> None:
        self.write_json("request.normalized.json" if normalized else "request.parsed.json", payload)
        self.event("normalized" if normalized else "json_parsed")

    def bind(self, **fields: Any) -> None:
        for key, value in fields.items():
            if value is not None:
                self.fields[key] = value

    def _extract_correlations(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        for target, candidates in {
            "session_id": ("session_id", "session", "conversation_id"),
            "thread_id": ("thread_id", "thread"),
            "client_request_id": ("request_id", "client_request_id"),
        }.items():
            for name in candidates:
                value = metadata.get(name, payload.get(name))
                if isinstance(value, (str, int)):
                    self.fields[target] = str(value)
                    break
        self.fields["stream"] = bool(payload.get("stream", False))

    def capture_request_json(self, payload: Any, *, normalized: bool = False) -> None:
        self._extract_correlations(payload)
        self.capture_parsed(payload, normalized=normalized)

    def finish(self, final_state: str, *, http_status: int | None = None,
               error_code: str | None = None, exception: Exception | None = None,
               disconnect_phase: str | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        terminal_at = time.time()
        self.fields.update(final_state=final_state, http_status=http_status,
                           error_code=error_code, terminal_at=terminal_at,
                           elapsed_ms=round((terminal_at - self.started_at) * 1000, 3))
        if exception is not None:
            self.fields["exception_class"] = type(exception).__name__
        if disconnect_phase:
            self.fields["disconnect_phase"] = disconnect_phase
        target = os.path.join(REQUEST_AUDIT_TRACE_DIR, "payload", self.date,
                              REQUEST_AUDIT_LISTENER_PORT, final_state, self.trace_id)
        if self._payload_enabled and os.path.isdir(self.staging_dir):
            try:
                self._safe_makedirs(os.path.dirname(target))
                os.replace(self.staging_dir, target)
                self.directory = target
                self.fields["payload_state"] = "archived"
            except Exception:
                self._trace_failure("archive", target)
                self.fields["payload_state"] = "archive_failed"
        elif not self._payload_enabled:
            self.fields["payload_state"] = "disabled"
        self.write_json("manifest.json", {
            "trace_id": self.trace_id, "endpoint": self.endpoint,
            "final_state": final_state, "response_id": self.fields.get("response_id"),
            "terminal_at": terminal_at, "payload_files": self._payload_file_names(),
        })
        self.write_json("terminal.json", {**self.fields, "trace_write_failed": self._write_failed})
        self.event("terminal", **self.fields)

    def _payload_file_names(self) -> list[str]:
        try:
            return sorted(
                os.path.relpath(os.path.join(base, name), self.directory)
                for base, _, names in os.walk(self.directory) for name in names
                if ".tmp." not in name
            )
        except OSError:
            return []

    async def wrap_stream(self, iterator: Any):
        try:
            async for item in iterator:
                if (RESPONSES_SSE_TRACE and self._payload_enabled
                        and self.api_kind == "responses"):
                    # Capture the final Responses bytes at the ASGI handoff,
                    # not the upstream Chat SSE. This is not a client receipt.
                    path = os.path.join(self.directory, "response.client.sse")
                    try:
                        payload = item.encode("utf-8") if isinstance(item, str) else bytes(item)
                        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "ab") as output:
                            output.write(payload)
                    except Exception:
                        self._trace_failure("write_client_sse", path)
                yield item
        except asyncio.CancelledError as exc:
            self.finish("disconnected", exception=exc, disconnect_phase="streaming")
            raise
        except Exception as exc:
            self.finish("failed", exception=exc)
            raise
        else:
            self.finish("accepted", http_status=200)
        finally:
            # ``aclose()`` may inject GeneratorExit rather than an asyncio
            # cancellation; it still represents a client-side early close.
            if not self._finished:
                self.finish("disconnected", disconnect_phase="streaming")


class WireTrace:
    def __init__(self, response_id: str):
        self.response_id = response_id
        self.audit = _active_request_audit.get()
        if self.audit is not None:
            self.audit.bind(response_id=response_id)
            self.directory = self.audit.directory
        else:
            self.directory = os.path.join(RESPONSES_WIRE_TRACE_DIR, response_id)
        if RESPONSES_WIRE_TRACE and self.audit is None:
            os.makedirs(self.directory, exist_ok=True)

    def write(self, name: str, payload: Any) -> None:
        payload = _redact_multimodal(payload)
        if self.audit is not None:
            self.audit.write_json(name, payload, append=True)
            return
        if not RESPONSES_WIRE_TRACE:
            return
        path = os.path.join(self.directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, default=str)
            output.write("\n")

    def write_json(self, name: str, payload: Any) -> None:
        if name not in {"request.raw.json", "request.parsed.json"}:
            payload = _redact_multimodal(payload)
        if self.audit is not None:
            self.audit.write_json(name, payload)
            return
        if not RESPONSES_WIRE_TRACE:
            return
        path = os.path.join(self.directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, default=str)

    def write_bytes(self, name: str, payload: bytes) -> None:
        if self.audit is not None:
            self.audit.write_bytes(name, payload)
            return
        if not RESPONSES_WIRE_TRACE:
            return
        path = os.path.join(self.directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as output:
            output.write(payload)


def record_rejected_responses_input(
    error: UnsupportedResponsesInputItemError,
    raw_body: bytes,
) -> str:
    """Persist only safe structure metadata for an unsupported input item."""
    diagnostic_id = f"rejected_{uuid.uuid4().hex}"
    payload = {
        "event": "responses_input_rejected",
        "diagnostic_id": diagnostic_id,
        "observed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": "/v1/responses",
        "body_bytes": len(raw_body),
        "body_sha256": hashlib.sha256(raw_body).hexdigest(),
        **error.diagnostic_payload(),
    }
    audit = _active_request_audit.get()
    if audit is not None:
        audit.bind(diagnostic_id=diagnostic_id)
        audit.write_json("input.rejected.json", payload)
        audit.event("conversion_failed", error_code=error.code, diagnostic_id=diagnostic_id)
    else:
        WireTrace(diagnostic_id).write_json("input.rejected.json", payload)
    logger.warning(
        "responses_input_rejected diagnostic_id=%s input_item_index=%s item_type=%r "
        "role=%r item_keys=%s body_bytes=%d",
        diagnostic_id,
        error.input_item_index,
        error.item_type,
        error.role,
        error.item_keys,
        len(raw_body),
    )
    return diagnostic_id


def record_decoder_attempt_trace(trace: WireTrace | None, event: str, **fields: Any) -> None:
    """Append only correlation and transport metadata for a Decoder attempt."""
    if trace is None:
        return
    trace.write(
        "decode.attempts.jsonl",
        {
            "event": event,
            "timestamp": _failure_capture_timestamp(),
            **fields,
        },
    )


def _failure_capture_timestamp(timestamp: float | None = None) -> str:
    return time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(time.time() if timestamp is None else timestamp),
    )


def _ensure_failure_capture_dir(path: str) -> None:
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


def _atomic_write_bytes(path: str, payload: bytes, mode: int = 0o600) -> None:
    temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(temporary_path, "wb") as output:
        output.write(payload)
    os.chmod(temporary_path, mode)
    os.replace(temporary_path, path)


def _atomic_write_json(path: str, payload: Any) -> None:
    _atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
    )


def _failure_capture_headers(headers: dict[str, str]) -> dict[str, str]:
    allowed = {
        "content-type",
        "content-length",
        "connection",
        "x-request-id",
        "x-response-id",
    }
    return {
        key: value
        for key, value in headers.items()
        if key.lower() in allowed
    }


def _failure_kind_for_http_status(status_code: int, response_body: bytes) -> str:
    text = response_body.decode("utf-8", errors="replace")
    if status_code == 400 and "Unterminated string" in text:
        return "prefill_invalid_json"
    return f"prefill_http_{status_code}"


def _failure_kind_for_transport_error(exc: httpx.RequestError) -> str:
    return f"prefill_transport_{type(exc).__name__.lower()}"


def _failure_capture_category(
    failure_kind: str,
    response: httpx.Response | None = None,
) -> str:
    if response is not None:
        return str(response.status_code)
    if failure_kind.startswith("prefill_transport_"):
        return "transport"
    return "other"


def _write_failure_capture(
    *,
    request_id: str,
    prefiller: ServerState,
    endpoint: str,
    attempt: int,
    headers: dict[str, str],
    body: bytes,
    body_hash: str,
    prefill_request: dict[str, Any],
    request_started: float,
    failure_kind: str,
    exc: Exception,
    response: httpx.Response | None = None,
) -> str | None:
    """Persist the exact failed Prefill request without affecting proxy error handling."""
    if not PREFILL_FAILURE_CAPTURE:
        return None

    context = _active_failure_capture_context.get()
    try:
        _ensure_failure_capture_dir(PREFILL_FAILURE_CAPTURE_DIR)
        category = _failure_capture_category(failure_kind, response)
        category_dir = os.path.join(PREFILL_FAILURE_CAPTURE_DIR, category)
        _ensure_failure_capture_dir(category_dir)
        request_dir = os.path.join(category_dir, request_id)
        _ensure_failure_capture_dir(request_dir)

        if context is not None:
            client_meta_path = os.path.join(request_dir, "client.request.meta.json")
            if not os.path.exists(client_meta_path):
                _atomic_write_bytes(
                    os.path.join(request_dir, "client.request.body"),
                    context.client_request_body,
                )
                _atomic_write_json(client_meta_path, {
                    "client_request_id": context.client_request_id,
                    "response_id": context.response_id,
                    "endpoint": context.api,
                    "request_kind": context.request_kind,
                    "received_at": context.received_at,
                    "body_bytes": len(context.client_request_body),
                    "body_sha256": hashlib.sha256(context.client_request_body).hexdigest(),
                })
                _atomic_write_json(
                    os.path.join(request_dir, "backend.request.json"),
                    context.normalized_backend_request,
                )

        attempt_name = f"attempt-{attempt:03d}"
        attempt_dir = os.path.join(request_dir, attempt_name)
        temporary_dir = os.path.join(
            request_dir, f".tmp-{attempt_name}-{uuid.uuid4().hex}"
        )
        _ensure_failure_capture_dir(temporary_dir)
        response_body = response.content if response is not None else b""
        response_headers = dict(response.headers) if response is not None else {}
        finished_at = _failure_capture_timestamp()
        duration_seconds = round(time.time() - request_started, 6)
        request_meta = {
            "request_id": request_id,
            "client_request_id": context.client_request_id if context else None,
            "response_id": context.response_id if context else None,
            "node": str(prefiller),
            "endpoint": endpoint,
            "attempt": attempt,
            "connection_mode": PREFILL_CONNECTION_MODE,
            "headers": _failure_capture_headers(headers),
            "content_length": int(headers["Content-Length"]),
            "serialized_body_bytes": len(body),
            "body_sha256": body_hash,
            "json_parse_verified_before_send": True,
            "started_at": _failure_capture_timestamp(request_started),
            "finished_at": finished_at,
            "duration_seconds": duration_seconds,
        }
        response_meta = {
            "status_code": response.status_code if response is not None else None,
            "headers": _failure_capture_headers(response_headers),
            "body_bytes": len(response_body),
            "body_sha256": hashlib.sha256(response_body).hexdigest() if response is not None else None,
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "failure_kind": failure_kind,
        }
        _atomic_write_bytes(os.path.join(temporary_dir, "prefill.request.body"), body)
        saved_body = os.path.join(temporary_dir, "prefill.request.body")
        with open(saved_body, "rb") as input_file:
            persisted_body = input_file.read()
        if len(persisted_body) != len(body) or hashlib.sha256(persisted_body).hexdigest() != body_hash:
            raise RuntimeError("Failure capture body verification failed.")
        _atomic_write_json(os.path.join(temporary_dir, "prefill.request.json"), prefill_request)
        _atomic_write_json(os.path.join(temporary_dir, "prefill.request.meta.json"), request_meta)
        _atomic_write_bytes(os.path.join(temporary_dir, "prefill.response.body"), response_body)
        _atomic_write_json(os.path.join(temporary_dir, "prefill.response.meta.json"), response_meta)
        replay_script = f'''#!/bin/sh
set -eu
cd "$(dirname "$0")"
curl --http1.1 --max-time 120 \\
  --data-binary @prefill.request.body \\
  -H 'Content-Type: application/json' \\
  -H 'Content-Length: {len(body)}' \\
  -H 'Connection: {headers.get("Connection", "keep-alive")}' \\
  -H 'X-Request-Id: {request_id}' \\
  -H "Authorization: Bearer ${{OPENAI_API_KEY:?OPENAI_API_KEY is required}}" \\
  '{prefiller.url}{endpoint}'
'''
        replay_path = os.path.join(temporary_dir, "replay.sh")
        _atomic_write_bytes(replay_path, replay_script.encode("utf-8"), mode=0o700)
        if os.path.exists(attempt_dir):
            raise RuntimeError(f"Failure capture attempt already exists: {attempt_name}")
        os.replace(temporary_dir, attempt_dir)

        manifest_path = os.path.join(request_dir, "manifest.json")
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, encoding="utf-8") as input_file:
                manifest = json.load(input_file)
        attempts = manifest.setdefault("attempts", [])
        if attempt_name not in attempts:
            attempts.append(attempt_name)
        manifest.update({
            "request_id": request_id,
            "client_request_id": context.client_request_id if context else None,
            "response_id": context.response_id if context else None,
            "api": context.api if context else None,
            "request_kind": context.request_kind if context else None,
            "failure_kind": failure_kind,
            "failure_category": category,
            "node": str(prefiller),
            "endpoint": f"/v1{endpoint}",
            "created_at": manifest.get("created_at", _failure_capture_timestamp()),
            "updated_at": finished_at,
        })
        _atomic_write_json(manifest_path, manifest)
        stats_collector.record_failure_capture_saved(failure_kind)
        logger.error(
            "failure_capture_saved request_id=%s attempt=%s failure_kind=%s node=%s "
            "body_bytes=%s body_sha256=%s capture_dir=%s",
            request_id,
            attempt,
            failure_kind,
            prefiller,
            len(body),
            body_hash,
            attempt_dir,
        )
        return attempt_dir
    except Exception as capture_exc:
        stats_collector.record_failure_capture_write_failed()
        logger.error(
            "failure_capture_write_failed request_id=%s attempt=%s failure_kind=%s error=%s",
            request_id,
            attempt,
            failure_kind,
            capture_exc,
        )
        return None


def initialize_failure_capture_storage() -> None:
    if PREFILL_FAILURE_CAPTURE:
        _ensure_failure_capture_dir(PREFILL_FAILURE_CAPTURE_DIR)


class ResponseStateStore:
    @property
    def enabled(self) -> bool:
        return bool(RESPONSES_REDIS_URL)

    def _connection_info(self) -> tuple[str, int, int, str | None, str | None]:
        parsed = urlparse(RESPONSES_REDIS_URL)
        if parsed.scheme != "redis":
            raise RuntimeError("RESPONSES_REDIS_URL must use redis://.")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 6379
        database = int(parsed.path.strip("/") or "0")
        password = unquote(parsed.password) if parsed.password else None
        username = unquote(parsed.username) if parsed.username else None
        return host, port, database, username, password

    @staticmethod
    def _encode_command(*parts: Any) -> bytes:
        encoded = [str(part).encode("utf-8") for part in parts]
        return b"*" + str(len(encoded)).encode() + b"\r\n" + b"".join(
            b"$" + str(len(part)).encode() + b"\r\n" + part + b"\r\n"
            for part in encoded
        )

    @staticmethod
    async def _read_response(reader: asyncio.StreamReader) -> Any:
        prefix = await reader.readexactly(1)
        line = await reader.readline()
        if prefix == b"+":
            return line[:-2].decode("utf-8", errors="replace")
        if prefix == b"-":
            raise RuntimeError(line[:-2].decode("utf-8", errors="replace"))
        if prefix == b":":
            return int(line[:-2])
        if prefix == b"$":
            length = int(line[:-2])
            if length < 0:
                return None
            data = await reader.readexactly(length)
            await reader.readexactly(2)
            return data.decode("utf-8", errors="replace")
        raise RuntimeError(f"Unsupported Redis response prefix: {prefix!r}")

    async def _command(self, *parts: Any) -> Any:
        host, port, database, username, password = self._connection_info()
        reader, writer = await asyncio.open_connection(host, port)
        try:
            if password:
                if username:
                    await self._send_command(reader, writer, "AUTH", username, password)
                else:
                    await self._send_command(reader, writer, "AUTH", password)
            if database:
                await self._send_command(reader, writer, "SELECT", database)
            writer.write(self._encode_command(*parts))
            await writer.drain()
            return await self._read_response(reader)
        finally:
            writer.close()
            await writer.wait_closed()

    async def _send_command(self, reader, writer, *parts: Any) -> Any:
        writer.write(self._encode_command(*parts))
        await writer.drain()
        return await self._read_response(reader)

    async def get(self, response_id: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        value = await self._command("GET", f"{RESPONSES_KEY_PREFIX}{response_id}")
        return json.loads(value) if value else None

    async def put(self, response: dict[str, Any]) -> None:
        if not self.enabled:
            raise RuntimeError("RESPONSES_REDIS_URL is not configured.")
        key = f"{RESPONSES_KEY_PREFIX}{response['id']}"
        value = json.dumps(response, ensure_ascii=False)
        await self._command("SETEX", key, RESPONSES_STATE_TTL_SECONDS, value)

    async def get_conversation(self, conversation_id: str) -> str | None:
        if not self.enabled:
            return None
        value = await self._command("GET", f"{RESPONSES_KEY_PREFIX}conversation:{conversation_id}")
        return value if value else None

    async def put_conversation(self, conversation_id: str, response_id: str) -> None:
        if not self.enabled:
            raise RuntimeError("RESPONSES_REDIS_URL is not configured.")
        await self._command(
            "SETEX",
            f"{RESPONSES_KEY_PREFIX}conversation:{conversation_id}",
            RESPONSES_STATE_TTL_SECONDS,
            response_id,
        )


response_state_store = ResponseStateStore()


def responses_error(message: str, code: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": "invalid_request_error" if status_code < 500 else "server_error",
                "code": code,
                "message": message,
            }
        },
    )


def _response_content_to_text(
    content: Any,
    *,
    allow_output_text: bool = False,
    allow_encrypted_content: bool = False,
    allow_multimodal: bool = True,
) -> Any:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ResponsesRequestError("Message content must be a string or an array.", "invalid_input_content")
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise ResponsesRequestError("Invalid input content part.", "invalid_input_content")
        kind = part.get("type")
        if not isinstance(kind, str):
            raise ResponsesRequestError("Input content type must be a string.", "invalid_input_content")
        if kind == "encrypted_content" and allow_encrypted_content:
            continue
        allowed = {"input_text", "text"}
        if allow_output_text:
            allowed.add("output_text")
        if allow_multimodal:
            allowed.update({"input_image", "image_url"})
        if not isinstance(kind, str) or kind not in allowed:
            raise ResponsesRequestError("Unsupported input content type.", "unsupported_input_type")
        parts.append(part)
    return _responses_multimodal_content(parts)

def _internal_tool_name(source_type: str, name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    safe_name = safe_name.strip("_") or "tool"
    return f"__codex_{source_type}_{safe_name}"


CODEX_TOOL_SEARCH_BACKEND_NAME = "__codex_tool_search"
CODEX_TOOL_SEARCH_SOURCE_NAME = "tool_search"


def _tool_search_arguments_object(
    arguments: Any,
    *,
    tool_call_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    serialized = validate_tool_call_arguments(
        arguments,
        tool_call_id=tool_call_id,
        tool_name=CODEX_TOOL_SEARCH_SOURCE_NAME,
    )
    try:
        parsed = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ResponsesRequestError(
            "tool_search arguments must be a JSON object.",
            "invalid_tool_search_call",
        ) from exc
    if not isinstance(parsed, dict):
        raise ResponsesRequestError(
            "tool_search arguments must be a JSON object.",
            "invalid_tool_search_call",
        )
    return serialized, parsed


def _tool_search_response_item(
    call_id: str,
    arguments: Any,
    *,
    item_id: str | None = None,
) -> dict[str, Any]:
    _, arguments_object = _tool_search_arguments_object(arguments, tool_call_id=call_id)
    return {
        "id": item_id or f"tsc_{uuid.uuid4().hex}",
        "type": "tool_search_call",
        "execution": "client",
        "call_id": call_id,
        "status": "completed",
        "arguments": arguments_object,
    }


def _tool_search_tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            namespace = tool.get("name")
            for nested in tool.get("tools") or []:
                if isinstance(nested, dict):
                    name = nested.get("name")
                    if isinstance(name, str) and name:
                        names.append(f"{namespace}.{name}" if namespace else name)
        else:
            function = tool.get("function")
            name = function.get("name") if isinstance(function, dict) else tool.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _responses_item_to_message(
    item: Any,
    tool_registry: dict[str, dict[str, Any]] | None = None,
    *,
    input_item_index: int | None = None,
    allow_multimodal: bool = True,
) -> dict[str, Any]:
    if isinstance(item, str):
        return {"role": "user", "content": item}
    if not isinstance(item, dict):
        raise ResponsesRequestError("Each input item must be an object or string.", "invalid_input")

    item_type = item.get("type")
    if item_type is not None and not isinstance(item_type, str):
        raise ResponsesRequestError("Input item type must be a string.", "invalid_input")
    if item_type == "reasoning":
        summary_parts = item.get("summary") or []
        if not isinstance(summary_parts, list):
            raise ResponsesRequestError(
                "Reasoning summary must be an array.", "invalid_reasoning"
            )
        reasoning_parts = []
        for part in summary_parts:
            if not isinstance(part, dict) or part.get("type") != "summary_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                reasoning_parts.append(text)
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": "".join(reasoning_parts),
        }
    if item_type == "agent_message":
        # Codex uses agent_message for prior assistant/commentary turns. The
        # Chat backend has no author/recipient fields, so preserve only content.
        return {
            "role": "assistant",
            "content": _response_content_to_text(
                item.get("content", ""),
                allow_output_text=True,
                allow_encrypted_content=True,
                allow_multimodal=allow_multimodal,
            ),
        }
    if item_type in {
        "function_call", "namespace", "custom_tool_call", "web_search_call",
        "tool_search_call",
    }:
        call_id = item.get("call_id") or item.get("id")
        if item_type == "tool_search_call":
            if item.get("execution") != "client" or item.get("status") not in (None, "completed"):
                raise ResponsesRequestError(
                    "Only completed client tool_search_call items are supported.",
                    "unsupported_tool_execution",
                )
            name = CODEX_TOOL_SEARCH_SOURCE_NAME
            arguments = item.get("arguments", {})
        else:
            name = item.get("name") or ("web_search" if item_type == "web_search_call" else None)
            arguments = item.get("arguments", item.get("input", "{}"))
        if item_type == "custom_tool_call" and isinstance(item.get("input"), str):
            arguments = json.dumps({"input": item["input"]}, ensure_ascii=False)
        if item_type == "web_search_call":
            action = item.get("action")
            arguments = item.get("arguments", item.get("input", action or "{}"))
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ResponsesRequestError(
                f"{item_type} requires call_id and name.",
                f"invalid_{item_type}",
            )
        if item_type == "tool_search_call":
            arguments, _ = _tool_search_arguments_object(arguments, tool_call_id=call_id)
        else:
            arguments = validate_tool_call_arguments(
                arguments,
                tool_call_id=call_id,
                tool_name=name,
            )
        backend_name = (
            CODEX_TOOL_SEARCH_BACKEND_NAME if item_type == "tool_search_call" else name
        )
        if tool_registry:
            for candidate, spec in tool_registry.items():
                if not isinstance(spec, dict) or spec.get("source_name") != name:
                    continue
                if item_type == "tool_search_call" and spec.get("source_type") != "tool_search":
                    continue
                if item.get("namespace") and spec.get("namespace") != item.get("namespace"):
                    continue
                backend_name = candidate
                break
        if item_type == "tool_search_call" and (
            not tool_registry
            or not any(
                isinstance(spec, dict) and spec.get("source_type") == "tool_search"
                for spec in tool_registry.values()
            )
        ):
            raise ResponsesRequestError(
                "tool_search_call was not declared in this conversation.",
                "tool_search_not_declared",
            )
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": backend_name, "arguments": arguments},
            }],
        }
    if item_type in {
        "function_call_output", "custom_tool_call_output", "web_search_call_output",
        "tool_search_output",
    }:
        call_id = item.get("call_id") or item.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ResponsesRequestError(
                f"{item_type} requires call_id.",
                f"invalid_{item_type}",
            )
        if item_type == "tool_search_output":
            loaded_tools = item.get("tools")
            if (
                item.get("execution") != "client"
                or item.get("status") != "completed"
                or not isinstance(loaded_tools, list)
            ):
                raise ResponsesRequestError(
                    "tool_search_output requires client execution, completed status, and a tools array.",
                    "invalid_tool_search_output",
                )
            names = _tool_search_tool_names(loaded_tools)
            content = (
                "Loaded tools: " + ", ".join(names)
                if names else "Tool search completed with no tools loaded."
            )
            return {"role": "tool", "tool_call_id": call_id, "content": content}
        output = item.get("output", item.get("result", ""))
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": _response_content_to_text(output, allow_multimodal=allow_multimodal),
        }

    role = item.get("role")
    if role is not None and not isinstance(role, str):
        raise ResponsesRequestError("Input message role must be a string.", "invalid_input_role")
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        if role is None:
            raise UnsupportedResponsesInputItemError(input_item_index, item)
        raise ResponsesRequestError(
            f"Unsupported input message role {role!r}.",
            "unsupported_input_role",
        )
    message = {
        "role": role,
        "content": _response_content_to_text(
            item.get("content", ""),
            allow_output_text=role == "assistant",
            allow_encrypted_content=role == "assistant",
            allow_multimodal=allow_multimodal,
        ),
    }
    if role == "tool" and item.get("tool_call_id"):
        message["tool_call_id"] = item["tool_call_id"]
    if item.get("tool_calls") is not None:
        message["tool_calls"] = copy.deepcopy(item["tool_calls"])
        validate_chat_tool_calls([message])
    if item.get("reasoning_content") is not None:
        message["reasoning_content"] = item["reasoning_content"]
    return message


def _convert_responses_tools(
    tools: Any,
    *,
    activate_deferred: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if tools is None:
        return [], {}
    if not isinstance(tools, list):
        raise ResponsesRequestError("tools must be an array.", "invalid_tools")
    expanded_tools = []
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "namespace":
            namespace = tool.get("name")
            nested = tool.get("tools")
            if not isinstance(namespace, str) or not namespace or not isinstance(nested, list):
                raise ResponsesRequestError(
                    "Namespace tools require a name and tools array.", "invalid_tool"
                )
            for nested_tool in nested:
                if not isinstance(nested_tool, dict):
                    raise ResponsesRequestError("Namespace entries must be objects.", "invalid_tool")
                child = copy.deepcopy(nested_tool)
                child["_namespace"] = namespace
                expanded_tools.append(child)
        else:
            expanded_tools.append(tool)

    converted = []
    registry = {}
    for tool in expanded_tools:
        if not isinstance(tool, dict):
            raise ResponsesRequestError("Each tool must be an object.", "invalid_tool")
        source_type = tool.get("type")
        namespace = tool.get("_namespace")
        if source_type is not None and not isinstance(source_type, str):
            raise ResponsesRequestError("Tool type must be a string.", "invalid_tool")
        if namespace:
            source_type = "namespace" if source_type == "function" else source_type
        if source_type in {"web_search_preview", "web_search_preview_2025_03_11"}:
            source_type = "web_search"
        if source_type == "tool_search":
            execution = tool.get("execution")
            if execution != "client":
                raise ResponsesRequestError(
                    "Only client-executed tool_search is supported.",
                    "unsupported_tool_execution",
                )
            if namespace:
                raise ResponsesRequestError(
                    "tool_search cannot be nested in a namespace.", "invalid_tool"
                )
            if tool.get("defer_loading", False):
                raise ResponsesRequestError(
                    "tool_search cannot be deferred.", "invalid_tool"
                )
            name = CODEX_TOOL_SEARCH_SOURCE_NAME
            description = tool.get("description", "Search for deferred tools relevant to the task.")
            parameters = tool.get("parameters") or {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
            strict = None
        elif source_type not in {"function", "namespace", "custom", "web_search"}:
            raise ResponsesRequestError(
                f"Unsupported tool type {tool.get('type')!r}.",
                "unsupported_tool_type",
            )
        else:
            function_spec = tool.get("function")
        if source_type in {"function", "namespace"}:
            if isinstance(function_spec, dict):
                name = function_spec.get("name")
                description = function_spec.get("description", tool.get("description", ""))
                parameters = function_spec.get("parameters", tool.get("parameters", {}))
                strict = tool.get("strict", function_spec.get("strict"))
            else:
                name = tool.get("name")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})
                strict = tool.get("strict")
        elif source_type == "custom":
            name = tool.get("name")
            description = tool.get("description", "")
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            }
            strict = None
        elif source_type == "web_search":
            name = tool.get("name", "web_search")
            description = tool.get("description", "Search the web for relevant information.")
            parameters = tool.get("parameters") or {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            }
            strict = None
        defer_loading = tool.get("defer_loading", False)
        if not isinstance(defer_loading, bool):
            raise ResponsesRequestError(
                "defer_loading must be a boolean.", "invalid_tool"
            )
        if defer_loading and not activate_deferred:
            continue
        if not isinstance(name, str) or not name:
            raise ResponsesRequestError("Tool requires name.", "invalid_tool")
        function = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
        if not isinstance(function["description"], str):
            raise ResponsesRequestError(
                "Function tool description must be a string.",
                "invalid_tool",
            )
        if not isinstance(function["parameters"], dict):
            raise ResponsesRequestError(
                "Function tool parameters must be an object.",
                "invalid_tool",
            )
        if source_type == "tool_search":
            backend_name = CODEX_TOOL_SEARCH_BACKEND_NAME
        else:
            backend_name = (
                name if source_type == "function"
                else _internal_tool_name(source_type, f"{namespace}_{name}" if namespace else name)
            )
        if backend_name in registry:
            raise ResponsesRequestError(f"Duplicate tool name {name!r}.", "invalid_tool")
        function["name"] = backend_name
        converted_tool = {"type": "function", "function": function}
        if isinstance(strict, bool):
            converted_tool["function"]["strict"] = strict
        converted.append(converted_tool)
        registry[backend_name] = {
            "source_type": source_type,
            "source_name": name,
            "namespace": namespace,
            "backend_name": backend_name,
            "description": description,
            "parameters": copy.deepcopy(parameters),
            "strict": strict if isinstance(strict, bool) else None,
            "format": tool.get("format"),
            "execution": "proxy" if source_type == "web_search" else "client",
        }
    return converted, registry


def _tool_registry_to_chat_tools(
    tool_registry: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    tools = []
    for backend_name, spec in tool_registry.items():
        if not isinstance(spec, dict):
            continue
        source_type = spec.get("source_type", "function")
        parameters = spec.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
                "additionalProperties": False,
            } if source_type == "custom" else {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }
        function = {
            "name": backend_name,
            "description": spec.get("description", ""),
            "parameters": parameters,
        }
        if isinstance(spec.get("strict"), bool):
            function["strict"] = spec["strict"]
        tools.append({"type": "function", "function": function})
    return tools


def _responses_store_default() -> bool:
    return os.environ.get("RESPONSES_DEFAULT_STORE", "true").lower() in {"1", "true", "yes"}


def _convert_responses_tool_choice(
    tool_choice: Any,
    tool_registry: dict[str, dict[str, Any]] | None = None,
) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str) and tool_choice in {"auto", "none", "required"}:
        return tool_choice
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        choice_name = tool_choice.get("name")
        choice_namespace = tool_choice.get("namespace")
        if choice_type == "namespace" and isinstance(choice_name, str) and "." in choice_name:
            choice_namespace, choice_name = choice_name.split(".", 1)
        if not isinstance(choice_type, str):
            raise ResponsesRequestError(
                "tool_choice type must be a string.", "invalid_tool_choice"
            )
        if choice_type not in {"function", "namespace", "custom", "web_search"} or not isinstance(choice_name, str):
            raise ResponsesRequestError(
                "tool_choice must be auto, none, required, or a function choice.",
                "invalid_tool_choice",
            )
        name = choice_name
        if choice_type == "namespace":
            if tool_registry:
                for backend_name, spec in tool_registry.items():
                    if (spec.get("source_name") == choice_name
                            and spec.get("namespace") == choice_namespace):
                        name = backend_name
                        break
            else:
                name = _internal_tool_name("namespace", f"{choice_namespace}_{choice_name}")
        elif choice_type not in {"function"}:
            name = _internal_tool_name(choice_type, name)
        elif tool_registry:
            for backend_name, spec in tool_registry.items():
                if spec.get("source_name") == name:
                    name = backend_name
                    break
        return {"type": "function", "function": {"name": name}}
    raise ResponsesRequestError(
        "tool_choice must be auto, none, required, or a function choice.",
        "invalid_tool_choice",
    )


def _extract_additional_tools(input_value: Any) -> tuple[Any, list[dict[str, Any]]]:
    """Separate non-message Responses control items from chat messages.

    ``compaction_trigger`` is a Codex-local remote-compaction boundary marker.
    Remote compaction v2 is not implemented by this proxy, so reject it before
    any message conversion, token counting, or P/D dispatch.  In particular,
    never discard the marker and accidentally report a normal response as a
    successful compaction.
    """
    if not isinstance(input_value, list):
        return input_value, []

    message_items = []
    additional_tools = []
    for index, item in enumerate(input_value):
        if isinstance(item, dict) and item.get("type") == "compaction_trigger":
            raise UnsupportedResponsesCompactionError(index, item)
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            message_items.append(item)
            continue
        declared_tools = item.get("tools")
        if not isinstance(declared_tools, list):
            raise ResponsesRequestError(
                "additional_tools requires a tools array.", "invalid_tools"
            )
        additional_tools.extend(copy.deepcopy(declared_tools))
    return message_items, additional_tools


def _validate_tool_search_outputs(
    input_value: Any,
    previous_response: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Validate the client tool-search round trip before constructing Chat history."""
    input_items = input_value if isinstance(input_value, list) else []
    calls: dict[str, tuple[dict[str, Any], int | None]] = {}
    if isinstance(previous_response, dict):
        previous_output = previous_response.get("output")
        if isinstance(previous_output, list):
            normalized_messages = previous_response.get("_normalized_messages", [])
            normalized_call_ids = set()
            if isinstance(normalized_messages, list):
                for message in normalized_messages:
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        continue
                    message_calls = message.get("tool_calls")
                    if not isinstance(message_calls, list):
                        continue
                    normalized_call_ids.update(
                        call.get("id") for call in message_calls
                        if isinstance(call, dict)
                    )
            for item in previous_output:
                if isinstance(item, dict) and item.get("type") == "tool_search_call":
                    call_id = item.get("call_id")
                    if (
                        item.get("execution") != "client"
                        or item.get("status") not in (None, "completed")
                        or not isinstance(call_id, str)
                        or not call_id
                    ):
                        raise ResponsesRequestError(
                            "Stored tool_search_call is invalid.",
                            "previous_response_invalid",
                            409,
                        )
                    if call_id not in normalized_call_ids:
                        raise ResponsesRequestError(
                            "Stored tool_search_call has no matching normalized tool call.",
                            "previous_response_invalid",
                            409,
                        )
                    if call_id in calls:
                        raise ResponsesRequestError(
                            "Stored tool_search_call call_id is duplicated.",
                            "previous_response_invalid",
                            409,
                        )
                    calls[call_id] = (item, None)

    outputs: list[dict[str, Any]] = []
    output_call_ids: set[str] = set()
    output_input_indexes: dict[str, int] = {}
    for input_index, item in enumerate(input_items):
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "tool_search_call":
            call_id = item.get("call_id")
            if (
                item.get("execution") != "client"
                or item.get("status") not in (None, "completed")
                or not isinstance(call_id, str)
                or not call_id
            ):
                raise ResponsesRequestError(
                    "tool_search_call requires client execution, call_id, and completed status if provided.",
                    "invalid_tool_search_call",
                )
            if call_id in calls:
                raise ResponsesRequestError(
                    "tool_search_call call_id is duplicated in the request history.",
                    "invalid_tool_search_call",
                )
            calls[call_id] = (item, input_index)
        elif item_type == "tool_search_output":
            call_id = item.get("call_id")
            if (
                item.get("execution") != "client"
                or item.get("status") != "completed"
                or not isinstance(call_id, str)
                or not call_id
                or not isinstance(item.get("tools"), list)
            ):
                raise ResponsesRequestError(
                    "tool_search_output requires client execution, completed status, call_id, and a tools array.",
                    "invalid_tool_search_output",
                )
            if call_id in output_call_ids:
                raise ResponsesRequestError(
                    "tool_search_output call_id is duplicated.",
                    "invalid_tool_search_output",
                )
            output_call_ids.add(call_id)
            output_input_indexes[call_id] = input_index
            outputs.append(item)

    for output in outputs:
        call_info = calls.get(output["call_id"])
        if call_info is None:
            raise ResponsesRequestError(
                "tool_search_output does not match a client tool_search_call.",
                "tool_search_call_not_found",
            )
        call_input_index = call_info[1]
        output_input_index = output_input_indexes[output["call_id"]]
        if call_input_index is not None and output_input_index <= call_input_index:
            raise ResponsesRequestError(
                "tool_search_output must follow its tool_search_call.",
                "invalid_tool_search_output",
            )
    missing = set(calls) - output_call_ids
    if missing:
        raise ResponsesRequestError(
            "A client tool_search_call is missing its tool_search_output.",
            "missing_tool_search_output",
        )
    return outputs


def _merge_loaded_tool_registry(
    tool_registry: dict[str, dict[str, Any]],
    loaded_registry: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    additions = {}
    comparable_fields = (
        "source_type", "source_name", "namespace", "description", "parameters",
        "strict", "format",
    )
    for backend_name, spec in loaded_registry.items():
        existing = tool_registry.get(backend_name)
        if existing is not None:
            if any(existing.get(field) != spec.get(field) for field in comparable_fields):
                raise ResponsesRequestError(
                    f"Loaded tool {spec.get('source_name')!r} conflicts with an existing tool.",
                    "tool_schema_conflict",
                )
            continue
        tool_registry[backend_name] = copy.deepcopy(spec)
        additions[backend_name] = copy.deepcopy(spec)
    return additions


async def responses_request_to_chat_request(
    response_request: dict[str, Any],
) -> tuple[dict[str, Any], ResponseRequestState]:
    if not isinstance(response_request, dict):
        raise ResponsesRequestError("Request body must be a JSON object.", "invalid_request")
    # Reject unsupported protocol control items before state lookup, tool
    # conversion, token counting, or any possible P/D dispatch.
    input_value, additional_tools = _extract_additional_tools(response_request.get("input", ""))
    previous_response_id = response_request.get("previous_response_id")
    conversation_id = response_request.get("conversation")
    if conversation_id and not isinstance(conversation_id, str):
        raise ResponsesRequestError("conversation must be a string.", "invalid_conversation")
    if conversation_id and not response_state_store.enabled:
        raise ResponsesRequestError(
            "conversation requires RESPONSES_REDIS_URL.",
            "response_state_unavailable",
            503,
        )
    if conversation_id and not previous_response_id:
        previous_response_id = await response_state_store.get_conversation(conversation_id)

    if previous_response_id:
        previous_id = previous_response_id
        if not response_state_store.enabled:
            raise ResponsesRequestError(
                "previous_response_id requires RESPONSES_REDIS_URL.",
                "response_state_unavailable",
                503,
            )
        previous = await response_state_store.get(previous_id)
        if previous is None:
            raise ResponsesRequestError(
                "The previous response does not exist or has expired.",
                "previous_response_not_found",
            )
    else:
        previous = None

    tool_search_outputs = _validate_tool_search_outputs(input_value, previous)

    args = globals().get("global_args")
    request_tools = response_request.get("tools")
    if request_tools is None:
        request_tools = additional_tools or None
    elif isinstance(request_tools, list):
        request_tools = [*request_tools, *additional_tools]
    tools, tool_registry = _convert_responses_tools(request_tools)
    added_registry = {}
    if previous and isinstance(previous.get("_tool_registry"), dict):
        restored_registry = previous["_tool_registry"]
        for backend_name, spec in restored_registry.items():
            if backend_name not in tool_registry and isinstance(spec, dict):
                tool_registry[backend_name] = copy.deepcopy(spec)
                added_registry[backend_name] = copy.deepcopy(spec)
    for search_output in tool_search_outputs:
        _, loaded_registry = _convert_responses_tools(
            search_output["tools"], activate_deferred=True
        )
        unsupported = [
            spec for spec in loaded_registry.values()
            if spec.get("source_type") not in {"function", "namespace", "custom"}
        ]
        if unsupported:
            raise ResponsesRequestError(
                "tool_search_output currently supports function, custom, and namespace tools only.",
                "unsupported_tool_type",
            )
        added_registry.update(_merge_loaded_tool_registry(tool_registry, loaded_registry))
    if added_registry:
        tools.extend(_tool_registry_to_chat_tools(added_registry))
    if not tools and tool_registry:
        tools = _tool_registry_to_chat_tools(tool_registry)

    model = response_request.get("model") or (previous or {}).get("model") or ""
    if not isinstance(model, str) or not model:
        raise ResponsesRequestError("model is required.", "missing_model")

    messages: list[dict[str, Any]] = []
    instructions = response_request.get("instructions")
    if instructions is not None:
        messages.append({
            "role": "system",
            "content": _response_content_to_text(instructions),
        })
    if previous:
        previous_messages = previous.get("_normalized_messages", [])
        if isinstance(previous_messages, list):
            previous_messages = copy.deepcopy(previous_messages)
            try:
                validate_chat_tool_calls(previous_messages)
            except ResponsesRequestError as exc:
                raise ResponsesRequestError(
                    "The previous response contains an invalid function call.",
                    "previous_response_invalid",
                    409,
                ) from exc
            messages.extend(previous_messages)

    if isinstance(input_value, list):
        messages.extend(
            _responses_item_to_message(
                item,
                tool_registry,
                input_item_index=index,
                allow_multimodal=True,
            )
            for index, item in enumerate(input_value)
        )
    else:
        messages.append({
            "role": "user",
            "content": _response_content_to_text(input_value),
        })

    validate_chat_tool_calls(messages)

    if tools:
        for message in messages:
            if message["role"] in {"system", "developer"}:
                message["tools"] = tools
                break
        else:
            messages.insert(0, {"role": "system", "content": "", "tools": tools})

    reasoning = response_request.get("reasoning")
    if reasoning is not None and not isinstance(reasoning, dict):
        raise ResponsesRequestError("reasoning must be an object.", "invalid_reasoning_effort")
    reasoning = reasoning or {}
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
    if effort is not None and not isinstance(effort, str):
        raise ResponsesRequestError("reasoning effort must be a string.", "invalid_reasoning_effort")

    chat_request = copy.deepcopy(response_request)
    chat_request.pop("input", None)
    chat_request.pop("instructions", None)
    chat_request.pop("reasoning", None)
    chat_request.pop("metadata", None)
    chat_request.pop("store", None)
    chat_request.pop("previous_response_id", None)
    chat_request.pop("conversation", None)
    chat_request.pop("tools", None)
    chat_request.pop("tool_choice", None)
    chat_request["model"] = model
    chat_request["messages"] = messages
    requested_max_tokens = response_request.get(
        "max_output_tokens",
        response_request.get("max_tokens", RESPONSES_DEFAULT_MAX_OUTPUT_TOKENS),
    )
    if isinstance(requested_max_tokens, int) and not isinstance(requested_max_tokens, bool):
        requested_max_tokens = max(RESPONSES_MIN_OUTPUT_TOKENS, requested_max_tokens)
    chat_request["max_tokens"] = requested_max_tokens
    chat_request.pop("max_output_tokens", None)
    chat_request["reasoning_effort"] = effort
    chat_request["stream"] = bool(response_request.get("stream", False))
    if tools:
        chat_request["tools"] = tools
    tool_choice = _convert_responses_tool_choice(response_request.get("tool_choice"), tool_registry)
    if tool_choice is not None:
        if not tools:
            if isinstance(tool_choice, str) and tool_choice in {"auto", "none"}:
                # Chat Completions rejects tool_choice without tools. These
                # values are already the backend defaults when no tools exist.
                tool_choice = None
            else:
                raise ResponsesRequestError(
                    "tool_choice requires tools to be provided.",
                    "tool_choice_requires_tools",
                )
        if tool_choice is not None:
            chat_request["tool_choice"] = tool_choice

    chat_request["messages"] = normalize_chat_messages_for_backend(messages)
    reasoning_settings = resolve_reasoning_settings(chat_request, source="responses")
    chat_request = normalize_reasoning_effort_for_backend(chat_request, reasoning_settings=reasoning_settings)
    effort = reasoning_settings["effective_client_effort"]
    logger.info("reasoning_normalized %s", json.dumps(reasoning_settings, ensure_ascii=True))

    response_id = f"resp_{uuid.uuid4().hex}"
    state = ResponseRequestState(
        response_id=response_id,
        model=model,
        created_at=int(time.time()),
        store=response_request.get("store", _responses_store_default()),
        metadata=response_request.get("metadata") or {},
        previous_response_id=previous_response_id,
        conversation_id=conversation_id,
        reasoning_effort=effort,
        reasoning_settings=reasoning_settings,
        tool_registry=tool_registry,
        backend_route="pd",
        contains_images=_has_multimodal_content(chat_request.get("messages", [])),
        backend_model=getattr(args, "backend_model_name", None),
    )
    return chat_request, state


def _synthetic_request(original: Request, body: bytes) -> Request:
    delivered = False

    async def receive():
        nonlocal delivered
        if delivered:
            # The internal completions handler polls this request for client
            # disconnects while streaming. Returning http.disconnect here
            # would make every Responses request look disconnected after its
            # synthetic body was read, causing the worker to drop all output.
            return await original.receive()
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    scope = dict(original.scope)
    scope["path"] = "/v1/chat/completions"
    scope["raw_path"] = b"/v1/chat/completions"
    return Request(scope, receive)


def _choice_to_response_output(
    chat_response: dict[str, Any],
    tool_registry: dict[str, dict[str, Any]] | None = None,
    state: ResponseRequestState | None = None,
) -> list[dict[str, Any]]:
    choices = chat_response.get("choices") or []
    if not choices:
        return []
    message = choices[0].get("message") or {}
    output: list[dict[str, Any]] = []
    reasoning = _extract_reasoning_text(message)
    if isinstance(reasoning, str) and reasoning:
        output.append({
            "id": f"rs_{uuid.uuid4().hex}",
            "type": "reasoning",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reasoning}],
        })
    tool_calls = message.get("tool_calls") or []
    for call in tool_calls:
        function = call.get("function") or {}
        if not function.get("name"):
            continue
        decoder_call_id = call.get("id")
        call_id = decoder_call_id or f"call_{uuid.uuid4().hex}"
        backend_name = function["name"]
        state_for_tools = state or ResponseRequestState(
            response_id="-", model="", created_at=0, store=False,
            tool_registry=tool_registry or {},
        )
        tool_spec, arguments, nested_tool_wrapped = _normalize_decoder_tool_call(
            backend_name,
            function.get("arguments", "{}"),
            decoder_call_id if (
                backend_name in CODEX_NESTED_EXEC_TOOL_NAMES
                or backend_name in CODEX_FUNCTIONS_EXEC_DECODER_ALIASES
                or backend_name in CODEX_DEFERRED_NESTED_EXEC_ALIASES
                or _exec_compatibility_kind(backend_name, state_for_tools) is not None
            ) else call_id,
            state_for_tools,
        )
        source_type = tool_spec.get("source_type", "function")
        source_name = tool_spec.get("source_name", backend_name)
        if not nested_tool_wrapped:
            arguments = validate_tool_call_arguments(
                arguments,
                tool_call_id=call_id,
                tool_name=backend_name,
            )
        if source_type == "custom":
            custom_input = _custom_input_from_arguments(arguments)
            output.append({
                "id": f"ctc_{uuid.uuid4().hex}",
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": source_name,
                "input": custom_input,
                "status": "completed",
                **({"namespace": tool_spec["namespace"]} if tool_spec.get("namespace") else {}),
            })
            continue
        if source_type == "web_search":
            try:
                action = json.loads(arguments)
            except (TypeError, json.JSONDecodeError):
                action = {"query": arguments}
            output.append({
                "id": f"ws_{uuid.uuid4().hex}",
                "type": "web_search_call",
                "call_id": call_id,
                "status": "completed",
                "action": {"type": "search", **action},
            })
            continue
        if source_type == "tool_search":
            if not isinstance(call_id, str) or not call_id:
                raise ResponsesRequestError(
                    "tool_search call is missing id.", "incomplete_function_call"
                )
            output.append(_tool_search_response_item(call_id, arguments))
            continue
        output_item = {
            "id": f"fc_{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": call_id,
            "name": source_name,
            "arguments": arguments,
            "status": "completed",
        }
        if tool_spec.get("namespace"):
            output_item["namespace"] = tool_spec["namespace"]
        output.append(output_item)
    content = message.get("content") or ""
    if content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        })
    return output


def _extract_reasoning_text(value: Any) -> str:
    """Normalize the reasoning field variants emitted by supported backends."""
    if not isinstance(value, dict):
        return ""
    for key in ("reasoning", "reasoning_content", "reasoning_summary"):
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def _custom_input_from_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except (TypeError, json.JSONDecodeError):
        return arguments
    return parsed.get("input", arguments) if isinstance(parsed, dict) else parsed


def _codex_functions_exec_spec(tool_registry: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Return the uniquely declared custom functions.exec tool, if present."""
    matches = [
        spec for spec in tool_registry.values()
        if isinstance(spec, dict)
        and spec.get("namespace") == "functions"
        and spec.get("source_type") == "custom"
        and spec.get("source_name") == "exec"
    ]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _log_codex_nested_tool_event(
    event: str,
    *,
    state: ResponseRequestState,
    tool_name: Any,
    call_id: Any,
    arguments: Any,
    reason: str | None = None,
    mapped_tool: str | None = None,
    conversion_type: str | None = None,
    normalization: dict[str, Any] | None = None,
) -> None:
    serialized = arguments if isinstance(arguments, str) else json.dumps(
        arguments, ensure_ascii=False, default=str
    )
    fields = (
        _active_client_request_id.get() or "-",
        state.response_id,
        tool_name if isinstance(tool_name, str) else "-",
        call_id if isinstance(call_id, str) else "-",
        len(serialized.encode("utf-8")),
        hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    )
    detail = " ".join(f"{key}={value}" for key, value in (normalization or {}).items())
    if reason:
        logger.warning(
            "%s request_id=%s response_id=%s tool_name=%s tool_call_id=%s "
            "argument_bytes=%s argument_sha256=%s reason=%s %s",
            event, *fields, reason, detail,
        )
    else:
        logger.info(
            "%s request_id=%s response_id=%s tool_name=%s tool_call_id=%s "
            "argument_bytes=%s argument_sha256=%s mapped_tool=%s conversion_type=%s %s",
            event, *fields, mapped_tool or "-", conversion_type or "-", detail,
        )


CODEX_NESTED_EXEC_TOOL_NAMES = frozenset({
    "apply_patch",
    "exec",
    "exec_command",
    "write_stdin",
    "update_plan",
    "update_goal",
    "create_goal",
    "get_goal",
    "view_image",
})
CODEX_NESTED_EXEC_OBJECT_TOOLS = CODEX_NESTED_EXEC_TOOL_NAMES - {"apply_patch", "exec"}
CODEX_FUNCTIONS_EXEC_DECODER_ALIASES = frozenset({
    "__codec_custom_functions_exec",
    "__codec_functions_exec",
    # Decoder variants observed in Codex streams. Names establish eligibility;
    # codex_exec_normalization determines payload language and declared target.
    "codex_exec",
    "exec_code",
    "codec_exec",
    "code_exec",
    "codex_execute",
    "code_execution",
    "codex_functions.exec",
    "__codex_mcp_exec",
    "__codex_mcp__codex_namespace_functions_exec",
    "__codex_exec_code",
    "__codex_exec",
    "codex_functions_exec",
    "__codex_functions_exec",
    "__codex_common_functions_exec",
    "__codex_custom_exec",
    "__codex_custom_functions_exec",
    "__codex_volcengine_functions_exec",
    "__exec",
    "codex_namespace_functions_exec",
    "codex custom functions exec",
    "__codex_functions.exec",
    "__codex_mem_functions_exec",
    "__codex_mcp__exec",
    "__codex_mcp__codegen_tool_exec",
})
CODEX_WAIT_DECODER_ALIASES = frozenset({
    "__codex_functions_wait",
    "codex_namespace_functions_wait",
    # Decoder may omit the codex prefix from the declared functions namespace.
    "__namespace_functions_wait",
})
CODEX_REQUEST_USER_INPUT_DECODER_ALIASES = frozenset({
    # Exact namespace variant observed in DSv4.1 Decoder output.
    "codex_namespace_functions_request_user_input",
})
CODEX_EXEC_COMMAND_DECODER_ALIASES = frozenset({
    "__codec_exec_command",
    "__codex_custom_exec_command",
    # Prefix/bare variants share the same content-based normalization as exec.
    "codex_exec_command",
    "codex.exec_command",
    "__codex_exec_command",
    "__codex_mcp__exec_command",
})
CODEX_PATCH_DECODER_ALIASES = frozenset({
    "__codex_patch",
    "__codex_apply_patch",
    "codex_apply_patch",
    "__codex_builtin_apply_patch",
    "__codex_functions_apply_patch",
    "__codex_custom_patch",
})
CODEX_UPDATE_GOAL_DECODER_ALIASES = frozenset({
    "__codex_namespace_functions_update_goal",
})
CODEX_DEFERRED_NESTED_EXEC_ALIASES = (
    CODEX_EXEC_COMMAND_DECODER_ALIASES | CODEX_UPDATE_GOAL_DECODER_ALIASES
    | CODEX_PATCH_DECODER_ALIASES
)
CODEX_PROTOCOL_LEAK_MARKERS = (
    "<tool_call", "</tool_call", "<invoke", "</invoke", "<parameter",
    "</parameter", "<｜DSML｜",
)

# Exact Decoder artifacts observed after structured tool calls. This deliberately
# does not buffer chunks or attempt to recognize split markers.
RESPONSES_DSML_TOOL_TAIL_MARKERS = (
    "</｜DSML｜ calls>",
    "</｜DSML｜ invoke>",
    "</｜DSML｜ parameter>",
)


def _resolve_unique_custom_functions_containment(
    tool_name: str,
    tool_registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Map only a uniquely declared custom-functions tool embedded in a Decoder name."""
    if "custom_functions" not in tool_name:
        return None
    matches = [
        spec
        for declared_name, spec in tool_registry.items()
        if (
            isinstance(declared_name, str)
            and isinstance(spec, dict)
            and "custom_functions" in declared_name
            and declared_name in tool_name
        )
    ]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _resolve_unique_custom_functions_exec_suffix(
    tool_name: str,
    tool_registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Repair a Decoder prefix variation only for one declared exec wrapper."""
    suffix = "custom_functions_exec"
    if not tool_name.endswith(suffix):
        return None
    matches = [
        spec
        for declared_name, spec in tool_registry.items()
        if (
            isinstance(declared_name, str)
            and isinstance(spec, dict)
            and declared_name.endswith(suffix)
        )
    ]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _parse_codex_nested_tool_object(tool_name: str, arguments: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
    except (TypeError, json.JSONDecodeError):
        parsed = None
    if not isinstance(parsed, dict):
        raise ResponsesRequestError(
            f"Codex {tool_name} arguments must be a JSON object.",
            "invalid_function_arguments",
        )
    if tool_name == "get_goal" and parsed:
        raise ResponsesRequestError(
            "Codex get_goal arguments must be empty.", "invalid_function_arguments"
        )
    if tool_name == "create_goal" and not isinstance(parsed.get("objective"), str):
        raise ResponsesRequestError(
            "Codex create_goal requires an objective.", "invalid_function_arguments"
        )
    if tool_name == "update_goal" and parsed.get("status") not in {"complete", "blocked"}:
        raise ResponsesRequestError(
            "Codex update_goal requires a valid status.", "invalid_function_arguments"
        )
    if tool_name == "update_plan" and not isinstance(parsed.get("plan"), list):
        raise ResponsesRequestError(
            "Codex update_plan requires a plan array.", "invalid_function_arguments"
        )
    if tool_name == "view_image" and not isinstance(parsed.get("path"), str):
        raise ResponsesRequestError(
            "Codex view_image requires a path.", "invalid_function_arguments"
        )
    return parsed


def _parse_exact_codex_tool_object(
    tool_name: str,
    arguments: Any,
    expected_keys: set[str],
) -> dict[str, Any]:
    """Parse a compatibility alias without accepting extra model-generated fields."""
    parsed = _parse_codex_nested_tool_object(tool_name, arguments)
    if set(parsed) != expected_keys:
        raise ResponsesRequestError(
            f"Codex {tool_name} arguments must contain exactly {sorted(expected_keys)!r}.",
            "invalid_function_arguments",
        )
    return parsed


def _validate_codex_wait_arguments(arguments: Any) -> None:
    parsed = _parse_codex_nested_tool_object("wait", arguments)
    allowed_keys = {"cell_id", "yield_time_ms", "max_tokens", "terminate"}
    if set(parsed) - allowed_keys:
        raise ResponsesRequestError(
            "Codex wait arguments contain unsupported fields.",
            "invalid_function_arguments",
        )
    if not isinstance(parsed.get("cell_id"), str) or not parsed["cell_id"].strip():
        raise ResponsesRequestError(
            "Codex wait requires a non-empty cell_id string.",
            "invalid_function_arguments",
        )
    for key in ("yield_time_ms", "max_tokens"):
        if key in parsed and (
            isinstance(parsed[key], bool)
            or not isinstance(parsed[key], (int, float))
        ):
            raise ResponsesRequestError(
                f"Codex wait {key} must be a number.",
                "invalid_function_arguments",
            )
    if "terminate" in parsed and not isinstance(parsed["terminate"], bool):
        raise ResponsesRequestError(
            "Codex wait terminate must be a boolean.",
            "invalid_function_arguments",
        )


def _codex_wait_spec(
    tool_registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    matches = [
        spec for spec in tool_registry.values()
        if (
            isinstance(spec, dict)
            and spec.get("namespace") == "functions"
            and spec.get("source_name") == "wait"
        )
    ]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _codex_namespace_tool_spec(
    source_name: str,
    tool_registry: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve one uniquely declared functions namespace tool by source name."""
    matches = [
        spec for spec in tool_registry.values()
        if (
            isinstance(spec, dict)
            and spec.get("namespace") == "functions"
            and spec.get("source_name") == source_name
        )
    ]
    return copy.deepcopy(matches[0]) if len(matches) == 1 else None


def _resolve_decoder_tool_spec(
    tool_name: Any,
    state: ResponseRequestState,
) -> tuple[dict[str, Any] | None, bool, bool, bool, bool, bool]:
    """Resolve declared tools, then pass through unresolved Codex custom calls."""
    if not isinstance(tool_name, str):
        return None, False, False, False, False, False
    declared = state.tool_registry.get(tool_name)
    if isinstance(declared, dict):
        return declared, False, False, False, False, False
    if tool_name in CODEX_WAIT_DECODER_ALIASES:
        wait_spec = _codex_wait_spec(state.tool_registry)
        if isinstance(wait_spec, dict):
            return wait_spec, False, False, False, False, True
    if tool_name in CODEX_REQUEST_USER_INPUT_DECODER_ALIASES:
        request_user_input_spec = _codex_namespace_tool_spec(
            "request_user_input", state.tool_registry,
        )
        if isinstance(request_user_input_spec, dict):
            return request_user_input_spec, False, False, False, False, False
    if tool_name in CODEX_FUNCTIONS_EXEC_DECODER_ALIASES:
        exec_spec = _codex_functions_exec_spec(state.tool_registry)
        if exec_spec is not None:
            return exec_spec, True, False, False, False, False
    contained_spec = _resolve_unique_custom_functions_containment(tool_name, state.tool_registry)
    if contained_spec is not None:
        return contained_spec, False, True, False, False, False
    suffix_spec = _resolve_unique_custom_functions_exec_suffix(tool_name, state.tool_registry)
    if suffix_spec is not None:
        return suffix_spec, False, False, True, False, False
    if "custom_functions" in tool_name:
        # The Decoder may extend a custom-functions tool name. Preserve it for the
        # Codex client only after every declared-tool repair path missed.
        return {
            "source_type": "function",
            "source_name": tool_name,
            "backend_name": tool_name,
        }, False, False, False, True, False
    return None, False, False, False, False, False


def _is_functions_exec_spec(spec):
    return (isinstance(spec, dict) and spec.get("namespace") == "functions"
            and spec.get("source_type") == "custom" and spec.get("source_name") == "exec")


def _exec_compatibility_kind(tool_name, state):
    """Name eligibility only; never inspect incomplete streaming arguments."""
    if not isinstance(tool_name, str) or tool_name in state.tool_registry:
        return None
    if tool_name in CODEX_FUNCTIONS_EXEC_DECODER_ALIASES | CODEX_EXEC_COMMAND_DECODER_ALIASES:
        return "explicit_alias"
    if tool_name in {"exec", "exec_command"}:
        return "bare_name"
    contained = _resolve_unique_custom_functions_containment(tool_name, state.tool_registry)
    if _is_functions_exec_spec(contained):
        return "containment"
    suffix = _resolve_unique_custom_functions_exec_suffix(tool_name, state.tool_registry)
    if _is_functions_exec_spec(suffix):
        return "suffix"
    # Do not let an unresolved exec candidate fall through to the legacy
    # arbitrary custom-functions passthrough. Non-exec passthrough is unchanged.
    if "custom_functions" in tool_name and re.search(r"(?:^|[_.\s])exec(?:$|[_.\s])", tool_name):
        return "unresolved_exec"
    return None


def _normalize_exec_compatibility(tool_name, arguments, call_id, state, kind):
    serialized = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    key = (tool_name, call_id if isinstance(call_id, str) else None,
           hashlib.sha256(serialized.encode("utf-8")).hexdigest())
    cached = state.exec_normalization_cache.get(key)
    if cached is not None:
        return copy.deepcopy(cached)
    started = time.perf_counter()
    try:
        result = normalize_exec(tool_name, arguments, call_id, state.tool_registry,
                                kind, CODEX_EXEC_COMMAND_PARAMETERS, CODEX_PROTOCOL_LEAK_MARKERS)
    except ExecNormalizationError as exc:
        _log_codex_nested_tool_event(
            "codex_nested_tool_rejected", state=state, tool_name=tool_name,
            call_id=call_id, arguments=arguments, reason=exc.code,
            normalization={"resolution_kind": kind, "classification_reason": exc.reason,
                           "envelope_type": exc.envelope_type, "payload_type": exc.payload_type,
                           "normalization_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        raise ResponsesRequestError(str(exc), exc.code) from None
    _log_codex_nested_tool_event(
        "codex_decoder_tool_alias_mapped", state=state, tool_name=tool_name,
        call_id=call_id, arguments=arguments, mapped_tool=result.target_tool["backend_name"],
        conversion_type=result.conversion_type,
        normalization={"resolution_kind": result.resolution_kind,
                       "envelope_type": result.envelope_type, "payload_type": result.payload_type,
                       "classification_reason": result.classification_reason,
                       "normalization_ms": round((time.perf_counter() - started) * 1000, 3)},
    )
    normalized = (result.target_tool, result.normalized_arguments, result.requires_deferred_output)
    state.exec_normalization_cache[key] = copy.deepcopy(normalized)
    return normalized


# Contract used only when the declared functions.exec documentation exposes the
# nested exec_command tool. Direct function declarations supply their own schema.
CODEX_EXEC_COMMAND_PARAMETERS = {
    "type": "object", "required": ["cmd"], "additionalProperties": False,
    "properties": {
        **{key: {"type": "string"} for key in (
            "cmd", "workdir", "shell", "justification",
        )},
        **{key: {"type": "boolean"} for key in ("login", "tty")},
        **{key: {"type": "number", "minimum": 0} for key in (
            "yield_time_ms", "max_output_tokens",
        )},
        "sandbox_permissions": {"enum": ["use_default", "require_escalated"]},
        "prefix_rule": {"type": "array", "items": {"type": "string"}},
    },
}


def _normalize_codex_patch_alias(
    tool_name: str, arguments: Any, call_id: Any, state: ResponseRequestState,
) -> tuple[dict[str, Any], str, bool]:
    """Keep the independent structured-patch repair contract."""
    try:
        if not isinstance(call_id, str) or not call_id:
            raise ResponsesRequestError("Function call is missing id.", "incomplete_function_call")
        parsed = _parse_exact_codex_tool_object(tool_name, arguments, {"input"})
        value = parsed["input"]
        if not isinstance(value, str) or not value.strip():
            raise ResponsesRequestError("Alias input must be a non-empty string.", "invalid_function_arguments")
        candidates = [spec for spec in state.tool_registry.values()
                      if spec.get("source_name") == "apply_patch"
                      and spec.get("namespace") in {None, "functions"}
                      and spec.get("source_type") == "custom"]
        exec_spec = _codex_functions_exec_spec(state.tool_registry)
        if len(candidates) > 1 or (not candidates and exec_spec is None):
            raise ResponsesRequestError("Alias target is not uniquely declared.", "undeclared_tool")
        direct = candidates[0] if candidates else None
        lines = value.splitlines()
        if (len(lines) < 3 or lines[0] != "*** Begin Patch"
                or lines[-1] != "*** End Patch"):
            raise ResponsesRequestError("Patch input must have complete patch boundaries.", "invalid_function_arguments")
        result_arguments = {"input": value}
        javascript = f"await tools.apply_patch({json.dumps(value, ensure_ascii=False)});"
        spec = direct or exec_spec
        serialized = json.dumps(result_arguments if direct else {"input": javascript}, ensure_ascii=False)
        _log_codex_nested_tool_event(
            "codex_decoder_tool_alias_mapped", state=state, tool_name=tool_name,
            call_id=call_id, arguments=arguments,
            mapped_tool=spec["backend_name"], conversion_type="patch",
        )
        # Defer the complete artifact in streaming paths, including direct tools.
        return copy.deepcopy(spec), serialized, True
    except ResponsesRequestError as exc:
        _log_codex_nested_tool_event(
            "codex_nested_tool_rejected", state=state, tool_name=tool_name,
            call_id=call_id, arguments=arguments, reason=exc.code,
        )
        raise


def _normalize_decoder_tool_call(
    tool_name: Any,
    arguments: Any,
    call_id: Any,
    state: ResponseRequestState,
) -> tuple[dict[str, Any], str, bool]:
    """Map known bare Codex nested tools back through declared functions.exec."""
    original_tool_name = tool_name
    declared = state.tool_registry.get(tool_name) if isinstance(tool_name, str) else None
    if isinstance(declared, dict):
        return declared, arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False), False
    kind = _exec_compatibility_kind(tool_name, state)
    if kind is not None:
        return _normalize_exec_compatibility(tool_name, arguments, call_id, state, kind)
    if tool_name in CODEX_PATCH_DECODER_ALIASES:
        return _normalize_codex_patch_alias(tool_name, arguments, call_id, state)
    if tool_name in CODEX_WAIT_DECODER_ALIASES:
        wait_spec = _codex_wait_spec(state.tool_registry)
        if not isinstance(wait_spec, dict):
            _log_codex_nested_tool_event(
                "codex_nested_tool_rejected",
                state=state,
                tool_name=original_tool_name,
                call_id=call_id,
                arguments=arguments,
                reason="undeclared_decoder_tool",
            )
            raise ResponsesRequestError(
                "Decoder called a wait tool that was not declared by the client.",
                "undeclared_tool",
            )
        try:
            _validate_codex_wait_arguments(arguments)
        except ResponsesRequestError as exc:
            _log_codex_nested_tool_event(
                "codex_nested_tool_rejected",
                state=state,
                tool_name=original_tool_name,
                call_id=call_id,
                arguments=arguments,
                reason=exc.code,
            )
            raise
        _log_codex_nested_tool_event(
            "codex_decoder_tool_alias_mapped",
            state=state,
            tool_name=original_tool_name,
            call_id=call_id,
            arguments=arguments,
        )
        return wait_spec, arguments if isinstance(arguments, str) else json.dumps(
            arguments, ensure_ascii=False
        ), False
    if tool_name in CODEX_UPDATE_GOAL_DECODER_ALIASES:
        try:
            _parse_exact_codex_tool_object(tool_name, arguments, {"status"})
            tool_name = "update_goal"
        except ResponsesRequestError as exc:
            _log_codex_nested_tool_event(
                "codex_nested_tool_rejected",
                state=state,
                tool_name=original_tool_name,
                call_id=call_id,
                arguments=arguments,
                reason=exc.code,
            )
            raise
    (
        declared,
        _is_functions_exec_alias,
        is_codex_custom_containment,
        is_custom_functions_exec_suffix_alias,
        is_codex_custom_passthrough,
        _is_wait_alias,
    ) = _resolve_decoder_tool_spec(tool_name, state)
    if declared is not None:
        if is_codex_custom_containment:
            _log_codex_nested_tool_event(
                "codex_decoder_tool_containment_mapped",
                state=state,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            )
        elif is_custom_functions_exec_suffix_alias:
            _log_codex_nested_tool_event(
                "codex_decoder_tool_exec_suffix_mapped",
                state=state,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            )
        elif is_codex_custom_passthrough:
            _log_codex_nested_tool_event(
                "codex_decoder_tool_passthrough",
                state=state,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
            )
        return declared, arguments if isinstance(arguments, str) else json.dumps(
                arguments, ensure_ascii=False
        ), False

    if tool_name not in CODEX_NESTED_EXEC_TOOL_NAMES:
        _log_codex_nested_tool_event(
            "codex_nested_tool_rejected",
            state=state,
            tool_name=tool_name,
            call_id=call_id,
            arguments=arguments,
            reason="undeclared_decoder_tool",
        )
        raise ResponsesRequestError(
            "Decoder called a tool that was not declared by the client.",
            "undeclared_tool",
        )

    exec_spec = _codex_functions_exec_spec(state.tool_registry)
    if exec_spec is None:
        _log_codex_nested_tool_event(
            "codex_nested_tool_rejected",
            state=state,
            tool_name=tool_name,
            call_id=call_id,
            arguments=arguments,
            reason="functions_exec_not_declared",
        )
        raise ResponsesRequestError(
            "Codex nested tool requires declared functions.exec.",
            "invalid_function_arguments",
        )
    if not isinstance(call_id, str) or not call_id:
        _log_codex_nested_tool_event(
            "codex_nested_tool_rejected",
            state=state,
            tool_name=tool_name,
            call_id=call_id,
            arguments=arguments,
            reason="missing_call_id",
        )
        raise ResponsesRequestError("Function call is missing id.", "incomplete_function_call")

    if tool_name == "apply_patch":
        if not isinstance(arguments, str) or not arguments.strip():
            _log_codex_nested_tool_event(
                "codex_nested_tool_rejected",
                state=state,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason="empty_patch",
            )
            raise ResponsesRequestError(
                f"Codex {tool_name} input must be non-empty.", "invalid_function_arguments"
            )
        javascript = f"await tools.apply_patch({json.dumps(arguments, ensure_ascii=False)});"
    else:
        try:
            tool_input = _parse_codex_nested_tool_object(tool_name, arguments)
        except ResponsesRequestError as exc:
            _log_codex_nested_tool_event(
                "codex_nested_tool_rejected",
                state=state,
                tool_name=tool_name,
                call_id=call_id,
                arguments=arguments,
                reason=exc.code,
            )
            raise
        javascript = f"await tools.{tool_name}(" + json.dumps(
            tool_input, ensure_ascii=False, separators=(",", ":")
        ) + ");"

    wrapped_arguments = validate_tool_call_arguments(
        json.dumps({"input": javascript}, ensure_ascii=False),
        request_id=_active_client_request_id.get(),
        tool_call_id=call_id,
        tool_name=exec_spec["backend_name"],
    )
    _log_codex_nested_tool_event(
        "codex_nested_tool_wrapped",
        state=state,
        tool_name=tool_name,
        call_id=call_id,
        arguments=arguments,
    )
    return exec_spec, wrapped_arguments, True


def chat_result_to_response_object(
    chat_response: dict[str, Any],
    req_data: dict[str, Any],
    state: ResponseRequestState,
    input_tokens: int | None = None,
) -> dict[str, Any]:
    usage = chat_response.get("usage") or {}
    input_count = usage.get("prompt_tokens", input_tokens or 0)
    output_count = usage.get("completion_tokens", 0)
    return {
        "id": state.response_id,
        "object": "response",
        "created_at": state.created_at,
        "status": "completed",
        "model": state.model,
        "metadata": state.metadata,
        "output": _choice_to_response_output(chat_response, state.tool_registry, state),
        "usage": {
            "input_tokens": input_count,
            "output_tokens": output_count,
            "total_tokens": input_count + output_count,
        },
    }


def _response_output_to_assistant_message(
    response_object: dict[str, Any],
    tool_registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    output = response_object.get("output") or []
    content_parts = []
    tool_calls = []
    reasoning_parts = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        content_parts.append(text)
        elif item.get("type") == "reasoning":
            for summary in item.get("summary") or []:
                if isinstance(summary, dict) and isinstance(summary.get("text"), str):
                    reasoning_parts.append(summary["text"])
        elif item.get("type") in {
            "function_call", "namespace", "custom_tool_call", "web_search_call",
            "tool_search_call",
        }:
            call_id = item.get("call_id")
            item_type = item.get("type")
            name = (
                CODEX_TOOL_SEARCH_SOURCE_NAME if item_type == "tool_search_call"
                else item.get("name") or ("web_search" if item_type == "web_search_call" else None)
            )
            arguments = item.get("arguments", item.get("input", item.get("action", "{}")))
            if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                if item_type == "custom_tool_call":
                    arguments = json.dumps({"input": arguments}, ensure_ascii=False)
                    backend_name = None
                    if tool_registry:
                        for candidate, spec in tool_registry.items():
                            if (spec.get("source_type") == "custom"
                                    and spec.get("source_name") == name
                                    and (not item.get("namespace")
                                         or spec.get("namespace") == item.get("namespace"))):
                                backend_name = candidate
                                break
                    name = backend_name or _internal_tool_name("custom", name)
                elif item_type == "web_search_call":
                    name = _internal_tool_name("web_search", name)
                elif item_type == "tool_search_call":
                    name = CODEX_TOOL_SEARCH_BACKEND_NAME
                elif tool_registry:
                    for backend_name, spec in tool_registry.items():
                        if (spec.get("source_name") == name
                                and (not item.get("namespace") or spec.get("namespace") == item.get("namespace"))):
                            name = backend_name
                            break
                if item_type == "tool_search_call":
                    arguments, _ = _tool_search_arguments_object(
                        arguments, tool_call_id=call_id
                    )
                else:
                    arguments = validate_tool_call_arguments(
                        arguments,
                        tool_call_id=call_id,
                        tool_name=name,
                    )
                tool_calls.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                    },
                })
    if not content_parts and not reasoning_parts and not tool_calls:
        return None
    message = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _state_payload(
    response_object: dict[str, Any],
    req_data: dict[str, Any],
    state: ResponseRequestState | None = None,
) -> dict[str, Any]:
    payload = copy.deepcopy(response_object)
    payload["_normalized_messages"] = copy.deepcopy(req_data.get("messages", []))
    payload["_tool_registry"] = copy.deepcopy(state.tool_registry if state else {})
    payload["_backend_route"] = "pd"
    payload["_backend_model"] = getattr(globals().get("global_args"), "backend_model_name", None) or req_data.get("model")
    payload["_contains_images"] = _has_multimodal_content(req_data.get("messages", []))
    assistant_message = _response_output_to_assistant_message(
        response_object, state.tool_registry if state else None
    )
    if assistant_message is not None:
        payload["_normalized_messages"].append(assistant_message)
    return payload


async def _store_response_state(
    response_object: dict[str, Any],
    req_data: dict[str, Any],
    state: ResponseRequestState,
) -> None:
    if not state.store:
        return
    if not response_state_store.enabled:
        raise RuntimeError("store=true requires RESPONSES_REDIS_URL.")
    await response_state_store.put(_state_payload(response_object, req_data, state))
    if state.conversation_id:
        await response_state_store.put_conversation(state.conversation_id, state.response_id)


def _public_response_state(response: dict[str, Any]) -> dict[str, Any]:
    """Remove proxy-only state before returning a stored response to a client."""
    public_response = copy.deepcopy(response)
    public_response.pop("_normalized_messages", None)
    public_response.pop("_tool_registry", None)
    for key in ("_backend_route", "_backend_model", "_contains_images"):
        public_response.pop(key, None)
    return public_response


def _sse_event(event_type: str, payload: dict[str, Any]) -> bytes:
    event = {"type": event_type, **payload}
    # The Responses API uses the JSON type field, without a named SSE event.
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def _sse_done() -> bytes:
    """Encode the Responses success-stream terminator."""
    return b"data: [DONE]\n\n"


def _response_in_progress_stub(state: ResponseRequestState) -> dict[str, Any]:
    return {
        "id": state.response_id,
        "object": "response",
        "created_at": state.created_at,
        "status": "in_progress",
        "model": state.model,
        "output": [],
    }


def _responses_incomplete_event(
    state: ResponseRequestState,
    code: Any,
    *,
    response: dict[str, Any] | None = None,
) -> bytes:
    """Build the single terminal Responses event for every incomplete path."""
    reason = code.strip() if isinstance(code, str) else ""
    if not reason or reason.lower() == "unknown":
        reason = "decoder_stream_error"
    incomplete_response = dict(response or _response_in_progress_stub(state))
    incomplete_response["status"] = "incomplete"
    incomplete_response["incomplete_details"] = {
        "reason": reason,
        "proxy_error_code": reason,
    }
    return _sse_event("response.incomplete", {"response": incomplete_response})


def _responses_sse_heartbeat_seconds() -> float:
    """Return the configured Responses pre-output heartbeat interval."""
    try:
        return max(0.0, float(getattr(global_args, "responses_sse_heartbeat_seconds", 10.0)))
    except (NameError, TypeError, ValueError):
        return 10.0


async def _responses_stream_failure(
    state: ResponseRequestState,
    *,
    code: str,
    message: str,
):
    """Emit the terminal failure sequence after an SSE response has begun."""
    yield _sse_event("error", {"error": {
        "type": "server_error",
        "code": code,
        "message": message,
    }})
    yield _responses_incomplete_event(state, code)


async def _aclose_async_iterator(iterator) -> None:
    """Close an owned async iterator without masking the caller's terminal path."""
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


async def convert_chat_stream_to_responses_stream(
    body_iterator,
    state: ResponseRequestState,
    normalized_request: dict[str, Any],
    trace: WireTrace | None = None,
    output_accounting: StreamOutputAccounting | None = None,
    input_tokens: int | None = None,
    *,
    emit_created: bool = True,
):
    if emit_created:
        yield _sse_event("response.created", {"response": _response_in_progress_stub(state)})
    item_id = f"msg_{uuid.uuid4().hex}"
    output_kind = None
    output_items = []
    next_output_index = 0
    buffer = ""
    reasoning_id = f"rs_{uuid.uuid4().hex}"
    reasoning = ""
    reasoning_started = False
    reasoning_output_index = None
    reasoning_closed = False
    text = ""
    text_started = False
    text_output_index = None
    tool_started = False
    output_tokens = 0
    usage = {}
    saw_done = False
    saw_finish_reason = False
    finish_reason = None
    stream_error: dict[str, str] | None = None
    dsml_filter_affected = False
    if input_tokens is None:
        # Direct callers do not have the request-level count produced before PD dispatch.
        # The normal Responses path passes that value to avoid delaying first output.
        try:
            _, input_tokens = await count_request_tokens(normalized_request, tokenizer)
        except ResponsesRequestError:
            raise
        except Exception:
            logger.debug("Unable to calculate streaming Responses input usage.", exc_info=True)
            input_tokens = 0

    def allocate_output_index() -> int:
        nonlocal next_output_index
        output_index = next_output_index
        next_output_index += 1
        return output_index

    async def start_text_output():
        nonlocal output_kind, text_started, text_output_index
        if text_started:
            return
        text_started = True
        output_kind = "message" if output_kind is None else "mixed"
        text_output_index = allocate_output_index()
        yield _sse_event("response.output_item.added", {
            "output_index": text_output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        })
        yield _sse_event("response.content_part.added", {
            "item_id": item_id,
            "output_index": text_output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        })

    async def start_reasoning_output():
        nonlocal reasoning_started, reasoning_output_index
        if reasoning_started:
            return
        reasoning_started = True
        reasoning_output_index = allocate_output_index()
        yield _sse_event("response.output_item.added", {
            "output_index": reasoning_output_index,
            "item": {
                "id": reasoning_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        })
        yield _sse_event("response.reasoning_summary_part.added", {
            "output_index": reasoning_output_index,
            "item": {
                "id": reasoning_id,
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            },
        })

    async def finish_reasoning_output():
        nonlocal reasoning_closed
        if not reasoning_started or reasoning_closed:
            return
        reasoning_closed = True
        yield _sse_event("response.reasoning_summary_text.done", {
            "item_id": reasoning_id,
            "output_index": reasoning_output_index,
            "summary_index": 0,
            "text": reasoning,
        })
        yield _sse_event("response.reasoning_summary_part.done", {
            "output_index": reasoning_output_index,
            "item": {
                "id": reasoning_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            },
        })
        yield _sse_event("response.output_item.done", {
            "output_index": reasoning_output_index,
            "item": {
                "id": reasoning_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            },
        })

    async def handle_sse_event(event: str):
        nonlocal text, reasoning, output_tokens, usage, output_kind, tool_started
        nonlocal saw_done, saw_finish_reason, finish_reason, stream_error
        nonlocal dsml_filter_affected
        data = "\n".join(
            line.lstrip()[5:].strip()
            for line in event.splitlines()
            if line.lstrip().startswith("data:")
        )
        if not data:
            return
        if data == "[DONE]":
            saw_done = True
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            stream_error = {
                "type": "server_error",
                "code": "decoder_stream_error",
                "message": "Invalid SSE payload received from decoder.",
            }
            return
        if trace:
            trace.write("response.events.jsonl", payload)
        if payload.get("error"):
            error = payload["error"]
            stream_error = {
                "type": error.get("type", "server_error"),
                "code": error.get("code", "decoder_stream_error"),
                "message": error.get("message", "Decoder stream error"),
            }
            return
        choices = payload.get("choices") or []
        choice = choices[0] if choices else {}
        delta = choice.get("delta") or {}
        reasoning_delta = _extract_reasoning_text(delta)
        if reasoning_delta:
            async for started in start_reasoning_output():
                yield started
            reasoning += reasoning_delta
            yield _sse_event("response.reasoning_summary_text.delta", {
                "item_id": reasoning_id,
                "output_index": reasoning_output_index,
                "summary_index": 0,
                "delta": reasoning_delta,
            })
        tool_deltas = delta.get("tool_calls") or []
        if tool_deltas:
            async for finished in finish_reasoning_output():
                yield finished
            tool_started = True
            output_kind = "function_call" if output_kind is None else "mixed"
            for tool_delta in tool_deltas:
                if not isinstance(tool_delta, dict):
                    continue
                index = tool_delta.get("index", 0)
                while len(output_items) <= index:
                    output_items.append({
                        "id": f"fc_{uuid.uuid4().hex}",
                        "call_id": None,
                        "name": None,
                        "backend_name": None,
                        "arguments": "",
                        "raw_arguments": "",
                    })
                item = output_items[index]
                function = tool_delta.get("function") or {}
                backend_name = function.get("name")
                (
                    tool_spec,
                    is_functions_exec_alias,
                    is_codex_custom_containment,
                    is_custom_functions_exec_suffix_alias,
                    is_codex_custom_passthrough,
                    is_wait_alias,
                ) = _resolve_decoder_tool_spec(backend_name, state)
                tool_spec = tool_spec or {}
                source_type = tool_spec.get("source_type", "function")
                source_name = tool_spec.get("source_name", backend_name)
                if tool_delta.get("id"):
                    item["call_id"] = tool_delta["id"]
                if backend_name:
                    item["backend_name"] = backend_name
                    item["name"] = source_name
                    item["source_type"] = source_type
                    if tool_spec.get("namespace"):
                        item["namespace"] = tool_spec["namespace"]
                    item["deferred_nested_tool"] = (
                        backend_name in (
                            CODEX_NESTED_EXEC_TOOL_NAMES
                            | CODEX_DEFERRED_NESTED_EXEC_ALIASES
                        )
                        and not tool_spec
                    )
                    item["deferred_functions_exec_alias"] = (
                        is_functions_exec_alias
                        or _exec_compatibility_kind(backend_name, state) is not None
                    )
                    item["deferred_wait_alias"] = is_wait_alias
                    item["deferred_unknown_tool"] = (
                        not tool_spec and not item["deferred_nested_tool"]
                    )
                    if (
                        item.get("output_index") is None
                        and (item["deferred_nested_tool"] or item["deferred_functions_exec_alias"]
                             or item["deferred_wait_alias"] or item["deferred_unknown_tool"])
                    ):
                        item["output_index"] = allocate_output_index()
                if (
                    not item.get("call_id")
                    and not item.get("deferred_nested_tool")
                    and not item.get("deferred_functions_exec_alias")
                    and not item.get("deferred_wait_alias")
                    and not item.get("deferred_unknown_tool")
                ):
                    item["call_id"] = f"call_{uuid.uuid4().hex}"
                if item.get("name") is None:
                    continue
                if (
                    not item.get("started")
                    and not item.get("deferred_nested_tool")
                    and not item.get("deferred_functions_exec_alias")
                    and not item.get("deferred_wait_alias")
                    and not item.get("deferred_unknown_tool")
                ):
                    item["started"] = True
                    item["output_index"] = allocate_output_index()
                    output_item_type = {
                        "custom": "custom_tool_call",
                        "web_search": "web_search_call",
                        "tool_search": "tool_search_call",
                    }.get(source_type, "function_call")
                    if source_type == "tool_search":
                        added_item = {
                            "id": item["id"],
                            "type": output_item_type,
                            "status": "in_progress",
                            "execution": "client",
                            "call_id": item["call_id"],
                            "arguments": {},
                        }
                    else:
                        added_item = {
                            "id": item["id"],
                            "type": output_item_type,
                            "status": "in_progress",
                            "call_id": item["call_id"],
                            "name": item["name"],
                            **({"namespace": tool_spec.get("namespace")} if tool_spec.get("namespace") else {}),
                            "input" if source_type == "custom" else "arguments": "",
                        }
                    yield _sse_event("response.output_item.added", {
                        "output_index": item["output_index"],
                        "item": added_item,
                    })
                arguments = function.get("arguments") or ""
                if arguments:
                    item["arguments"] += arguments
                    item["raw_arguments"] += arguments
                    if (
                        item.get("deferred_nested_tool")
                        or item.get("deferred_functions_exec_alias")
                        or item.get("deferred_wait_alias")
                        or item.get("deferred_unknown_tool")
                    ):
                        continue
                    if source_type == "tool_search":
                        continue
                    delta_event = {
                        "custom": "response.custom_tool_call_input.delta",
                        "web_search": "response.web_search_call.delta",
                    }.get(source_type, "response.function_call_arguments.delta")
                    yield _sse_event(delta_event, {
                        "item_id": item["id"],
                        "output_index": item["output_index"],
                        "delta": arguments,
                    })
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
                saw_finish_reason = True
        content, _, usage_completion_tokens, has_output = extract_choice_info(payload)
        if tool_started and content:
            filtered_content = content
            for marker in RESPONSES_DSML_TOOL_TAIL_MARKERS:
                filtered_content = filtered_content.replace(marker, "")
            if filtered_content != content:
                content = filtered_content
                if not dsml_filter_affected:
                    dsml_filter_affected = True
                    stats_collector.record_responses_dsml_filter_affected()
        if content:
            async for finished in finish_reasoning_output():
                yield finished
            async for started in start_text_output():
                yield started
            text += content
            if usage_completion_tokens is not None:
                output_tokens = usage_completion_tokens
            yield _sse_event("response.output_text.delta", {
                "item_id": item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "delta": content,
            })
        if payload.get("usage"):
            usage = payload["usage"]
        if not has_output and usage_completion_tokens is not None:
            output_tokens = usage_completion_tokens
        if choices and choices[0].get("finish_reason"):
            finish_reason = choices[0]["finish_reason"]
            saw_finish_reason = True

    try:
        async for raw_chunk in body_iterator:
            if isinstance(raw_chunk, bytes):
                buffer += raw_chunk.decode("utf-8", errors="replace")
            else:
                buffer += str(raw_chunk)
            buffer = buffer.replace("\r\n", "\n").replace("\r", "\n")
            while "\n\n" in buffer:
                event, buffer = buffer.split("\n\n", 1)
                async for converted in handle_sse_event(event):
                    yield converted
        if buffer.strip():
            async for converted in handle_sse_event(buffer):
                yield converted
    except Exception as exc:
        logger.exception("Decoder stream failed while converting Responses stream")
        stream_error = {
            "type": "server_error",
            "code": "decoder_stream_error",
            "message": "Decoder stream terminated unexpectedly.",
        }
    finally:
        # aclose() injects GeneratorExit at the current yield point. Do not emit
        # terminal SSE events while the ASGI server is closing the iterator.
        if sys.exc_info()[0] is GeneratorExit:
            return
        complete = (
            stream_error is None
            and saw_done
            and saw_finish_reason
            and finish_reason is not None
        )
        if tool_started and complete:
            try:
                for item in output_items:
                    if (
                        not item.get("started")
                        and not item.get("deferred_nested_tool")
                        and not item.get("deferred_functions_exec_alias")
                        and not item.get("deferred_wait_alias")
                        and not item.get("deferred_unknown_tool")
                    ):
                        continue
                    if (
                        item.get("deferred_nested_tool")
                        or item.get("deferred_functions_exec_alias")
                        or item.get("deferred_wait_alias")
                        or item.get("deferred_unknown_tool")
                    ):
                        tool_spec, arguments, nested_tool_wrapped = _normalize_decoder_tool_call(
                            item.get("backend_name"),
                            item.get("arguments", ""),
                            item.get("call_id"),
                            state,
                        )
                        if (
                            not nested_tool_wrapped
                            and not item.get("deferred_functions_exec_alias")
                            and not item.get("deferred_wait_alias")
                        ):
                            raise ResponsesRequestError(
                                "Unable to normalize Codex nested tool call.",
                                "incomplete_function_call",
                            )
                        item["arguments"] = arguments
                        item["name"] = tool_spec["source_name"]
                        item["source_type"] = tool_spec["source_type"]
                        item["namespace"] = tool_spec.get("namespace")
                        item["started"] = True
                    else:
                        if item.get("source_type") == "tool_search":
                            item["arguments"], item["arguments_object"] = _tool_search_arguments_object(
                                item.get("arguments", ""),
                                tool_call_id=item.get("call_id"),
                            )
                        else:
                            item["arguments"] = validate_tool_call_arguments(
                                item.get("arguments", ""),
                                tool_call_id=item.get("call_id"),
                                tool_name=item.get("name"),
                            )
                    if not item.get("call_id") or (
                        not item.get("name") and item.get("source_type") != "tool_search"
                    ):
                        raise ResponsesRequestError(
                            "Function call is missing id or name.",
                            "incomplete_function_call",
                        )
                    if item.get("output_index") is None:
                        item["output_index"] = allocate_output_index()
            except ResponsesRequestError as exc:
                complete = False
                stream_error = {
                    "type": "server_error",
                    "code": exc.code,
                    "message": str(exc),
                }

        if complete:
            async for finished in finish_reasoning_output():
                yield finished
        if complete and text_started:
            yield _sse_event("response.output_text.done", {
                "item_id": item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "text": text,
            })
            yield _sse_event("response.content_part.done", {
                "item_id": item_id,
                "output_index": text_output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            })
            yield _sse_event("response.output_item.done", {
                "output_index": text_output_index,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                },
            })
        if complete and tool_started:
            for index, item in enumerate(output_items):
                if not item.get("started"):
                    continue
                source_type = item.get("source_type", "function")
                output_index = item["output_index"]
                if (item.get("deferred_nested_tool") or item.get("deferred_functions_exec_alias")
                        or item.get("deferred_wait_alias")):
                    deferred_type = {
                        "custom": "custom_tool_call", "web_search": "web_search_call",
                        "tool_search": "tool_search_call",
                    }.get(source_type, "function_call")
                    yield _sse_event("response.output_item.added", {
                        "output_index": output_index,
                        "item": {
                            "id": item["id"],
                            "type": deferred_type,
                            "status": "in_progress",
                            "call_id": item["call_id"],
                            "name": item["name"],
                            **({"namespace": item["namespace"]} if item.get("namespace") else {}),
                            "input" if source_type == "custom" else "arguments": "",
                        },
                    })
                    yield _sse_event(
                        "response.custom_tool_call_input.delta" if source_type == "custom"
                        else "response.function_call_arguments.delta", {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "delta": (_custom_input_from_arguments(item["arguments"])
                                  if source_type == "custom" else item["arguments"]),
                    })
                output_item_type = {
                    "custom": "custom_tool_call",
                    "web_search": "web_search_call",
                    "tool_search": "tool_search_call",
                }.get(source_type, "function_call")
                if source_type != "tool_search":
                    done_event = {
                        "custom": "response.custom_tool_call_input.done",
                        "web_search": "response.web_search_call.done",
                    }.get(source_type, "response.function_call_arguments.done")
                    yield _sse_event(done_event, {
                        "item_id": item["id"],
                        "output_index": output_index,
                        "input" if source_type == "custom" else "arguments": (
                            _custom_input_from_arguments(item["arguments"])
                            if source_type == "custom" else item["arguments"]
                        ),
                    })
                if source_type == "tool_search":
                    completed_item = _tool_search_response_item(
                        item["call_id"], item["arguments"], item_id=item["id"]
                    )
                else:
                    completed_item = {
                        "id": item["id"],
                        "type": output_item_type,
                        "status": "completed",
                        "call_id": item["call_id"],
                        "name": item["name"],
                        **({"namespace": item["namespace"]} if item.get("namespace") else {}),
                        "input" if source_type == "custom" else "arguments": (
                            _custom_input_from_arguments(item["arguments"])
                            if source_type == "custom" else item["arguments"]
                        ),
                    }
                yield _sse_event("response.output_item.done", {
                    "output_index": output_index,
                    "item": completed_item,
                })
        input_tokens = usage.get("prompt_tokens", input_tokens)
        if output_accounting is not None:
            accounted_output_tokens = output_accounting.total_tokens
            if accounted_output_tokens is not None:
                output_tokens = accounted_output_tokens
        else:
            # Direct unit-test callers do not have the PD worker's accounting.
            fallback = StreamOutputAccounting()
            attempt = fallback.start_attempt()
            fallback.observe(
                attempt,
                StreamChoiceInfo(
                    text=text,
                    reasoning=reasoning,
                    tool_arguments={
                        str(index): item.get("raw_arguments", item.get("arguments", ""))
                        for index, item in enumerate(output_items)
                    },
                    backend_completion_tokens=usage.get("completion_tokens"),
                ),
                time.time(),
            )
            fallback.finalize_attempt(attempt)
            if fallback.total_tokens is not None:
                output_tokens = fallback.total_tokens
        if not complete:
            if stream_error is None:
                failure_code = "decoder_stream_error"
                failure_message = "Decoder stream ended before a complete response was received."
                if tool_started:
                    failure_code = "incomplete_function_call"
                    failure_message = "Decoder ended before function call arguments were complete."
                stream_error = {
                    "type": "server_error",
                    "code": failure_code,
                    "message": failure_message,
                }
            yield _sse_event("error", {"error": stream_error})
            yield _responses_incomplete_event(
                state,
                stream_error.get("code"),
                response={
                    "id": state.response_id,
                    "object": "response",
                    "created_at": state.created_at,
                    "model": state.model,
                    "output": [],
                },
            )
            return

        final_output_items = []
        if reasoning_started:
            final_output_items.append((reasoning_output_index, {
                "id": reasoning_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}],
            }))
        if text_started:
            final_output_items.append((text_output_index, {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }))
        if tool_started:
            for item in output_items:
                if not item.get("started"):
                    continue
                source_type = item.get("source_type", "function")
                if source_type == "custom":
                    final_item = {
                        "id": item["id"],
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "input": _custom_input_from_arguments(item["arguments"]),
                        **({"namespace": item["namespace"]} if item.get("namespace") else {}),
                    }
                elif source_type == "tool_search":
                    final_item = _tool_search_response_item(
                        item["call_id"], item["arguments"], item_id=item["id"]
                    )
                else:
                    final_item = {
                        "id": item["id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": item["call_id"],
                        "name": item["name"],
                        "arguments": item["arguments"],
                        **({"namespace": item["namespace"]} if item.get("namespace") else {}),
                    }
                final_output_items.append((item["output_index"], final_item))
        final_output = [item for _, item in sorted(final_output_items, key=lambda entry: entry[0])]
        final_response = {
            "id": state.response_id,
            "object": "response",
            "created_at": state.created_at,
            "status": "completed",
            "model": state.model,
            "output": final_output,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        if state.store and response_state_store.enabled:
            await _store_response_state(final_response, normalized_request, state)
        yield _sse_event("response.completed", {"response": final_response})
        yield _sse_done()


async def _convert_web_chat_stream_to_responses_stream(
    body_iterator,
    chat_request: dict[str, Any],
    request: Request,
    state: ResponseRequestState,
    trace: WireTrace,
    *,
    emit_created: bool = True,
    backend_runner: "ResponsesBackendRunner | None" = None,
):
    """Complete proxy-owned web tool rounds and emit their Responses events."""
    try:
        chat_response = await _collect_chat_stream_response(body_iterator)
        chat_response = await _run_proxy_web_tool_rounds(
            chat_response, chat_request, request, state, trace, backend_runner
        )
        final = chat_result_to_response_object(chat_response, chat_request, state)
        if state.store:
            try:
                await _store_response_state(final, chat_request, state)
            except Exception as exc:
                raise ResponsesBackendError(
                    "Unable to store response state.", "response_state_unavailable", 503
                ) from exc
        if trace:
            trace.write_json("response.final.json", final)
    except Exception as exc:
        logger.exception("Web tool continuation failed response_id=%s", state.response_id)
        message, code, _ = _web_round_failure(exc)
        async for event in _responses_stream_failure(
            state,
            code=code,
            message=message,
        ):
            yield event
        return

    if emit_created:
        yield _sse_event("response.created", {
            "response": {**final, "status": "in_progress", "output": []},
        })
    for output_index, item in enumerate(final.get("output", [])):
        if item.get("type") == "message":
            item_id = item.get("id", f"msg_{uuid.uuid4().hex}")
            content = item.get("content") or []
            text_value = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            yield _sse_event("response.output_item.added", {
                "output_index": output_index,
                "item": {**item, "status": "in_progress", "content": []},
            })
            yield _sse_event("response.content_part.added", {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            })
            if text_value:
                yield _sse_event("response.output_text.delta", {
                    "item_id": item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text_value,
                })
            yield _sse_event("response.output_text.done", {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text_value,
            })
            yield _sse_event("response.content_part.done", {
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text_value, "annotations": []},
            })
            yield _sse_event("response.output_item.done", {
                "output_index": output_index,
                "item": item,
            })
        else:
            yield _sse_event("response.output_item.added", {
                "output_index": output_index,
                "item": {**item, "status": "in_progress"},
            })
            yield _sse_event("response.output_item.done", {
                "output_index": output_index,
                "item": item,
            })
    yield _sse_event("response.completed", {"response": final})
    yield _sse_done()


async def _run_responses_backend(
    request: Request,
    chat_request: dict[str, Any],
    state: ResponseRequestState,
    trace: WireTrace,
    raw_body: bytes,
):
    """Run the internal Chat request with Responses tracing in this task's context."""
    trace_token = _active_wire_trace.set(trace)
    source_token = _active_failure_capture_source.set(
        FailureCaptureSource(
            client_request_body=raw_body,
            api="/responses",
            response_id=state.response_id,
        )
    )
    try:
        synthetic = _synthetic_request(request, json.dumps(chat_request, ensure_ascii=False).encode())
        return await _handle_completions(
            "/chat/completions",
            synthetic,
            log_api="/responses",
            emit_done_on_stream_error=False,
            response_state=state,
        )
    finally:
        _active_wire_trace.reset(trace_token)
        _active_failure_capture_source.reset(source_token)


ResponsesBackendRunner = Callable[
    [Request, dict[str, Any], ResponseRequestState, WireTrace, bytes],
    Awaitable[Any],
]


async def _responses_stream_lifecycle(
    request: Request,
    chat_request: dict[str, Any],
    state: ResponseRequestState,
    trace: WireTrace,
    raw_body: bytes,
    backend_runner: ResponsesBackendRunner | None = None,
):
    """Start Responses SSE before the internal Prefill request can block."""
    first_sse_at = time.time()
    heartbeat_interval = _responses_sse_heartbeat_seconds()
    heartbeat_total = 0
    heartbeat_prefill = 0
    heartbeat_decoder_first_sse = 0
    backend_ready_at = None
    first_decoder_sse_at = None
    logger.info(
        "responses_stream_created response_id=%s first_sse_at=%.6f",
        state.response_id,
        first_sse_at,
    )
    yield _sse_event("response.created", {"response": _response_in_progress_stub(state)})
    # Match the standard Responses stream: created is immediately followed by
    # a state transition before any Prefill or Decoder wait can begin.
    yield _sse_event("response.in_progress", {"response": _response_in_progress_stub(state)})

    backend_task = None
    first_event_task = None
    iterator = None
    backend_body_iterator = None
    decoder_stream_started = False
    try:
        runner = backend_runner or _run_responses_backend
        backend_task = asyncio.create_task(
            runner(request, chat_request, state, trace, raw_body)
        )
        while not backend_task.done():
            if heartbeat_interval <= 0:
                await backend_task
                break
            done, _ = await asyncio.wait({backend_task}, timeout=heartbeat_interval)
            if not done:
                heartbeat_total += 1
                heartbeat_prefill += 1
                stats_collector.record_responses_sse_heartbeat("prefill")
                yield _sse_event(
                    "response.in_progress",
                    {"response": _response_in_progress_stub(state)},
                )
        result = backend_task.result()
        backend_ready_at = time.time()
        logger.info(
            "responses_stream_backend_ready response_id=%s prefill_after_first_sse=%.3fs",
            state.response_id,
            backend_ready_at - first_sse_at,
        )
        if isinstance(result, JSONResponse):
            try:
                payload = json.loads(result.body)
            except (TypeError, json.JSONDecodeError):
                payload = {}
            error = payload.get("error") if isinstance(payload.get("error"), dict) else payload
            code = error.get("code", "backend_error") if isinstance(error, dict) else "backend_error"
            message = (
                error.get("message", "Backend request failed")
                if isinstance(error, dict) else "Backend request failed"
            )
            logger.warning(
                "responses_stream_initialization_failed response_id=%s code=%s status=%s",
                state.response_id,
                code,
                result.status_code,
            )
            async for event in _responses_stream_failure(state, code=code, message=message):
                yield event
            return
        if not isinstance(result, StreamingResponse):
            logger.error(
                "responses_stream_initialization_failed response_id=%s code=invalid_stream_response",
                state.response_id,
            )
            async for event in _responses_stream_failure(
                state,
                code="invalid_stream_response",
                message="Proxy did not create a streaming backend response.",
            ):
                yield event
            return

        backend_body_iterator = result.body_iterator
        if any(spec.get("source_type") == "web_search" for spec in state.tool_registry.values()):
            iterator = _convert_web_chat_stream_to_responses_stream(
                backend_body_iterator,
                chat_request,
                request,
                state,
                trace,
                emit_created=False,
                backend_runner=runner,
            )
        else:
            iterator = convert_chat_stream_to_responses_stream(
                backend_body_iterator,
                state,
                chat_request,
                trace,
                getattr(result, "output_accounting", None),
                getattr(result, "input_tokens", None),
                emit_created=False,
            )
        decoder_stream_started = True

        # Keep exactly one pending read alive while waiting for the first real
        # converted event. Heartbeats must not cancel or restart that read.
        first_event_task = asyncio.create_task(anext(iterator))
        while not first_event_task.done():
            if heartbeat_interval <= 0:
                await first_event_task
                break
            done, _ = await asyncio.wait({first_event_task}, timeout=heartbeat_interval)
            if not done:
                heartbeat_total += 1
                heartbeat_decoder_first_sse += 1
                stats_collector.record_responses_sse_heartbeat("decoder_first_sse")
                yield _sse_event(
                    "response.in_progress",
                    {"response": _response_in_progress_stub(state)},
                )
        try:
            first_event = first_event_task.result()
        except StopAsyncIteration:
            return
        first_event_task = None
        first_decoder_sse_at = time.time()
        logger.info(
            "responses_stream_first_decoder_sse response_id=%s first_decoder_sse=%.3fs",
            state.response_id,
            first_decoder_sse_at - first_sse_at,
        )
        yield first_event
        async for event in iterator:
            yield event
    except asyncio.CancelledError:
        leak_state = state.pipeline_leak_state
        if (
            leak_state is not None
            and leak_state.stream
            and leak_state.pipeline_phase == "prefill_pending"
        ):
            # The internal task must observe cancellation so its shielded
            # Prefill handoff can create the controlled continuation.
            mark_client_disconnected(leak_state)
            logger.info(
                "responses_stream_prefill_disconnect_handoff response_id=%s request_id=%s",
                state.response_id,
                leak_state.request_id or "unknown",
            )
        elif leak_state is not None and leak_state.pipeline_phase == "decoder_running":
            mark_client_disconnected(leak_state)
            logger.info(
                "responses_stream_decoder_disconnect_close response_id=%s request_id=%s",
                state.response_id,
                leak_state.request_id or "unknown",
            )
        if backend_task is not None and not backend_task.done():
            backend_task.cancel()
            with suppress(asyncio.CancelledError):
                await backend_task
        if not decoder_stream_started:
            logger.info(
                "responses_stream_client_disconnected_before_decoder response_id=%s",
                state.response_id,
            )
        raise
    except Exception as exc:
        logger.exception(
            "responses_stream_initialization_failed response_id=%s code=proxy_internal_error",
            state.response_id,
        )
        async for event in _responses_stream_failure(
            state,
            code=exc.code if isinstance(exc, ResponsesRequestError) else "proxy_internal_error",
            message=str(exc),
        ):
            yield event
    finally:
        if first_event_task is not None and not first_event_task.done():
            first_event_task.cancel()
            with suppress(asyncio.CancelledError):
                await first_event_task
        if iterator is not None:
            with suppress(asyncio.CancelledError, Exception):
                await _aclose_async_iterator(iterator)
        # Closing the converter alone does not close the nested Chat iterator.
        # That iterator owns consume_stream(), which cancels the Decoder worker.
        if backend_body_iterator is not None:
            with suppress(asyncio.CancelledError, Exception):
                await _aclose_async_iterator(backend_body_iterator)
        if backend_task is not None and not backend_task.done():
            backend_task.cancel()
            with suppress(asyncio.CancelledError):
                await backend_task
        logger.info(
            "responses_stream_lifecycle_finished response_id=%s heartbeats=%d "
            "heartbeats_prefill=%d heartbeats_decoder_first_sse=%d first_sse_at=%s "
            "backend_ready_at=%s first_decoder_sse_at=%s",
            state.response_id,
            heartbeat_total,
            heartbeat_prefill,
            heartbeat_decoder_first_sse,
            f"{first_sse_at:.6f}",
            f"{backend_ready_at:.6f}" if backend_ready_at is not None else "-",
            f"{first_decoder_sse_at:.6f}" if first_decoder_sse_at is not None else "-",
        )


async def handle_responses(
    request: Request,
    backend_runner: ResponsesBackendRunner | None = None,
):
    raw_body = b""
    audit = _active_request_audit.get()
    try:
        raw_body = await request.body()
        if audit is not None:
            audit.capture_body(raw_body)
        response_request = json.loads(raw_body)
        if audit is not None:
            audit.capture_request_json(response_request)
    except json.JSONDecodeError as exc:
        return responses_error(f"Invalid JSON request: {exc}", "invalid_json")

    try:
        chat_request, state = await responses_request_to_chat_request(response_request)
        if audit is not None:
            audit.capture_request_json(chat_request, normalized=True)
    except UnsupportedResponsesInputItemError as exc:
        diagnostic_id = record_rejected_responses_input(exc, raw_body)
        response = responses_error(str(exc), exc.code, exc.status_code)
        response.headers["X-Proxy-Diagnostic-Id"] = diagnostic_id
        return response
    except ResponsesRequestError as exc:
        logger.warning(
            "responses_request_rejected code=%s status=%s body_bytes=%d body_sha256=%s",
            exc.code,
            exc.status_code,
            len(raw_body),
            hashlib.sha256(raw_body).hexdigest(),
        )
        return responses_error(str(exc), exc.code, exc.status_code)
    except TypeError:
        logger.exception(
            "responses_request_type_error body_bytes=%d body_sha256=%s",
            len(raw_body),
            hashlib.sha256(raw_body).hexdigest(),
        )
        return responses_error("Invalid request field type.", "invalid_request")
    except Exception as exc:
        logger.exception("Failed to prepare Responses request")
        return responses_error(str(exc), "response_state_unavailable", 503)

    trace = WireTrace(state.response_id)
    trace.write_json("request.raw.json", response_request)
    trace.write_json("request.normalized.json", _redact_multimodal(chat_request))
    trace.write_json("routing.decision.json", {
        "reasoning": state.reasoning_settings, "backend_route": "pd",
        "contains_images": state.contains_images,
    })
    if state.store and not response_state_store.enabled:
        return responses_error(
            "store=true requires RESPONSES_REDIS_URL.",
            "response_state_unavailable",
            503,
        )
    if bool(chat_request.get("stream", False)):
        return StreamingResponse(
            _responses_stream_lifecycle(
                request,
                chat_request,
                state,
                trace,
                raw_body,
                backend_runner=backend_runner,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        runner = backend_runner or _run_responses_backend
        result = await runner(request, chat_request, state, trace, raw_body)
    except Exception as exc:
        return responses_error(str(exc), "proxy_error", 500)

    if isinstance(result, JSONResponse):
        try:
            chat_response = json.loads(result.body)
        except (TypeError, json.JSONDecodeError):
            return responses_error("Invalid decoder response.", "invalid_decoder_response", 502)
        if result.status_code >= 400:
            error = chat_response.get("error") if isinstance(chat_response.get("error"), dict) else chat_response
            return responses_error(
                error.get("message", "Backend request failed"),
                error.get("code", "backend_error"),
                result.status_code,
            )
        if any(spec.get("source_type") == "web_search" for spec in state.tool_registry.values()):
            try:
                chat_response = await _run_proxy_web_tool_rounds(
                    chat_response, chat_request, request, state, trace, backend_runner
                )
            except Exception as exc:
                logger.exception("Web tool continuation failed")
                message, code, status_code = _web_round_failure(exc)
                return responses_error(message, code, status_code)
        try:
            response_object = chat_result_to_response_object(chat_response, chat_request, state)
        except ResponsesRequestError as exc:
            logger.warning(
                "decoder_function_arguments_invalid response_id=%s code=%s",
                state.response_id,
                exc.code,
            )
            return responses_error(
                "Decoder returned invalid function call arguments.",
                exc.code,
                502,
            )
        trace.write_json("response.final.json", response_object)
        if state.store:
            try:
                await _store_response_state(response_object, chat_request, state)
            except Exception as exc:
                return responses_error("Unable to store response state.", "response_state_unavailable", 503)
        return JSONResponse(content=response_object)

    return responses_error("Unexpected proxy response type.", "proxy_error", 500)


app = FastAPI(lifespan=lifespan)


def is_backend_client_error(exc: Exception) -> bool:
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response is not None
        and 400 <= exc.response.status_code < 500
    )


def classify_decoder_failure(exc: Exception) -> str:
    if isinstance(exc, (ChatEmptyCompletionError, PdFirstOutputTimeoutError,
                        DecoderStreamIdleTimeoutError, DecoderStartProtocolError,
                        DecoderStreamIncompleteError)):
        return "stream_incomplete" if isinstance(exc, DecoderStreamIncompleteError) else "request_protocol"
    if isinstance(exc, asyncio.CancelledError):
        return "client_disconnect"
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        if 500 <= exc.response.status_code <= 599:
            return "http_5xx"
        if 400 <= exc.response.status_code <= 499:
            return "request_protocol"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout,
                        httpx.WriteError, httpx.ReadError, httpx.RemoteProtocolError)):
        return "transport"
    return "unknown"


def should_taint_decoder_for_error(exc: Exception, server: ServerState | None = None) -> tuple[str, int, bool]:
    category = classify_decoder_failure(exc)
    if server is None:
        return category not in {"request_protocol", "stream_incomplete", "client_disconnect"}
    proxy_state.record_decoder_failure_observation(server, category)
    taint_enabled = getattr(globals().get("global_args"), "decoder_taint_enabled", DECODER_TAINT_ENABLED)
    if not taint_enabled or category in {"request_protocol", "stream_incomplete", "client_disconnect"}:
        proxy_state.record_decoder_taint_suppressed(server, category)
        return category, 0, False
    count, threshold_reached = proxy_state.record_decoder_failure(server, category)
    if not threshold_reached:
        proxy_state.record_decoder_taint_suppressed(server, category)
    # A ConnectError means the endpoint could not establish a TCP connection
    # (including connection refused); isolate it immediately. Timeouts and
    # mid-stream resets remain subject to the failure window.
    immediate_transport_failure = isinstance(exc, httpx.ConnectError)
    return category, count, immediate_transport_failure or threshold_reached


def backend_error_summary(exc: httpx.HTTPStatusError, max_body_chars: int = 200) -> str:
    response = exc.response
    status_code = response.status_code
    url = response.request.url if response.request is not None else response.url
    body = _safe_response_text(response, max_body_chars=max_body_chars)
    if body:
        return f"status={status_code} url={url} body={body}"
    return f"status={status_code} url={url}"


def backend_error_response(exc: httpx.HTTPStatusError) -> JSONResponse:
    status_code = exc.response.status_code
    try:
        content = exc.response.json()
    except ValueError:
        content = {
            "object": "error",
            "message": _safe_response_text(exc.response) or str(exc),
            "type": "backend_client_error",
            "code": f"backend_http_{status_code}",
        }
    if isinstance(content, dict) and "error" not in content:
        content = {"error": content}
    return JSONResponse(status_code=status_code, content=content)


def backend_error_category(exc: Exception) -> str:
    if isinstance(exc, ResponsesRequestError):
        return exc.code
    if isinstance(exc, PdFirstOutputTimeoutError):
        return "pd_first_output_timeout"
    if isinstance(exc, DecoderStreamIdleTimeoutError):
        return "decoder_stream_idle_timeout"
    if isinstance(exc, DecoderStartProtocolError):
        return "decoder_start_protocol_recompute_exhausted"
    if isinstance(exc, DecoderStreamIncompleteError):
        return "decoder_stream_incomplete"
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else 0
        if 400 <= status_code < 500:
            return "upstream_4xx"
        if 500 <= status_code < 600:
            return "upstream_5xx"
        return "upstream_http_error"
    if isinstance(exc, httpx.RequestError):
        return "upstream_request_error"
    return "proxy_error"


def _safe_response_text(response: httpx.Response, max_body_chars: int = 200) -> str:
    try:
        if getattr(response, "is_stream_consumed", False) is False and not getattr(response, "is_closed", True):
            return ""
        body = response.text.replace("\n", " ").strip()
        if len(body) > max_body_chars:
            body = f"{body[:max_body_chars]}..."
        return body
    except Exception:
        return ""


def backend_error_status(exc: Exception) -> int | None:
    if isinstance(exc, ResponsesRequestError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        return exc.response.status_code
    return None


def summarize_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return backend_error_summary(exc)
    return str(exc)


def describe_exception(exc: Exception) -> str:
    exc_type = type(exc).__name__
    status_code = backend_error_status(exc)
    summary = summarize_error(exc).strip()
    parts = [f"type={exc_type}", f"status={status_code}" if status_code is not None else "status=None"]
    if summary:
        parts.append(f"detail={summary}")
    return " ".join(parts)


def trigger_prefiller_abort_cleanup(server: ServerState, req_data: dict, request_id: str, reason: str) -> None:
    logger.info(
        "Abort cleanup disabled for prefiller %s request %s reason=%s",
        server.url,
        request_id,
        reason,
    )
    stats_collector.record_disabled_abort_cleanup()


def _serialize_json_payload(payload: dict) -> tuple[bytes, str]:
    body = json.dumps(
        copy.deepcopy(payload),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json.loads(body)
    return body, hashlib.sha256(body).hexdigest()


def _prefill_request_headers(request_id: str, body: bytes) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(body)),
        "X-Request-Id": request_id,
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    if PREFILL_CONNECTION_MODE == "close":
        headers["Connection"] = "close"
    trace = _active_wire_trace.get()
    if trace:
        headers["X-Response-Id"] = trace.response_id
    return headers


async def send_request_to_service(
    client: httpx.AsyncClient,
    prefiller: ServerState,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    prefill_req_data = copy.deepcopy(prepare_prefill_request_data(req_data))
    # Prefiller only supports /chat/completions endpoint.
    # Convert completions-style requests (with "prompt") to chat format.
    endpoint = "/chat/completions"
    prefill_req_data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": [],
    }
    prefill_req_data["stream"] = False
    prefill_req_data["max_tokens"] = 1
    prefill_req_data["min_tokens"] = 1
    if "max_completion_tokens" in prefill_req_data:
        prefill_req_data["max_completion_tokens"] = 1
    if "stream_options" in prefill_req_data:
        del prefill_req_data["stream_options"]
    body, body_hash = _serialize_json_payload(prefill_req_data)
    stats_collector.record_prefill_wire(len(body))
    trace = _active_wire_trace.get()
    if trace:
        trace.write_json("prefill.request.json", prefill_req_data)
        trace.write_bytes("prefill.request.body", body)
    client_request_id = _active_client_request_id.get()
    last_exc = None
    for attempt in range(1, max_retries + 1):
        request_started = time.time()
        headers = _prefill_request_headers(request_id, body)
        if trace:
            trace.write("prefill.request.meta.jsonl", {
                "request_id": request_id,
                "client_request_id": client_request_id,
                "response_id": trace.response_id,
                "endpoint": endpoint,
                "node": str(prefiller),
                "content_length": len(body),
                "body_sha256": body_hash,
                "attempt": attempt,
                "connection_mode": PREFILL_CONNECTION_MODE,
                "started_at": request_started,
            })
        try:
            response = await client.post(endpoint, content=body, headers=headers)
            response.raise_for_status()
            if trace:
                trace.write("prefill.request.meta.jsonl", {
                    "request_id": request_id,
                    "client_request_id": client_request_id,
                    "response_id": trace.response_id,
                    "endpoint": endpoint,
                    "node": str(prefiller),
                    "content_length": len(body),
                    "body_sha256": body_hash,
                    "attempt": attempt,
                    "connection_mode": PREFILL_CONNECTION_MODE,
                    "status_code": response.status_code,
                    "duration_seconds": round(time.time() - request_started, 6),
                })
            if trace:
                try:
                    trace.write_json("prefill.response.json", response.json())
                except ValueError:
                    trace.write("prefill.response.jsonl", response.text)
            return response
        except httpx.HTTPStatusError as e:
            response_body = _safe_response_text(e.response, max_body_chars=500)
            failure_kind = _failure_kind_for_http_status(
                e.response.status_code,
                e.response.content,
            )
            _write_failure_capture(
                request_id=request_id,
                prefiller=prefiller,
                endpoint=endpoint,
                attempt=attempt,
                headers=headers,
                body=body,
                body_hash=body_hash,
                prefill_request=prefill_req_data,
                request_started=request_started,
                failure_kind=failure_kind,
                exc=e,
                response=e.response,
            )
            if failure_kind == "prefill_invalid_json":
                stats_collector.record_prefill_invalid_json()
                logger.error(
                    "prefill_invalid_json request_id=%s node=%s status=400 "
                    "body_bytes=%s body_sha256=%s attempt=%s duration=%.3fs response=%s",
                    request_id,
                    prefiller,
                    len(body),
                    body_hash,
                    attempt,
                    time.time() - request_started,
                    response_body,
                )
            if trace:
                trace.write("prefill.request.meta.jsonl", {
                    "request_id": request_id,
                    "client_request_id": client_request_id,
                    "response_id": trace.response_id if trace else None,
                    "endpoint": endpoint,
                    "node": str(prefiller),
                    "content_length": len(body),
                    "body_sha256": body_hash,
                    "attempt": attempt,
                    "connection_mode": PREFILL_CONNECTION_MODE,
                    "status_code": e.response.status_code,
                    "duration_seconds": round(time.time() - request_started, 6),
                    "response": response_body,
                })
            if is_backend_client_error(e):
                logger.info("Backend client error for %s, not retrying: %s",
                            endpoint, backend_error_summary(e))
                raise
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc
        except httpx.RequestError as e:
            stats_collector.record_prefill_transport_error()
            _write_failure_capture(
                request_id=request_id,
                prefiller=prefiller,
                endpoint=endpoint,
                attempt=attempt,
                headers=headers,
                body=body,
                body_hash=body_hash,
                prefill_request=prefill_req_data,
                request_started=request_started,
                failure_kind=_failure_kind_for_transport_error(e),
                exc=e,
            )
            if trace:
                trace.write("prefill.request.meta.jsonl", {
                    "request_id": request_id,
                    "response_id": trace.response_id if trace else None,
                    "endpoint": endpoint,
                    "node": str(prefiller),
                    "content_length": len(body),
                    "body_sha256": body_hash,
                    "attempt": attempt,
                    "connection_mode": PREFILL_CONNECTION_MODE,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "duration_seconds": round(time.time() - request_started, 6),
                })
            logger.warning(f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                stats_collector.record_prefill_retry()
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for {endpoint}.")
                raise last_exc


async def stream_service_response_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    decoder_node: str,
    decoder_attempt: int,
    prefiller_node: str | None = None,
    first_output_deadline_at: float | None = None,
    first_prefill_sent_at: float | None = None,
    stream_progress: DecoderStreamProgress | None = None,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    trace = _active_wire_trace.get()
    request_body = json.dumps(
        req_data, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    request_body_sha256 = hashlib.sha256(request_body).hexdigest()
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **({"X-Response-Id": trace.response_id} if trace else {}),
    }
    for transport_attempt in range(1, max_retries + 1):
        attempt_id = (
            f"{request_id}:decode-{decoder_attempt}:transport-{transport_attempt}"
        )
        started_at = time.time()
        response_headers_received = False
        first_chunk_received = False
        first_valid_output_recorded = False
        request_submitted_monotonic = None
        valid_output_at_start = (
            stream_progress.last_valid_output_at if stream_progress is not None else None
        )
        if stream_progress is not None:
            stream_progress.first_valid_output_delay = None
        chunk_count = 0
        total_bytes = 0
        watchdog_stop = asyncio.Event()
        observe_seconds = getattr(global_args, "decoder_first_response_observe_seconds", 30.0)
        log_interval_seconds = getattr(
            global_args, "decoder_first_response_log_interval_seconds", 60.0
        )

        record_decoder_attempt_trace(
            trace,
            "started",
            request_id=request_id,
            response_id=trace.response_id if trace else None,
            prefiller_node=prefiller_node,
            decoder_node=decoder_node,
            decoder_attempt=decoder_attempt,
            transport_attempt=transport_attempt,
            endpoint=endpoint,
            request_body_bytes=len(request_body),
            request_body_sha256=request_body_sha256,
        )
        stats_collector.record_decoder_attempt_started()
        stats_collector.start_decoder_first_response_waiting(
            attempt_id,
            decoder_node,
            "waiting_response_headers",
            started_at,
        )

        async def first_response_watchdog() -> None:
            if observe_seconds <= 0:
                return
            delay = observe_seconds
            first_warning = True
            while True:
                try:
                    await asyncio.wait_for(watchdog_stop.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    phase = (
                        "waiting_first_valid_output"
                        if first_chunk_received
                        else "waiting_first_sse"
                        if response_headers_received
                        else "waiting_response_headers"
                    )
                    elapsed = time.time() - started_at
                    if first_warning:
                        stats_collector.record_decoder_first_response_waiting(
                            attempt_id, decoder_node, phase, started_at
                        )
                    record_decoder_attempt_trace(
                        trace,
                        "first_response_waiting",
                        request_id=request_id,
                        response_id=trace.response_id if trace else None,
                        decoder_node=decoder_node,
                        decoder_attempt=decoder_attempt,
                        transport_attempt=transport_attempt,
                        phase=phase,
                        elapsed_seconds=round(elapsed, 6),
                    )
                    logger.warning(
                        "decoder_first_response_waiting request_id=%s response_id=%s "
                        "decoder=%s decoder_attempt=%d transport_attempt=%d phase=%s elapsed_seconds=%.3f",
                        request_id,
                        trace.response_id if trace else None,
                        decoder_node,
                        decoder_attempt,
                        transport_attempt,
                        phase,
                        elapsed,
                    )
                    first_warning = False
                    delay = log_interval_seconds

        watchdog_task = asyncio.create_task(first_response_watchdog())
        try:
            timeout_context = (
                asyncio.timeout_at(first_output_deadline_at)
                if first_output_deadline_at is not None
                else asyncio.timeout(None)
            )
            async with timeout_context as first_output_timeout:
                request_submitted_monotonic = time.monotonic()
                async with client.stream("POST", endpoint, json=req_data, headers=headers) as response:
                    response_headers_received = True
                    record_decoder_attempt_trace(
                        trace,
                        "response_headers_received",
                        request_id=request_id,
                        response_id=trace.response_id if trace else None,
                        decoder_node=decoder_node,
                        decoder_attempt=decoder_attempt,
                        transport_attempt=transport_attempt,
                        status_code=response.status_code,
                        elapsed_seconds=round(time.time() - started_at, 6),
                    )
                    stats_collector.record_decoder_response_headers()
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        chunk_count += 1
                        total_bytes += len(chunk)
                        if trace:
                            trace.write("decode.response.jsonl", {
                                "kind": "chunk",
                                "data": chunk.decode("utf-8", errors="replace"),
                            })
                        if not first_chunk_received:
                            first_chunk_received = True
                            stats_collector.record_decoder_first_sse(attempt_id)
                            stats_collector.set_decoder_first_response_phase(
                                attempt_id, "waiting_first_valid_output"
                            )
                            record_decoder_attempt_trace(
                                trace,
                                "first_sse_chunk_received",
                                request_id=request_id,
                                response_id=trace.response_id if trace else None,
                                decoder_node=decoder_node,
                                decoder_attempt=decoder_attempt,
                                transport_attempt=transport_attempt,
                                elapsed_seconds=round(time.time() - started_at, 6),
                                chunk_bytes=len(chunk),
                            )
                        yield chunk
                        if (
                            stream_progress is not None
                            and stream_progress.last_valid_output_at is not None
                            and stream_progress.last_valid_output_at != valid_output_at_start
                        ):
                            if not first_valid_output_recorded:
                                first_valid_output_recorded = True
                                stream_progress.first_valid_output_delay = max(
                                    0.0,
                                    stream_progress.last_valid_output_at - request_submitted_monotonic,
                                )
                                watchdog_stop.set()
                                stats_collector.record_decoder_first_valid_output(attempt_id)
                                record_decoder_attempt_trace(
                                    trace,
                                    "first_valid_output_received",
                                    request_id=request_id,
                                    response_id=trace.response_id if trace else None,
                                    decoder_node=decoder_node,
                                    decoder_attempt=decoder_attempt,
                                    transport_attempt=transport_attempt,
                                    elapsed_seconds=round(time.time() - started_at, 6),
                                )
                            idle_timeout_seconds = getattr(
                                global_args, "decoder_stream_idle_timeout_seconds", 480.0
                            )
                            first_output_timeout.reschedule(
                                stream_progress.last_valid_output_at + idle_timeout_seconds
                                if idle_timeout_seconds > 0
                                else None
                            )
                    record_decoder_attempt_trace(
                        trace,
                        "stream_eof",
                        request_id=request_id,
                        response_id=trace.response_id if trace else None,
                        decoder_node=decoder_node,
                        decoder_attempt=decoder_attempt,
                        transport_attempt=transport_attempt,
                        chunk_count=chunk_count,
                        total_bytes=total_bytes,
                        elapsed_seconds=round(time.time() - started_at, 6),
                    )
                    return  # Success, exit after streaming
        except TimeoutError:
            watchdog_stop.set()
            elapsed_since_prefill = (
                time.monotonic() - first_prefill_sent_at
                if first_prefill_sent_at is not None
                else None
            )
            if first_valid_output_recorded:
                stats_collector.record_decoder_stream_idle_timeout()
                record_decoder_attempt_trace(
                    trace,
                    "stream_idle_timeout",
                    request_id=request_id,
                    response_id=trace.response_id if trace else None,
                    decoder_node=decoder_node,
                    decoder_attempt=decoder_attempt,
                    transport_attempt=transport_attempt,
                    elapsed_seconds=round(time.time() - started_at, 6),
                )
                raise DecoderStreamIdleTimeoutError(
                    "Decoder stream exceeded the semantic-output idle deadline."
                )
            stats_collector.record_pd_first_valid_output_timeout()
            record_decoder_attempt_trace(
                trace,
                "first_valid_output_timeout",
                request_id=request_id,
                response_id=trace.response_id if trace else None,
                decoder_node=decoder_node,
                decoder_attempt=decoder_attempt,
                transport_attempt=transport_attempt,
                elapsed_seconds=round(time.time() - started_at, 6),
                elapsed_since_prefill_seconds=(
                    round(elapsed_since_prefill, 6)
                    if elapsed_since_prefill is not None
                    else None
                ),
            )
            logger.warning(
                "pd_first_valid_output_timeout request_id=%s response_id=%s decoder=%s "
                "decoder_attempt=%d transport_attempt=%d elapsed_since_prefill_seconds=%s",
                request_id,
                trace.response_id if trace else None,
                decoder_node,
                decoder_attempt,
                transport_attempt,
                f"{elapsed_since_prefill:.3f}" if elapsed_since_prefill is not None else "-",
            )
            raise PdFirstOutputTimeoutError(
                "PD attempt exceeded the first-valid-output deadline."
            )
        except httpx.HTTPStatusError as e:
            watchdog_stop.set()
            stats_collector.finish_decoder_first_response_waiting(attempt_id)
            record_decoder_attempt_trace(
                trace,
                "http_error",
                request_id=request_id,
                response_id=trace.response_id if trace else None,
                decoder_node=decoder_node,
                decoder_attempt=decoder_attempt,
                transport_attempt=transport_attempt,
                status_code=e.response.status_code,
                error_type=type(e).__name__,
                elapsed_seconds=round(time.time() - started_at, 6),
                first_chunk_received=first_chunk_received,
            )
            if is_backend_client_error(e):
                logger.info("Backend client error for streaming %s, not retrying: %s",
                            endpoint, backend_error_summary(e))
                raise
            if transport_attempt < max_retries:
                logger.warning(f"Attempt {transport_attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (transport_attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise
        except httpx.RequestError as e:
            watchdog_stop.set()
            stats_collector.finish_decoder_first_response_waiting(attempt_id)
            stats_collector.record_decode_transport_error()
            record_decoder_attempt_trace(
                trace,
                "transport_error",
                request_id=request_id,
                response_id=trace.response_id if trace else None,
                decoder_node=decoder_node,
                decoder_attempt=decoder_attempt,
                transport_attempt=transport_attempt,
                error_type=type(e).__name__,
                elapsed_seconds=round(time.time() - started_at, 6),
                first_chunk_received=first_chunk_received,
            )
            logger.warning(
                "decode_transport_error request_id=%s endpoint=%s attempt=%d error=%s",
                request_id,
                endpoint,
                transport_attempt,
                str(e),
            )
            if transport_attempt < max_retries:
                logger.warning(f"Attempt {transport_attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (transport_attempt - 1)))
            else:
                logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                raise
        except asyncio.CancelledError:
            watchdog_stop.set()
            stats_collector.finish_decoder_first_response_waiting(attempt_id)
            record_decoder_attempt_trace(
                trace,
                "cancelled",
                request_id=request_id,
                response_id=trace.response_id if trace else None,
                decoder_node=decoder_node,
                decoder_attempt=decoder_attempt,
                transport_attempt=transport_attempt,
                elapsed_seconds=round(time.time() - started_at, 6),
                first_chunk_received=first_chunk_received,
            )
            raise
        except Exception as e:
            watchdog_stop.set()
            stats_collector.finish_decoder_first_response_waiting(attempt_id)
            record_decoder_attempt_trace(
                trace,
                "transport_error",
                request_id=request_id,
                response_id=trace.response_id if trace else None,
                decoder_node=decoder_node,
                decoder_attempt=decoder_attempt,
                transport_attempt=transport_attempt,
                error_type=type(e).__name__,
                elapsed_seconds=round(time.time() - started_at, 6),
                first_chunk_received=first_chunk_received,
            )
            if first_chunk_received:
                logger.error(f"Streaming to client interrupted after response started: {str(e)}")
                raise
            else:
                if transport_attempt < max_retries:
                    logger.warning(f"Attempt {transport_attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (transport_attempt - 1)))
                else:
                    logger.error(f"All {max_retries} attempts failed for streaming {endpoint}.")
                    raise
        finally:
            watchdog_stop.set()
            watchdog_task.cancel()
            try:
                await watchdog_task
            except asyncio.CancelledError:
                pass
            stats_collector.finish_decoder_first_response_waiting(attempt_id)


async def send_decode_request_with_retry(
    client: httpx.AsyncClient,
    endpoint: str,
    req_data: dict,
    request_id: str,
    max_retries: int = 3,
    base_delay: float = 0.2,
):
    trace = _active_wire_trace.get()
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        **({"X-Response-Id": trace.response_id} if trace else {}),
    }
    if trace:
        trace.write_json("decode.request.json", req_data)
    try:
        async with asyncio.timeout(NONSTREAM_DECODER_TIMEOUT_SECONDS):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = await client.post(endpoint, json=req_data, headers=headers)
                    response.raise_for_status()
                    if trace:
                        try:
                            trace.write_json("decode.response.json", response.json())
                        except ValueError:
                            trace.write("decode.response.jsonl", response.text)
                    return response
                except httpx.HTTPStatusError as e:
                    if is_backend_client_error(e):
                        logger.info("Backend client error for decode %s, not retrying: %s",
                                    endpoint, backend_error_summary(e))
                        raise
                    logger.warning(f"Attempt {attempt} failed for decode {endpoint}: {str(e)}")
                    last_exc = e
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                    else:
                        logger.error(f"All {max_retries} attempts failed for decode {endpoint}.")
                        raise last_exc
                except httpx.RequestError as e:
                    stats_collector.record_decode_transport_error()
                    logger.warning(
                        "decode_transport_error request_id=%s endpoint=%s attempt=%d error=%s",
                        request_id,
                        endpoint,
                        attempt,
                        str(e),
                    )
                    logger.warning(f"Attempt {attempt} failed for decode {endpoint}: {str(e)}")
                    last_exc = e
                    if attempt < max_retries:
                        await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                    else:
                        logger.error(f"All {max_retries} attempts failed for decode {endpoint}.")
                        raise last_exc
    except TimeoutError:
        logger.warning(
            "decoder_nonstream_timeout request_id=%s endpoint=%s timeout_seconds=%.0f",
            request_id,
            endpoint,
            NONSTREAM_DECODER_TIMEOUT_SECONDS,
        )
        raise


async def _handle_select_instance(
    api: str,
    req_data: Any,
    prefill_input_tokens: int,
    decoder_input_tokens: int,
    max_output_tokens: int,
    excluded_decoders: set[ServerState] | None = None,
    leak_state: RequestLeakState | None = None,
    allow_prefill_disconnect_drain: bool = False,
):
    prefiller_score = proxy_state.calculate_prefill_scores(prefill_input_tokens)
    logger.debug(f"Prefill input tokens: {prefill_input_tokens}, Prefiller score: {prefiller_score}")
    request_id = await proxy_state.next_req_id()
    # Select prefiller (returns ServerState directly)
    prefiller = proxy_state.select_prefiller(prefiller_score)
    # Send request to prefiller
    first_prefill_sent_at = time.monotonic()
    first_output_timeout_seconds = getattr(
        global_args, "pd_first_output_timeout_seconds", 480.0
    )
    first_output_deadline_at = (
        first_prefill_sent_at + first_output_timeout_seconds
        if first_output_timeout_seconds > 0
        else None
    )
    prefill_start = time.time()
    try:
        prefill_request = asyncio.create_task(send_request_to_service(
            prefiller.prefill_client,
            prefiller,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        ))
        if first_output_deadline_at is None:
            response = await asyncio.shield(prefill_request)
        else:
            remaining_seconds = max(0.0, first_output_deadline_at - time.monotonic())
            try:
                response = await asyncio.wait_for(
                    asyncio.shield(prefill_request), timeout=remaining_seconds
                )
            except asyncio.TimeoutError as exc:
                raise PdFirstOutputTimeoutError(
                    "PD attempt exceeded the first-output deadline while waiting for Prefill."
                ) from exc
    except asyncio.CancelledError:
        if (
            leak_state is not None
            and allow_prefill_disconnect_drain
            and not proxy_shutdown_started
        ):
            leak_state.request_id = request_id
            leak_state.prefiller = str(prefiller)
            mark_client_disconnected(leak_state)
            continuation = asyncio.create_task(
                drain_prefill_after_client_disconnect(
                    prefill_request,
                    prefiller,
                    prefiller_score,
                    api,
                    req_data,
                    request_id,
                    first_output_deadline_at,
                    leak_state,
                )
            )
            detached_task_manager.track(continuation)
            logger.info(
                "client_disconnect_during_prefill request_id=%s prefiller=%s action=continue_prefill",
                request_id,
                prefiller,
            )
            raise
        prefill_request.cancel()
        with suppress(asyncio.CancelledError):
            await prefill_request
        proxy_state.release_prefiller(prefiller, prefiller_score)
        proxy_state.release_prefiller_kv(prefiller, prefiller_score)
        logger.info("Prefill request cancelled before completion node=%s request_id=%s", prefiller, request_id)
        raise
    except Exception as e:
        if "prefill_request" in locals() and not prefill_request.done():
            prefill_request.cancel()
            with suppress(asyncio.CancelledError):
                await prefill_request
        proxy_state.release_prefiller(prefiller, prefiller_score)
        proxy_state.release_prefiller_kv(prefiller, prefiller_score)
        stats_collector.record_error(
            category=backend_error_category(e),
            role=InstanceType.PREFILL,
            node=str(prefiller),
            status_code=backend_error_status(e),
            endpoint="/chat/completions",
            message=summarize_error(e),
        )
        setattr(e, "_proxy_stats_recorded", True)
        if not is_backend_client_error(e):
            proxy_state.taint_prefillers([prefiller])
        raise
    prefill_time = time.time() - prefill_start
    proxy_state.release_prefiller(prefiller, prefiller_score)
    if leak_state is not None:
        leak_state.prefill_completed_at = time.time()
        set_pipeline_phase(leak_state, "decoder_running")
        leak_state.prefiller = str(prefiller)
    try:
        response_json = response.json()
        proxy_state.mark_backend_success(prefiller, "prefill_completed")
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        # Decoder scheduling is least-connections only. Token counts remain
        # relevant to context validation, but must not affect node selection.
        decoder = proxy_state.select_decoder(excluded_decoders=excluded_decoders)
        decoder_score = 1.0
        logger.debug(
            "Decoder selected by active_requests: node=%s active_requests=%s",
            decoder,
            decoder.active_requests,
        )
    except asyncio.CancelledError:
        proxy_state.release_prefiller_kv(prefiller, prefiller_score)
        logger.info("Prefill post-processing cancelled node=%s request_id=%s", prefiller, request_id)
        raise
    except Exception:
        proxy_state.release_prefiller_kv(prefiller, prefiller_score)
        raise
    logger.debug("Using %s %s", prefiller.url, decoder.url)
    return InstanceInfo(
        request_id=request_id,
        prefiller_score=prefiller_score,
        prefiller=prefiller,
        decoder_score=decoder_score,
        decoder=decoder,
        prefill_time=prefill_time,
        first_prefill_sent_at=first_prefill_sent_at,
        first_output_deadline_at=first_output_deadline_at,
    )


def _has_model_output(payload: Any) -> bool:
    if isinstance(payload, str):
        return bool(payload)
    if isinstance(payload, list):
        return len(payload) > 0
    if isinstance(payload, dict):
        return len(payload) > 0
    return payload is not None


def extract_stream_choice_info(chunk_json: dict) -> StreamChoiceInfo:
    """Extract all decoder output classes from one Chat Completions SSE payload."""
    choices = chunk_json.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or {}
    message = choice.get("message") or {}
    text = delta.get("content") or message.get("content") or choice.get("text") or ""
    reasoning = _extract_reasoning_text(delta) or _extract_reasoning_text(message) or ""
    tool_arguments: dict[str, str] = {}
    tool_calls = delta.get("tool_calls") or message.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str) or not arguments:
                continue
            key = str(tool_call.get("id") or tool_call.get("index") or position)
            tool_arguments[key] = tool_arguments.get(key, "") + arguments
    usage = chunk_json.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 0:
        completion_tokens = None
    return StreamChoiceInfo(
        text=text if isinstance(text, str) else "",
        reasoning=reasoning if isinstance(reasoning, str) else "",
        tool_arguments=tool_arguments,
        backend_completion_tokens=completion_tokens,
        stop_reason=choice.get("stop_reason") or choice.get("finish_reason"),
    )


def extract_choice_info(chunk_json: dict) -> tuple[str, str | None, int | None, bool]:
    info = extract_stream_choice_info(chunk_json)
    # Retained for the non-streaming path and compatibility with existing callers.
    return info.text, info.stop_reason, info.backend_completion_tokens, info.has_output


def get_request_max_tokens(req_data: dict) -> int:
    if "max_tokens" in req_data:
        return req_data["max_tokens"]
    if "max_completion_tokens" in req_data:
        return req_data["max_completion_tokens"]
    return 16


def set_request_max_tokens(req_data: dict, value: int) -> None:
    value = max(0, value)
    if "max_completion_tokens" in req_data and "max_tokens" not in req_data:
        req_data["max_completion_tokens"] = value
        return
    req_data["max_tokens"] = value
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = value


def completion_delta(total_completion_tokens: int, previous_completion_tokens: int, content: str) -> int:
    delta = total_completion_tokens - previous_completion_tokens
    if delta > 0:
        return delta
    return 1 if content else 0


def prepare_retry_request(
    req_data: dict,
    origin_prompt: Any,
    generated_text: str,
    chat_flag: bool,
    base_messages: list[dict] | None = None,
) -> None:
    if chat_flag:
        messages = copy.deepcopy(base_messages if base_messages is not None else req_data.get("messages", []))
        if messages and messages[-1].get("role") == "assistant":
            if _has_multimodal_content(messages[-1].get("content")):
                messages[-1]["content"].append({"type": "text", "text": generated_text})
            else:
                messages[-1]["content"] = generated_text
        else:
            messages.append({"role": "assistant", "content": generated_text})
        req_data["messages"] = messages
    else:
        req_data["prompt"] = origin_prompt + generated_text


@dataclass
class InstanceInfo:
    request_id: str
    prefiller_score: float
    prefiller: ServerState
    decoder_score: float
    decoder: ServerState
    prefill_time: float = 0.0
    decode_time: float = 0.0
    first_prefill_sent_at: float | None = None
    first_output_deadline_at: float | None = None


def build_ttft_detail(
    ttft_total: float | None,
    preprocess_time: float,
    select_time: float,
    prefill_time: float,
) -> str:
    if ttft_total is None:
        return "N/A"
    dispatch_time = max(0.0, select_time - prefill_time)
    decode_wait_time = max(0.0, ttft_total - preprocess_time - select_time)
    return f"{preprocess_time:.2f}s,{dispatch_time:.2f}s,{prefill_time:.2f}s,{decode_wait_time:.2f}s"


def build_ttft_components(
    ttft_total: float | None,
    preprocess_time: float,
    select_time: float,
    prefill_time: float,
) -> tuple[float, float, float, float, float] | None:
    if ttft_total is None:
        return None
    dispatch_time = max(0.0, select_time - prefill_time)
    decode_wait_time = max(0.0, ttft_total - preprocess_time - select_time)
    return ttft_total, preprocess_time, dispatch_time, prefill_time, decode_wait_time


def update_request_leak_targets(leak_state: RequestLeakState, instance_info: InstanceInfo | None) -> None:
    if instance_info is None:
        return
    leak_state.request_id = instance_info.request_id
    leak_state.prefiller = str(instance_info.prefiller)
    leak_state.decoder = str(instance_info.decoder)


def classify_leak_level(reason: str) -> str:
    certain_reasons = {
        "decoder_start_protocol_recompute_exhausted",
        "decoder_error_nonstream",
        "decoder_error_stream",
        "decoder_recomputed_nonstream_failed",
        "decoder_recomputed_stream_failed",
        "prefill_disconnect_drain_failed",
        "decoder_recomputed_context_exhausted",
        "decoder_recomputed_budget_exhausted",
        "decode_attempt_budget_exhausted",
        "proxy_internal_error_after_prefill",
    }
    if reason in certain_reasons:
        return LeakLevel.CERTAIN
    return LeakLevel.SUSPECTED


def mark_request_leak(leak_state: RequestLeakState, reason: str, final_request_state: str) -> None:
    if not leak_state.prefill_committed or leak_state.decoder_finished_normally:
        return
    leak_state.final_request_state = final_request_state
    leak_state.leak_reason = reason
    leak_level = classify_leak_level(reason)
    leak_state.prefill_disconnect_drain_failed = True
    if (
        leak_state.prefill_disconnect_draining
        and leak_state.prefill_disconnect_drain_started_at
        and not leak_state.prefill_disconnect_drain_ended_at
    ):
        leak_state.prefill_disconnect_drain_ended_at = time.time()
    leak_tracker.record(leak_state, leak_level, reason)


def validate_decode_attempt_budget(leak_state: RequestLeakState) -> str | None:
    """Apply retry limits to every request, independent of client disconnects."""
    if leak_state.recomputed_attempts > global_args.max_recomputed_attempts:
        return "decoder_recomputed_budget_exhausted"
    if leak_state.decode_attempts > global_args.max_total_decode_attempts:
        return "decode_attempt_budget_exhausted"
    return None


def record_prefill_disconnect_lifecycle(
    leak_state: RequestLeakState,
    event: str,
    **fields: Any,
) -> None:
    record_decoder_attempt_trace(
        leak_state.trace,
        event,
        request_id=leak_state.request_id or None,
        disconnect_phase=leak_state.disconnect_phase or None,
        pipeline_phase=leak_state.pipeline_phase,
        **fields,
    )


def start_prefill_disconnect_drain(leak_state: RequestLeakState) -> None:
    if leak_state.prefill_disconnect_draining:
        return
    leak_state.prefill_disconnect_draining = True
    leak_state.prefill_disconnect_drain_started_at = time.time()
    stats_collector.record_prefill_disconnect_drain_started(leak_state)
    record_prefill_disconnect_lifecycle(leak_state, "prefill_disconnect_drain_started")
    logger.info(
        "prefill_disconnect_drain_started request_id=%s prefiller=%s disconnect_at=%.6f",
        leak_state.request_id,
        leak_state.prefiller or "unknown",
        leak_state.disconnect_detected_at,
    )


def finish_prefill_disconnect_drain(leak_state: RequestLeakState) -> None:
    if (
        not leak_state.prefill_disconnect_draining
        or leak_state.prefill_disconnect_drain_completed
        or leak_state.prefill_disconnect_drain_failed
    ):
        return
    leak_state.prefill_disconnect_drain_completed = True
    leak_state.prefill_disconnect_drain_ended_at = time.time()
    stats_collector.record_prefill_disconnect_drain_finished(
        leak_state.prefiller or None,
        leak_state.decoder or None,
        leak_state.request_id,
    )
    record_prefill_disconnect_lifecycle(
        leak_state,
        "prefill_disconnect_drain_finished",
        drain_stop_reason=leak_state.drain_stop_reason or None,
    )
    logger.info(
        "prefill_disconnect_drain_finished request_id=%s prefiller=%s decoder=%s elapsed=%.2f",
        leak_state.request_id,
        leak_state.prefiller or "unknown",
        leak_state.decoder or "unknown",
        leak_state.prefill_disconnect_drain_ended_at - leak_state.prefill_disconnect_drain_started_at,
    )


def fail_prefill_disconnect_drain(leak_state: RequestLeakState, reason: str) -> bool:
    """Finalize the Prefill disconnect continuation once; callers own PD cleanup."""
    if (
        not leak_state.prefill_disconnect_draining
        or leak_state.prefill_disconnect_drain_completed
        or leak_state.prefill_disconnect_drain_failed
    ):
        return False
    leak_state.prefill_disconnect_drain_failed = True
    leak_state.prefill_disconnect_drain_ended_at = time.time()
    stats_collector.record_prefill_disconnect_drain_failed(leak_state.request_id)
    record_prefill_disconnect_lifecycle(leak_state, "prefill_disconnect_drain_failed", reason=reason)
    logger.warning(
        "prefill_disconnect_drain_failed request_id=%s prefiller=%s decoder=%s elapsed=%.2f reason=%s",
        leak_state.request_id,
        leak_state.prefiller or "unknown",
        leak_state.decoder or "unknown",
        leak_state.prefill_disconnect_drain_ended_at - leak_state.prefill_disconnect_drain_started_at,
        reason,
    )
    return True


def mark_client_disconnected(leak_state: RequestLeakState) -> None:
    if leak_state.client_disconnected:
        return
    leak_state.client_disconnected = True
    leak_state.disconnect_detected_at = time.time()
    leak_state.disconnect_phase = leak_state.pipeline_phase
    if leak_state.pipeline_phase == "prefill_pending":
        leak_state.drain_after_prefill = True
        start_prefill_disconnect_drain(leak_state)
        record_prefill_disconnect_lifecycle(leak_state, "client_disconnect_during_prefill")
        logger.info(
            "client_disconnect_during_prefill request_id=%s prefiller=%s",
            leak_state.request_id or "unknown",
            leak_state.prefiller or "unknown",
        )
    else:
        record_prefill_disconnect_lifecycle(leak_state, "client_disconnect_during_decoder")
        logger.info(
            "client_disconnect_during_decoder request_id=%s decoder=%s",
            leak_state.request_id or "unknown",
            leak_state.decoder or "unknown",
        )


def set_pipeline_phase(leak_state: RequestLeakState, phase: str) -> None:
    """Record a monotonic request phase for disconnect handling and traces."""
    if leak_state.pipeline_phase == "terminal":
        return
    leak_state.pipeline_phase = phase


def _decoder_sse_has_semantic_output(buffer: str, chunk: bytes) -> tuple[bool, str]:
    """Parse complete SSE events across HTTP chunks and find model output."""
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False, buffer
    buffer += text.replace("\r\n", "\n")
    while "\n\n" in buffer:
        event, buffer = buffer.split("\n\n", 1)
        data_lines = [
            line[len("data:"):].strip()
            for line in event.splitlines()
            if line.strip().startswith("data:")
        ]
        payload = "\n".join(data_lines)
        if not payload or payload == "[DONE]":
            continue
        try:
            info = extract_stream_choice_info(json.loads(payload))
        except (TypeError, ValueError):
            continue
        if info.has_output and not is_decoder_start_protocol_leak(info, False):
            return True, buffer
    return False, buffer


def _decoder_chunk_has_semantic_output(chunk: bytes) -> bool:
    """Single-chunk convenience wrapper used by focused unit tests."""
    found, _ = _decoder_sse_has_semantic_output("", chunk)
    return found


async def drain_prefill_after_client_disconnect(
    prefill_request: asyncio.Task,
    prefiller: ServerState,
    prefiller_score: float,
    api: str,
    req_data: dict,
    request_id: str,
    first_output_deadline_at: float | None,
    leak_state: RequestLeakState,
) -> None:
    """Finish an already-sent Prefill, then close Decoder after its first output."""
    decoder = None
    prefiller_released = False
    prefiller_kv_released = False
    decoder_released = False
    decoder_stream = None
    decoder_sse_buffer = ""
    try:
        if first_output_deadline_at is None:
            response = await prefill_request
        else:
            remaining = max(0.0, first_output_deadline_at - time.monotonic())
            response = await asyncio.wait_for(asyncio.shield(prefill_request), timeout=remaining)
        proxy_state.release_prefiller(prefiller, prefiller_score)
        prefiller_released = True
        leak_state.prefill_completed_at = time.time()
        record_prefill_disconnect_lifecycle(leak_state, "prefill_completed_after_client_disconnect")

        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
        proxy_state.mark_backend_success(prefiller, "prefill_completed_after_disconnect")
        leak_state.prefill_committed = True
        leak_state.kv_transfer_params_received = bool(kv_transfer_params)
        decoder = proxy_state.select_decoder()
        leak_state.decoder = str(decoder)
        leak_state.decoder_started_at = time.time()
        set_pipeline_phase(leak_state, "decoder_running")
        record_prefill_disconnect_lifecycle(
            leak_state,
            "decoder_started_after_client_disconnect",
            decoder_node=str(decoder),
        )

        decoder_stream = stream_service_response_with_retry(
            decoder.client,
            api,
            req_data,
            request_id=request_id,
            prefiller_node=str(prefiller),
            decoder_node=str(decoder),
            decoder_attempt=1,
            first_output_deadline_at=first_output_deadline_at,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
        async for chunk in decoder_stream:
            if not prefiller_kv_released and chunk:
                proxy_state.release_prefiller_kv(prefiller, prefiller_score)
                prefiller_kv_released = True
                leak_state.local_prefiller_kv_released_at = time.time()
                record_prefill_disconnect_lifecycle(leak_state, "local_prefiller_kv_released")
            has_output, decoder_sse_buffer = _decoder_sse_has_semantic_output(
                decoder_sse_buffer, chunk
            )
            if has_output:
                leak_state.first_decoder_output_received = True
                leak_state.first_decoder_output_at = time.time()
                leak_state.drain_stop_reason = "first_decoder_output"
                leak_state.drain_stop_at = leak_state.first_decoder_output_at
                record_prefill_disconnect_lifecycle(
                    leak_state,
                    "first_decoder_semantic_output",
                    decoder_node=str(decoder),
                )
                logger.info(
                    "drain_stopped_after_first_decoder_output request_id=%s prefiller=%s decoder=%s",
                    request_id,
                    prefiller,
                    decoder,
                )
                break
        if leak_state.first_decoder_output_received:
            leak_state.final_request_state = "drain_stopped_after_first_decoder_output"
            finish_prefill_disconnect_drain(leak_state)
        else:
            leak_state.drain_stop_reason = "decoder_stream_ended_before_output"
            leak_state.drain_stop_at = time.time()
            leak_state.final_request_state = "decoder_stream_ended_before_output"
            fail_prefill_disconnect_drain(leak_state, leak_state.drain_stop_reason)
    except asyncio.TimeoutError:
        leak_state.drain_stop_reason = "prefill_timeout_after_client_disconnect"
        leak_state.drain_stop_at = time.time()
        leak_state.final_request_state = "prefill_timeout_after_client_disconnect"
        fail_prefill_disconnect_drain(leak_state, leak_state.drain_stop_reason)
        logger.warning("prefill_timeout_after_client_disconnect request_id=%s", request_id)
    except asyncio.CancelledError:
        leak_state.drain_stop_reason = "shutdown_or_external"
        leak_state.drain_stop_at = time.time()
        leak_state.final_request_state = "shutdown_or_external"
        fail_prefill_disconnect_drain(leak_state, leak_state.drain_stop_reason)
        raise
    except Exception as exc:
        leak_state.drain_stop_reason = "drain_error"
        leak_state.drain_stop_at = time.time()
        leak_state.final_request_state = "drain_error"
        fail_prefill_disconnect_drain(leak_state, "drain_error")
        logger.warning("prefill_disconnect_drain_failed request_id=%s error=%s", request_id, exc)
    finally:
        if decoder_stream is not None:
            with suppress(asyncio.CancelledError, Exception):
                await decoder_stream.aclose()
            leak_state.decoder_stream_closed_at = time.time()
            record_prefill_disconnect_lifecycle(leak_state, "decoder_stream_closed")
        if not prefiller_released:
            proxy_state.release_prefiller(prefiller, prefiller_score)
        if not prefiller_kv_released:
            proxy_state.release_prefiller_kv(prefiller, prefiller_score)
            leak_state.local_prefiller_kv_released_at = time.time()
            record_prefill_disconnect_lifecycle(leak_state, "local_prefiller_kv_released")
        if decoder is not None and not decoder_released:
            proxy_state.release_decoder(decoder, 1.0)
            decoder_released = True
            leak_state.local_decoder_released_at = time.time()
            record_prefill_disconnect_lifecycle(leak_state, "local_decoder_released")
        leak_state.worker_finished_at = time.time()
        set_pipeline_phase(leak_state, "terminal")
        record_prefill_disconnect_lifecycle(
            leak_state,
            "prefill_disconnect_drain_terminal",
            final_request_state=leak_state.final_request_state or "unknown",
        )


def format_request_kind(
    api: str,
    stream_flag: bool | None,
    log_api: str | None = None,
) -> str:
    displayed_api = log_api or api
    if displayed_api == "/responses":
        api_name = "response"
    elif displayed_api == "/chat/completions":
        api_name = "chat"
    else:
        api_name = "completion"
    if stream_flag is None:
        stream_name = "unknown"
    else:
        stream_name = "true" if stream_flag else "false"
    return f"api={api_name}|stream={stream_name}"


async def _handle_completions(
    api: str,
    request: Request,
    log_api: str | None = None,
    *,
    emit_done_on_stream_error: bool = True,
    response_state: ResponseRequestState | None = None,
):
    request_start_time = time.time()
    audit = _active_request_audit.get()
    client_request_id = uuid.uuid4().hex
    if audit is not None:
        audit.bind(client_request_id=client_request_id)
    request_id_token = _active_client_request_id.set(client_request_id)
    failure_capture_token = None
    request_receive_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_start_time))
    client_addr = f"{request.client.host}:{request.client.port}" if request.client else "unknown"
    request_kind = format_request_kind(api, None, log_api)
    input_tokens = 0
    decoder_input_tokens = 0
    requested_max_tokens = 0
    allowed_max_tokens: Any = "N/A"
    instance_info = None
    streaming_started = False
    body_bytes = 0
    read_body_time = 0.0
    json_parse_time = 0.0
    token_count_time = 0.0
    ttft_select_time = 0.0
    ttft_prefill_time = 0.0
    input_chars = 0
    stream_flag = False
    leak_state = RequestLeakState(
        client_addr=client_addr,
        api=api,
        trace=_active_wire_trace.get(),
    )
    if response_state is not None:
        response_state.pipeline_leak_state = leak_state
    try:
        proxy_state.inc_request_num()
        stats_collector.record_rcvd()
        read_body_start = time.time()
        request_body = await request.body()
        if audit is not None:
            audit.capture_body(request_body)
        read_body_time = time.time() - read_body_start
        body_bytes = len(request_body)

        json_parse_start = time.time()
        try:
            req_data = await asyncio.to_thread(json.loads, request_body)
            if audit is not None:
                audit.capture_request_json(req_data)
        except json.JSONDecodeError as exc:
            stats_collector.record_error(
                category="client_invalid_json",
                role="proxy",
                status_code=400,
                endpoint=api,
                message=str(exc),
            )
            print(
                f"{request_receive_time} | {client_addr} | {request_kind} | "
                f"{time.time() - request_start_time:.2f}s | 400 | "
                f"I/O=0/0 | max_tokens=0/N/A | prefill=N/A(0.00s) | "
                f"decode=N/A(0.00s) | ttft=N/A | client_invalid_json",
                flush=True,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "object": "error",
                    "message": "Request body must contain valid JSON.",
                    "type": "invalid_request_error",
                    "code": "client_invalid_json",
                },
            )
        if not isinstance(req_data, dict):
            stats_collector.record_error(
                category="client_invalid_json",
                role="proxy",
                status_code=400,
                endpoint=api,
                message="Request body must be a JSON object.",
            )
            return JSONResponse(
                status_code=400,
                content={
                    "object": "error",
                    "message": "Request body must be a JSON object.",
                    "type": "invalid_request_error",
                    "code": "client_invalid_json",
                },
            )
        json_parse_time = time.time() - json_parse_start
        try:
            reasoning_settings = (
                response_state.reasoning_settings if response_state and response_state.reasoning_settings
                else resolve_reasoning_settings(req_data, source="chat")
            )
            req_data = await asyncio.to_thread(
                prepare_backend_request_data, req_data, reasoning_settings=reasoning_settings,
            )
            if not (response_state and response_state.reasoning_settings):
                logger.info("reasoning_normalized %s", json.dumps(reasoning_settings, ensure_ascii=True))
            if audit is not None:
                audit.bind(reasoning=reasoning_settings)
                audit.write_json("routing.decision.json", {"reasoning": reasoning_settings,
                    "backend_route": "pd", "contains_images": _has_multimodal_content(req_data.get("messages", [])),
                    "image_count": image_input_count(req_data.get("messages", []))})
                audit.capture_request_json(req_data, normalized=True)
        except ResponsesRequestError as exc:
            stats_collector.record_error(
                category="client_invalid_json",
                role="proxy",
                status_code=exc.status_code,
                endpoint=api,
                message=exc.code,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "object": "error",
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": exc.code,
                },
            )
        requested_max_tokens = get_request_max_tokens(req_data)
        stream_flag = bool(req_data.get("stream", False))
        request_kind = format_request_kind(api, stream_flag, log_api)
        leak_state.stream = stream_flag
        leak_state.request_kind = request_kind
        leak_state.requested_max_tokens = requested_max_tokens
        source = _active_failure_capture_source.get()
        trace = _active_wire_trace.get()
        failure_capture_token = _active_failure_capture_context.set(
            FailureCaptureContext(
                client_request_id=client_request_id,
                response_id=(source.response_id if source else (trace.response_id if trace else None)),
                api=source.api if source else api,
                request_kind=request_kind,
                client_request_body=(source.client_request_body if source else request_body),
                normalized_backend_request=copy.deepcopy(req_data),
                received_at=request_receive_time,
            )
        )
        if trace:
            trace.write_json("routing.decision.json", {"reasoning": reasoning_settings,
                    "backend_route": "pd", "contains_images": _has_multimodal_content(req_data.get("messages", [])),
                    "image_count": image_input_count(req_data.get("messages", []))})
            trace.write_json("client.request.meta.json", {
                "client_request_id": client_request_id,
                "response_id": trace.response_id,
                "endpoint": api,
                "request_kind": request_kind,
                "body_bytes": body_bytes,
                "body_sha256": hashlib.sha256(request_body).hexdigest(),
                "received_at": request_receive_time,
            })
            trace.write_bytes("client.request.body", request_body)

        max_input_chars = global_args.max_input_chars
        if max_input_chars > 0:
            input_chars = await asyncio.to_thread(estimate_json_string_chars, req_data)
            if input_chars > max_input_chars:
                stats_collector.record_rejected()
                print(
                    f"{request_receive_time} | {client_addr} | {request_kind} | "
                    f"{time.time() - request_start_time:.2f}s | 400 | "
                    f"I/O=N/A/0 | max_tokens={requested_max_tokens}/N/A | "
                    f"prefill=N/A(0.00s) | decode=N/A(0.00s) | ttft=N/A | input_chars_too_large"
                    f" | chars={input_chars}/{max_input_chars} | body={body_bytes}B"
                    f" | read_body={read_body_time:.2f}s | json_parse={json_parse_time:.2f}s"
                    f" | count_tokens=0.00s",
                    flush=True,
                )
                return JSONResponse(
                    status_code=400,
                    content={
                        "object": "error",
                        "message": f"Request text payload has about {input_chars} characters, "
                        f"exceeding the limit of {max_input_chars} characters.",
                        "type": "invalid_request_error",
                        "code": "input_chars_too_large",
                    },
                )

        # Token limit check: validate against both prefiller and decoder max_model_len
        # In PD disaggregated deployment, prefiller and decoder may have different limits.
        # - Prefiller: input_tokens + 1 must fit because prefill sends max_tokens=1
        # - Decoder: input_tokens + requested max_tokens must fit
        token_count_start = time.time()
        input_tokens, decoder_input_tokens = await count_request_tokens(req_data, tokenizer)
        leak_state.input_tokens = input_tokens
        leak_state.decoder_input_tokens = decoder_input_tokens
        token_count_time = time.time() - token_count_start
        allowed_max_tokens = (
            decoder_max_model_len - decoder_input_tokens - SAFE_CONTEXT_BUFFER_TOKENS
            if decoder_max_model_len is not None else 0
        )
        if prefiller_max_model_len is not None and (
            input_tokens + 1 + SAFE_CONTEXT_BUFFER_TOKENS > prefiller_max_model_len
        ):
            stats_collector.record_rejected()
            print(
                f"{request_receive_time} | {client_addr} | {request_kind} | "
                f"{time.time() - request_start_time:.2f}s | 400 | "
                f"I/O={input_tokens}/0 | "
                f"max_tokens={requested_max_tokens} | "
                f"prefill=N/A(0.00s) | decode=N/A(0.00s) | ttft=N/A | prefill_context_length_exceeded",
                f" | body={body_bytes}B | read_body={read_body_time:.2f}s"
                f" | json_parse={json_parse_time:.2f}s | count_tokens={token_count_time:.2f}s",
                flush=True,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "object": "error",
                "message": f"Prefiller's maximum context length is {prefiller_max_model_len} tokens, "
                    f"however the input has {input_tokens} tokens and the proxy reserves "
                    f"{SAFE_CONTEXT_BUFFER_TOKENS} safety-buffer tokens after DeepSeek-V4 prompt encoding. "
                    f"Please reduce the length of the input.",
                    "type": "invalid_request_error",
                    "code": "prefill_context_length_exceeded",
                },
            )
        if decoder_max_model_len is not None and (
            decoder_input_tokens + requested_max_tokens + SAFE_CONTEXT_BUFFER_TOKENS > decoder_max_model_len
        ):
            stats_collector.record_rejected()
            print(
                f"{request_receive_time} | {client_addr} | {request_kind} | "
                f"{time.time() - request_start_time:.2f}s | 400 | "
                f"I/O={decoder_input_tokens}/0 | "
                f"max_tokens={requested_max_tokens} | "
                f"prefill=N/A(0.00s) | decode=N/A(0.00s) | ttft=N/A | decode_context_length_exceeded",
                f" | body={body_bytes}B | read_body={read_body_time:.2f}s"
                f" | json_parse={json_parse_time:.2f}s | count_tokens={token_count_time:.2f}s",
                flush=True,
            )
            return JSONResponse(
                status_code=400,
                content={
                    "object": "error",
                "message": f"Decoder's maximum context length is {decoder_max_model_len} tokens, "
                    f"however you requested {decoder_input_tokens + requested_max_tokens} tokens "
                    f"({decoder_input_tokens} input + {requested_max_tokens} max_tokens), and the proxy reserves "
                    f"{SAFE_CONTEXT_BUFFER_TOKENS} safety-buffer tokens after DeepSeek-V4 prompt encoding. "
                    f"Please reduce the length of the input or max_tokens.",
                    "type": "invalid_request_error",
                    "code": "decode_context_length_exceeded",
                },
            )

        effective_tokens = input_tokens
        effective_decoder_tokens = decoder_input_tokens
        input_log_value = input_tokens
        select_instance_start = time.time()
        instance_info = await _handle_select_instance(
            api,
            req_data,
            effective_tokens,
            effective_decoder_tokens,
            requested_max_tokens,
            leak_state=leak_state,
            allow_prefill_disconnect_drain=bool(stream_flag),
        )
        ttft_select_time += time.time() - select_instance_start
        ttft_prefill_time += instance_info.prefill_time
        if audit is not None:
            audit.bind(
                prefill_completed_at=time.time(),
                decoder_started_at=time.time(),
                upstream_request_id=instance_info.request_id,
            )
            audit.event("backend_started")
        update_request_leak_targets(leak_state, instance_info)
        set_pipeline_phase(leak_state, "decoder_running")
        leak_state.decoder_started_at = time.time()
        leak_state.prefill_committed = True
        leak_state.kv_transfer_params_received = bool(req_data.get("kv_transfer_params"))
        pf_key = f"{instance_info.prefiller.host}:{instance_info.prefiller.port}"
        dc_key = f"{instance_info.decoder.host}:{instance_info.decoder.port}"
        stats_collector.record_pf_rcvd(pf_key)
        stats_collector.record_dc_rcvd(dc_key)
        # Bug 1 fix: record prefill stats immediately after prefill completes
        stats_collector.record_prefill(pf_key, input_tokens, instance_info.prefill_time)
        chat_flag = "messages" in req_data
        base_messages = copy.deepcopy(req_data.get("messages", [])) if chat_flag else None

        if "prompt" in req_data:
            origin_prompt = req_data["prompt"]
        else:
            origin_prompt = ""
        # refer to vLLM sampling_params: max_token default value
        origin_max_tokens = requested_max_tokens
        leak_state.origin_max_tokens = origin_max_tokens

        if not stream_flag:
            streaming_started = True
            generated_token = ""
            released_kv = False
            decoder_released = False
            first_token_time = None
            retry_count = 0
            output_accounting = StreamOutputAccounting()
            decode_total_time = 0.0
            decode_attempt_start = time.time()
            try:
                while True:
                    leak_state.decode_attempts += 1
                    attempt_accounting = output_accounting.start_attempt()
                    if not released_kv:
                        proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                        released_kv = True
                    response = await send_decode_request_with_retry(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    )
                    response_json = response.json()
                    response_info = extract_stream_choice_info(response_json)
                    output_accounting.observe(attempt_accounting, response_info, time.time())
                    output_accounting.finalize_attempt(attempt_accounting)
                    generated_token += response_info.text
                    attempt_time = time.time() - decode_attempt_start

                    if response_info.stop_reason != "recomputed":
                        decode_total_time += attempt_time
                        decode_time = decode_total_time
                        instance_info.decode_time = decode_time
                        leak_state.decoder_finished_normally = True
                        total_output_tokens = output_accounting.total_tokens
                        leak_state.completion_tokens = total_output_tokens or 0
                        leak_state.final_request_state = "finished"
                        proxy_state.mark_backend_success(instance_info.decoder, "decode_completed")
                        stats_collector.record_completed()
                        stats_collector.record_decode(
                            dc_key, attempt_accounting.total_tokens, attempt_time,
                            token_source=attempt_accounting.token_source,
                            text_tokens=attempt_accounting.text_tokens,
                            reasoning_tokens=attempt_accounting.reasoning_tokens,
                            tool_argument_tokens=attempt_accounting.tool_argument_tokens,
                        )
                        pf = f"{pf_key}({instance_info.prefill_time:.2f}s)"
                        dc = f"{dc_key}({decode_time:.2f}s)"
                        ttft = "N/A"
                        throughput = (
                            f"{total_output_tokens / decode_time:.2f}tok/s"
                            if total_output_tokens is not None and decode_time > 0 else "N/A"
                        )
                        print(
                            f"{request_receive_time} | {client_addr} | {request_kind} | "
                            f"{time.time() - request_start_time:.2f}s | 200 | "
                            f"I/O={input_log_value}/{total_output_tokens if total_output_tokens is not None else 'unknown'} | "
                            f"throughput={throughput} | "
                            f"max_tokens={requested_max_tokens} | "
                            f"P={pf} | D={dc} | ttft={ttft}",
                            flush=True,
                        )
                        return JSONResponse(content=response_json)

                    retry_count += 1
                    leak_state.recomputed_seen = True
                    leak_state.recomputed_attempts = retry_count
                    recomputed_pf_key = pf_key
                    recomputed_dc_key = dc_key
                    decode_total_time += attempt_time
                    stats_collector.record_decode(
                        dc_key, attempt_accounting.total_tokens, attempt_time,
                        token_source=attempt_accounting.token_source,
                        text_tokens=attempt_accounting.text_tokens,
                        reasoning_tokens=attempt_accounting.reasoning_tokens,
                        tool_argument_tokens=attempt_accounting.tool_argument_tokens,
                    )
                    trigger_prefiller_abort_cleanup(
                        instance_info.prefiller,
                        req_data,
                        instance_info.request_id,
                        "decoder_recomputed_nonstream",
                    )
                    proxy_state.release_decoder(instance_info.decoder, instance_info.decoder_score)
                    decoder_released = True
                    prepare_retry_request(req_data, origin_prompt, generated_token, chat_flag, base_messages)
                    set_request_max_tokens(req_data, origin_max_tokens - (output_accounting.total_tokens or 0) + retry_count)
                    retry_tokens, retry_decoder_tokens = await count_request_tokens(req_data, tokenizer)
                    remaining_max_tokens = get_request_max_tokens(req_data)
                    if prefiller_max_model_len is not None and retry_tokens + 1 + SAFE_CONTEXT_BUFFER_TOKENS > prefiller_max_model_len:
                        raise ResponsesRequestError("Recomputed image/text context exceeds Prefiller capacity.", "prefill_context_length_exceeded", 400)
                    if decoder_max_model_len is not None and (
                        retry_decoder_tokens + remaining_max_tokens + SAFE_CONTEXT_BUFFER_TOKENS > decoder_max_model_len
                    ):
                        raise RuntimeError("decoder_recomputed_context_exhausted")
                    retry_select_instance_start = time.time()
                    instance_info = await _handle_select_instance(
                        api, req_data, retry_tokens, retry_decoder_tokens, remaining_max_tokens
                    )
                    stats_collector.record_prefill_recompute(recomputed_pf_key)
                    stats_collector.record_decode_recompute(recomputed_dc_key)
                    if first_token_time is None:
                        ttft_select_time += time.time() - retry_select_instance_start
                        ttft_prefill_time += instance_info.prefill_time
                    update_request_leak_targets(leak_state, instance_info)
                    leak_state.prefill_committed = True
                    leak_state.kv_transfer_params_received = bool(req_data.get("kv_transfer_params"))
                    pf_key = f"{instance_info.prefiller.host}:{instance_info.prefiller.port}"
                    dc_key = f"{instance_info.decoder.host}:{instance_info.decoder.port}"
                    stats_collector.record_pf_rcvd(pf_key)
                    stats_collector.record_dc_rcvd(dc_key)
                    stats_collector.record_prefill(pf_key, retry_tokens, instance_info.prefill_time)
                    decode_attempt_start = time.time()
                    released_kv = False
                    decoder_released = False
                    leak_reason = validate_decode_attempt_budget(leak_state)
                    if leak_reason is not None:
                        raise RuntimeError(leak_reason)
            except Exception:
                leak_state.completion_tokens = output_accounting.total_tokens or 0
                detail = str(sys.exc_info()[1])
                if detail == "decoder_recomputed_context_exhausted":
                    reason = "decoder_recomputed_context_exhausted"
                elif detail in {
                    "decoder_recomputed_budget_exhausted",
                    "decode_attempt_budget_exhausted",
                }:
                    reason = detail
                else:
                    reason = "decoder_recomputed_nonstream_failed" if leak_state.recomputed_seen else "decoder_error_nonstream"
                trigger_prefiller_abort_cleanup(instance_info.prefiller, req_data, instance_info.request_id, reason)
                mark_request_leak(leak_state, reason, "nonstream_decode_failed")
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                decode_exc = sys.exc_info()[1]
                stats_collector.record_error(
                    category=backend_error_category(decode_exc),
                    role=InstanceType.DECODE,
                    node=str(instance_info.decoder),
                    status_code=backend_error_status(decode_exc),
                    endpoint=api,
                    message=summarize_error(decode_exc),
                )
                setattr(decode_exc, "_proxy_stats_recorded", True)
                failure_category, failure_count, taint_decision = should_taint_decoder_for_error(
                    decode_exc, instance_info.decoder
                )
                logger.info(
                    "decoder_failure request_id=%s client_request_id=%s upstream_request_id=%s "
                    "decoder=%s failure_category=%s failure_count=%s threshold=%s window_seconds=%s taint_decision=%s",
                    instance_info.request_id, _active_client_request_id.get(), instance_info.request_id,
                    instance_info.decoder, failure_category, failure_count,
                    getattr(global_args, "decoder_taint_failure_threshold", DECODER_TAINT_FAILURE_THRESHOLD),
                    getattr(global_args, "decoder_taint_failure_window_seconds", DECODER_TAINT_FAILURE_WINDOW_SECONDS),
                    taint_decision,
                )
                if not decoder_released and taint_decision:
                    proxy_state.taint_decoder(instance_info.decoder, reason=failure_category)
                raise
            finally:
                if not decoder_released:
                    proxy_state.release_decoder(instance_info.decoder, instance_info.decoder_score)
                proxy_state.dec_request_num()

        stream_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        stream_consumer_closed = asyncio.Event()
        worker_task: asyncio.Task | None = None
        output_accounting = StreamOutputAccounting()
        chat_guard = ChatStreamGuard(
            enabled=(chat_flag and stream_flag and getattr(global_args, "enable_chat_stream_guard", CHAT_STREAM_GUARD_ENABLED)),
            trace_enabled=getattr(global_args, "chat_stream_guard_trace", CHAT_STREAM_GUARD_TRACE_ENABLED),
            max_bytes=CHAT_STREAM_GUARD_TRACE_MAX_BYTES,
        )

        async def enqueue_stream_chunk(data: bytes) -> None:
            if not leak_state.client_disconnected and not stream_consumer_closed.is_set():
                await stream_queue.put(data)

        async def run_stream_worker():
            nonlocal instance_info, pf_key, dc_key, ttft_select_time, ttft_prefill_time
            generated_token = ""
            # The first Prefill mutates req_data with KV transfer parameters.
            # A start-protocol failure must not carry those consumed KV handles
            # into the replacement Prefill attempt.
            protocol_recompute_request = copy.deepcopy(req_data)
            protocol_recompute_request.pop("kv_transfer_params", None)
            released_kv = False
            decoder_released = False
            retry_count = 0
            protocol_recompute_count = 0
            protocol_excluded_decoders: set[ServerState] = set()
            protocol_retry_prompt_applied = False
            retry = True
            decode_total_time = 0.0
            decode_attempt_start = time.time()
            decode_time = 0.0
            first_token_time = None
            first_stream_output_time = None
            stream_finished = False
            stream_error = None
            sse_buffer = ""

            def record_attempt_decode(
                key: str,
                attempt: StreamOutputAttempt,
                attempt_time: float,
            ) -> None:
                output_accounting.finalize_attempt(attempt)
                first_output_at, last_output_at = attempt.first_output_at, attempt.last_output_at
                tpot_seconds = None
                if (
                    attempt.total_tokens is not None
                    and attempt.total_tokens > 1
                    and first_output_at is not None
                    and last_output_at is not None
                ):
                    tpot_seconds = max(0.0, last_output_at - first_output_at) / (attempt.total_tokens - 1)
                stats_collector.record_decode(
                    key,
                    attempt.total_tokens,
                    attempt_time,
                    token_source=attempt.token_source,
                    text_tokens=attempt.text_tokens,
                    reasoning_tokens=attempt.reasoning_tokens,
                    tool_argument_tokens=attempt.tool_argument_tokens,
                    tpot_seconds=tpot_seconds,
                )

            try:
                while retry:
                    retry = False
                    leak_state.decode_attempts += 1
                    attempt_accounting = output_accounting.start_attempt()
                    attempt_done_observed = False
                    attempt_has_valid_output = False
                    attempt_stream_progress = DecoderStreamProgress()
                    protocol_marker_buffer = ""
                    buffered_protocol_events: list[tuple[str, StreamChoiceInfo]] = []

                    async def flush_buffered_protocol_events() -> None:
                        nonlocal first_stream_output_time, generated_token
                        nonlocal first_token_time, attempt_has_valid_output
                        for buffered_event, buffered_info in buffered_protocol_events:
                            if first_stream_output_time is None:
                                first_stream_output_time = time.time() - request_start_time
                            output_accounting.observe(
                                attempt_accounting, buffered_info, time.time()
                            )
                            generated_token += buffered_info.text
                            leak_state.generated_text = generated_token
                            if buffered_info.has_output and first_token_time is None:
                                first_token_time = time.time() - request_start_time
                            if buffered_info.has_output:
                                attempt_has_valid_output = True
                                attempt_stream_progress.observe_valid_output()
                            await enqueue_stream_chunk(
                                f"{buffered_event}\n\n".encode("utf-8")
                            )
                        buffered_protocol_events.clear()
                    async for chunk in stream_service_response_with_retry(
                        instance_info.decoder.client,
                        api,
                        req_data,
                        request_id=instance_info.request_id,
                        prefiller_node=str(instance_info.prefiller),
                        decoder_node=str(instance_info.decoder),
                        decoder_attempt=leak_state.decode_attempts,
                        first_output_deadline_at=instance_info.first_output_deadline_at,
                        first_prefill_sent_at=instance_info.first_prefill_sent_at,
                        stream_progress=attempt_stream_progress,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay,
                    ):
                        leak_state.last_decoder_chunk_at = time.time()
                        if chat_guard.enabled and chunk:
                            chat_guard.record_raw(chunk)
                        if not released_kv and chunk:
                            proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                            released_kv = True
                        try:
                            chunk_str = chunk.decode("utf-8")
                        except UnicodeDecodeError:
                            logger.debug(f"Skipping chunk: {chunk}")
                            await enqueue_stream_chunk(chunk)
                            continue
                        if not chunk_str:
                            continue
                        sse_buffer += chunk_str.replace("\r\n", "\n")
                        while "\n\n" in sse_buffer:
                            event, sse_buffer = sse_buffer.split("\n\n", 1)
                            data_lines = []
                            for line in event.splitlines():
                                line = line.strip()
                                if line.startswith("data:"):
                                    data_lines.append(line[len("data:"):].strip())
                            if not data_lines:
                                continue
                            payload = "\n".join(data_lines)
                            if payload == "[DONE]":
                                if chat_guard.enabled:
                                    chat_guard.done()
                                    if chat_guard.anomaly:
                                        promote_chat_guard_trace(_active_request_audit.get(), chat_guard)
                                        raise ChatEmptyCompletionError(chat_guard.anomaly)
                                # A legitimate reasoning response can happen to begin with a
                                # prefix of the reserved marker.  Do not lose it merely because
                                # the Decoder completed before the prefix became distinguishable.
                                if buffered_protocol_events:
                                    await flush_buffered_protocol_events()
                                    protocol_marker_buffer = ""
                                attempt_done_observed = True
                                leak_state.decoder_done_observed = True
                                record_decoder_attempt_trace(
                                    _active_wire_trace.get(),
                                    "protocol_done_observed",
                                    request_id=instance_info.request_id,
                                    response_id=(
                                        _active_wire_trace.get().response_id
                                        if _active_wire_trace.get()
                                        else None
                                    ),
                                    decoder_node=str(instance_info.decoder),
                                    decoder_attempt=leak_state.decode_attempts,
                                )
                                output_accounting.finalize_attempt(attempt_accounting)
                                if decode_time == 0.0:
                                    decode_time = decode_total_time + (time.time() - decode_attempt_start)
                                await enqueue_stream_chunk(b"data: [DONE]\n\n")
                                continue
                            try:
                                chunk_json = json.loads(payload)
                            except json.JSONDecodeError:
                                logger.debug(f"Skipping SSE payload: {payload}")
                                if buffered_protocol_events:
                                    await flush_buffered_protocol_events()
                                    protocol_marker_buffer = ""
                                await enqueue_stream_chunk(f"{event}\n\n".encode("utf-8"))
                                continue

                            stream_info = extract_stream_choice_info(chunk_json)
                            if chat_guard.enabled:
                                raw_event = event
                                chat_guard.record(raw_event, chunk_json)
                                if not chat_guard.inspect(chunk_json, stream_info):
                                    continue
                                if chat_guard.anomaly:
                                    promote_chat_guard_trace(_active_request_audit.get(), chat_guard)
                                    raise ChatEmptyCompletionError(chat_guard.anomaly)
                            if not attempt_has_valid_output and buffered_protocol_events:
                                marker_state = decoder_start_protocol_marker_state(
                                    protocol_marker_buffer,
                                    stream_info,
                                    attempt_has_valid_output,
                                )
                                if marker_state != "not_marker":
                                    protocol_marker_buffer += stream_info.reasoning
                                    buffered_protocol_events.append((event, stream_info))
                                    if marker_state == "prefix":
                                        continue
                                else:
                                    await flush_buffered_protocol_events()
                                    protocol_marker_buffer = ""

                            if (
                                not attempt_has_valid_output
                                and not buffered_protocol_events
                                and stream_info.reasoning
                                and not stream_info.text
                                and not stream_info.tool_arguments
                            ):
                                marker_state = decoder_start_protocol_marker_state(
                                    "", stream_info, attempt_has_valid_output
                                )
                                if marker_state != "not_marker":
                                    protocol_marker_buffer = stream_info.reasoning
                                    buffered_protocol_events.append((event, stream_info))
                                    if marker_state == "prefix":
                                        continue

                            protocol_leak = (
                                not attempt_has_valid_output
                                and protocol_marker_buffer in DECODER_START_PROTOCOL_MARKERS
                                and bool(buffered_protocol_events)
                            ) or is_decoder_start_protocol_leak(
                                stream_info, attempt_has_valid_output
                            )
                            if protocol_leak:
                                leak_state.decoder_start_protocol_seen = True
                                stats_collector.record_error(
                                    category="decoder_start_protocol_detected",
                                    role=InstanceType.DECODE,
                                    node=str(instance_info.decoder),
                                    endpoint=api,
                                    message="Decoder emitted its reserved start protocol marker.",
                                )
                                marker_variant = (
                                    protocol_marker_buffer
                                    if protocol_marker_buffer in DECODER_START_PROTOCOL_MARKERS
                                    else stream_info.reasoning
                                )
                                record_decoder_attempt_trace(
                                    _active_wire_trace.get(),
                                    "decoder_start_protocol_detected",
                                    request_id=instance_info.request_id,
                                    response_id=(
                                        _active_wire_trace.get().response_id
                                        if _active_wire_trace.get()
                                        else None
                                    ),
                                    prefiller_node=str(instance_info.prefiller),
                                    decoder_node=str(instance_info.decoder),
                                    decoder_attempt=leak_state.decode_attempts,
                                    marker_variant=decoder_start_protocol_marker_variant(marker_variant),
                                )
                                logger.warning(
                                    "decoder_start_protocol_detected request_id=%s prefiller=%s "
                                    "decoder=%s decoder_attempt=%s marker=%s action=alert_forward",
                                    instance_info.request_id,
                                    instance_info.prefiller,
                                    instance_info.decoder,
                                    leak_state.decode_attempts,
                                    decoder_start_protocol_marker_variant(marker_variant),
                                )
                                # The marker alone is not proof of a failed Decoder. Preserve
                                # the upstream stream exactly while recording the anomaly.
                                await flush_buffered_protocol_events()
                                protocol_marker_buffer = ""
                                continue

                            if first_stream_output_time is None:
                                first_stream_output_time = time.time() - request_start_time
                            output_accounting.observe(attempt_accounting, stream_info, time.time())
                            generated_token += stream_info.text
                            leak_state.generated_text = generated_token
                            if stream_info.has_output and first_token_time is None:
                                first_token_time = time.time() - request_start_time
                            if stream_info.has_output:
                                attempt_has_valid_output = True
                                attempt_stream_progress.observe_valid_output()

                            if stream_info.stop_reason == "recomputed":
                                retry = True
                                retry_count += 1
                                leak_state.recomputed_seen = True
                                leak_state.recomputed_attempts = retry_count
                                recomputed_pf_key = pf_key
                                recomputed_dc_key = dc_key
                                attempt_time = time.time() - decode_attempt_start
                                record_attempt_decode(dc_key, attempt_accounting, attempt_time)
                                decode_total_time += attempt_time
                                trigger_prefiller_abort_cleanup(
                                    instance_info.prefiller,
                                    req_data,
                                    instance_info.request_id,
                                    "decoder_recomputed_stream",
                                )
                                proxy_state.release_decoder(instance_info.decoder, instance_info.decoder_score)
                                decoder_released = True
                                if not released_kv:
                                    proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                                prepare_retry_request(req_data, origin_prompt, generated_token, chat_flag, base_messages)
                                completed_output_tokens = output_accounting.total_tokens or 0
                                set_request_max_tokens(req_data, origin_max_tokens - completed_output_tokens + retry_count)
                                retry_tokens, retry_decoder_tokens = await count_request_tokens(req_data, tokenizer)
                                remaining_max_tokens = get_request_max_tokens(req_data)
                                if prefiller_max_model_len is not None and retry_tokens + 1 + SAFE_CONTEXT_BUFFER_TOKENS > prefiller_max_model_len:
                                    raise ResponsesRequestError("Recomputed image/text context exceeds Prefiller capacity.", "prefill_context_length_exceeded", 400)
                                if decoder_max_model_len is not None and (
                                    retry_decoder_tokens + remaining_max_tokens + SAFE_CONTEXT_BUFFER_TOKENS
                                    > decoder_max_model_len
                                ):
                                    raise RuntimeError("decoder_recomputed_context_exhausted")
                                retry_select_instance_start = time.time()
                                instance_info = await _handle_select_instance(
                                    api, req_data, retry_tokens, retry_decoder_tokens, remaining_max_tokens
                                )
                                stats_collector.record_prefill_recompute(recomputed_pf_key)
                                stats_collector.record_decode_recompute(recomputed_dc_key)
                                if first_token_time is None:
                                    ttft_select_time += time.time() - retry_select_instance_start
                                    ttft_prefill_time += instance_info.prefill_time
                                update_request_leak_targets(leak_state, instance_info)
                                leak_state.prefill_committed = True
                                leak_state.kv_transfer_params_received = bool(req_data.get("kv_transfer_params"))
                                pf_key = f"{instance_info.prefiller.host}:{instance_info.prefiller.port}"
                                dc_key = f"{instance_info.decoder.host}:{instance_info.decoder.port}"
                                stats_collector.record_pf_rcvd(pf_key)
                                stats_collector.record_dc_rcvd(dc_key)
                                stats_collector.record_prefill(pf_key, retry_tokens, instance_info.prefill_time)
                                decode_attempt_start = time.time()
                                released_kv = False
                                decoder_released = False
                                sse_buffer = ""
                                leak_reason = validate_decode_attempt_budget(leak_state)
                                if leak_reason is not None:
                                    raise RuntimeError(leak_reason)
                                break
                            await enqueue_stream_chunk(f"{event}\n\n".encode("utf-8"))
                        if retry:
                            break
                    if not retry and buffered_protocol_events:
                        await flush_buffered_protocol_events()
                        protocol_marker_buffer = ""
                    if not retry and not attempt_done_observed:
                        if chat_guard.enabled:
                            chat_guard.anomaly = chat_guard.anomaly or "incomplete_stream"
                            promote_chat_guard_trace(_active_request_audit.get(), chat_guard)
                        record_decoder_attempt_trace(
                            _active_wire_trace.get(),
                            "protocol_incomplete",
                            request_id=instance_info.request_id,
                            response_id=(
                                _active_wire_trace.get().response_id
                                if _active_wire_trace.get()
                                else None
                            ),
                            decoder_node=str(instance_info.decoder),
                            decoder_attempt=leak_state.decode_attempts,
                        )
                        stats_collector.record_decoder_stream_eof_without_done()
                        raise DecoderStreamIncompleteError(
                            "Decoder SSE stream ended without data: [DONE]."
                        )
                stream_finished = True
            except asyncio.CancelledError:
                if proxy_shutdown_started:
                    stream_error = RuntimeError("proxy_shutdown")
                    return
                raise
            except Exception as e:
                stream_error = e
                if chat_guard.enabled and not chat_guard.has_semantic_output:
                    chat_guard.anomaly = chat_guard.anomaly or "stream_error_before_semantic_output"
                    promote_chat_guard_trace(_active_request_audit.get(), chat_guard)
                leak_state.completion_tokens = output_accounting.total_tokens or 0
                stats_collector.record_error(
                    category=backend_error_category(e),
                    role=InstanceType.DECODE,
                    node=str(instance_info.decoder),
                    status_code=backend_error_status(e),
                    endpoint=api,
                    message=summarize_error(e),
                )
                setattr(e, "_proxy_stats_recorded", True)
                if is_backend_client_error(e):
                    logger.info(
                        "Backend client error during streaming from decoder %s: %s",
                        instance_info.decoder.url,
                        backend_error_summary(e),
                    )
                elif isinstance(e, ChatEmptyCompletionError):
                    logger.warning(
                        "Chat stream guard rejected an empty completion from decoder %s; "
                        "decoder remains active",
                        instance_info.decoder.url,
                    )
                else:
                    logger.error(
                        f"Error during streaming from decoder {instance_info.decoder.url}: {str(e)} "
                        f"the aborted request {instance_info.request_id} will be routing to the target "
                        "prefiller when new request is ready to dispatch to it"
                    )
                if isinstance(e, PdFirstOutputTimeoutError):
                    reason = "pd_first_output_timeout"
                elif isinstance(e, DecoderStreamIdleTimeoutError):
                    reason = "decoder_stream_idle_timeout"
                elif isinstance(e, DecoderStartProtocolError):
                    reason = "decoder_start_protocol_recompute_exhausted"
                elif str(e) == "decoder_start_protocol_context_exhausted":
                    reason = "decoder_start_protocol_recompute_exhausted"
                elif str(e) == "decoder_recomputed_context_exhausted":
                    reason = "decoder_recomputed_context_exhausted"
                elif str(e) in {
                    "decoder_recomputed_budget_exhausted",
                    "decode_attempt_budget_exhausted",
                }:
                    reason = str(e)
                elif leak_state.recomputed_seen:
                    reason = "decoder_recomputed_stream_failed"
                else:
                    reason = "decoder_error_stream"
                trigger_prefiller_abort_cleanup(instance_info.prefiller, req_data, instance_info.request_id, reason)
                mark_request_leak(leak_state, reason, "stream_failed")
                if not released_kv:
                    proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                failure_category, failure_count, taint_decision = should_taint_decoder_for_error(e, instance_info.decoder)
                logger.info(
                    "decoder_failure request_id=%s client_request_id=%s upstream_request_id=%s "
                    "decoder=%s failure_category=%s failure_count=%s threshold=%s window_seconds=%s taint_decision=%s",
                    instance_info.request_id, _active_client_request_id.get(), instance_info.request_id,
                    instance_info.decoder, failure_category, failure_count,
                    getattr(global_args, "decoder_taint_failure_threshold", DECODER_TAINT_FAILURE_THRESHOLD),
                    getattr(global_args, "decoder_taint_failure_window_seconds", DECODER_TAINT_FAILURE_WINDOW_SECONDS),
                    taint_decision,
                )
                if not decoder_released and taint_decision:
                    proxy_state.taint_decoder(instance_info.decoder, reason=failure_category)
                if stream_flag and not leak_state.client_disconnected:
                    if is_backend_client_error(e):
                        error_payload = {
                            "error": {
                                "object": "error",
                                "message": _safe_response_text(e.response) or "Backend client error",
                                "type": "backend_client_error",
                                "code": f"backend_http_{e.response.status_code}",
                            }
                        }
                    else:
                        error_code = e.code if isinstance(e, ResponsesRequestError) else (
                            "empty_model_completion"
                            if isinstance(e, ChatEmptyCompletionError)
                            else
                            "pd_first_output_timeout"
                            if isinstance(e, PdFirstOutputTimeoutError)
                            else
                            "decoder_stream_idle_timeout"
                            if isinstance(e, DecoderStreamIdleTimeoutError)
                            else
                            "decoder_start_protocol_recompute_exhausted"
                            if isinstance(e, DecoderStartProtocolError)
                            else
                            "decoder_stream_incomplete"
                            if isinstance(e, DecoderStreamIncompleteError)
                            else "decoder_stream_error"
                        )
                        error_payload = {
                            "error": {
                                "object": "error",
                                "message": str(e),
                                "type": "stream_error",
                                "code": error_code,
                            }
                        }
                    await enqueue_stream_chunk(
                        f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    )
                    if emit_done_on_stream_error:
                        await enqueue_stream_chunk(b"data: [DONE]\n\n")
                else:
                    raise
            finally:
                if not stream_finished and stream_error is None and not released_kv:
                    reason = "stream_ended_without_normal_finish"
                    trigger_prefiller_abort_cleanup(instance_info.prefiller, req_data, instance_info.request_id, reason)
                    mark_request_leak(leak_state, reason, "stream_ended_early")
                    proxy_state.release_prefiller_kv(instance_info.prefiller, instance_info.prefiller_score)
                if not decoder_released:
                    proxy_state.release_decoder(instance_info.decoder, instance_info.decoder_score)
                proxy_state.dec_request_num()
                leak_state.worker_finished_at = time.time()
                set_pipeline_phase(leak_state, "terminal")
                if stream_finished and stream_error is None:
                    leak_state.decoder_finished_normally = True
                    leak_state.final_request_state = "finished"
                    output_accounting.finalize_attempt(attempt_accounting)
                    total_output_tokens = output_accounting.total_tokens
                    leak_state.completion_tokens = total_output_tokens or 0
                    if decode_time == 0.0:
                        decode_time = decode_total_time + (time.time() - decode_attempt_start)
                    instance_info.decode_time = decode_time
                    final_attempt_time = max(0.0, decode_time - decode_total_time)
                    proxy_state.mark_backend_success(instance_info.decoder, "decode_completed")
                    stats_collector.record_completed()
                    record_attempt_decode(dc_key, attempt_accounting, final_attempt_time)
                    pf = f"{pf_key}({instance_info.prefill_time:.2f}s)"
                    decoder_wait_first_token = attempt_stream_progress.first_valid_output_delay
                    dc = (
                        f"{dc_key}(first_token={decoder_wait_first_token:.2f}s,"
                        f"total={decode_time:.2f}s)"
                        if decoder_wait_first_token is not None
                        else f"{dc_key}(first_token=N/A,total={decode_time:.2f}s)"
                    )
                    ttft_value = first_token_time if first_token_time is not None else first_stream_output_time
                    ttft = f"{ttft_value:.2f}s" if ttft_value is not None else "N/A"
                    ttft_detail = build_ttft_detail(
                        ttft_value,
                        read_body_time + json_parse_time + token_count_time,
                        ttft_select_time,
                        ttft_prefill_time,
                    )
                    ttft_components = build_ttft_components(
                        ttft_value,
                        read_body_time + json_parse_time + token_count_time,
                        ttft_select_time,
                        ttft_prefill_time,
                    )
                    if ttft_components is not None:
                        stats_collector.record_ttft(*ttft_components)
                    first_output_at, last_output_at = output_accounting.output_window()
                    throughput = (
                        f"{total_output_tokens / decode_time:.2f}tok/s"
                        if total_output_tokens is not None and decode_time > 0 else "N/A"
                    )
                    tpot = "N/A"
                    if (
                        total_output_tokens is not None and total_output_tokens > 1
                        and first_output_at is not None and last_output_at is not None
                    ):
                        tpot = f"{(last_output_at - first_output_at) / (total_output_tokens - 1):.4f}s/tok"
                    print(
                        f"{request_receive_time} | {client_addr} | {request_kind} | "
                        f"{time.time() - request_start_time:.2f}s | 200 | "
                        f"I/O={input_log_value}/{total_output_tokens if total_output_tokens is not None else 'unknown'} | "
                        f"throughput={throughput} | "
                        f"max_tokens={requested_max_tokens} | "
                        f"P={pf} | D={dc} | ttft={ttft_detail}",
                        flush=True,
                    )
                stream_consumer_closed.set()
                await stream_queue.put(None)

        async def consume_stream():
            completed_normally = False
            try:
                while True:
                    chunk = await stream_queue.get()
                    if chunk is None:
                        completed_normally = True
                        break
                    yield chunk
            except asyncio.CancelledError:
                mark_client_disconnected(leak_state)
                if leak_state.pipeline_phase == "decoder_running" and worker_task is not None and not worker_task.done():
                    worker_task.cancel("client_disconnected_during_decoder")
                raise
            finally:
                stream_consumer_closed.set()
                if not completed_normally and not leak_state.decoder_finished_normally:
                    mark_client_disconnected(leak_state)
                    if leak_state.pipeline_phase == "decoder_running" and worker_task is not None and not worker_task.done():
                        worker_task.cancel("client_disconnected_during_decoder")

        streaming_started = True
        worker_task = asyncio.create_task(run_stream_worker())
        detached_task_manager.track(worker_task)
        media_type = "text/event-stream; charset=utf-8" if stream_flag else "application/json"
        response = StreamingResponse(consume_stream(), media_type=media_type)
        response.output_accounting = output_accounting
        response.input_tokens = decoder_input_tokens
        return response
    except Exception as e:
        import traceback

        exc_info = sys.exc_info()
        if leak_state.prefill_committed and not leak_state.decoder_finished_normally and not leak_state.leak_recorded:
            reason = "prefill_disconnect_drain_unknown" if leak_state.prefill_disconnect_draining else "proxy_internal_error_after_prefill"
            mark_request_leak(leak_state, reason, "outer_exception")
        if isinstance(e, ResponsesRequestError):
            stats_collector.record_error(
                category=e.code, role="proxy", status_code=e.status_code,
                endpoint=api, message=str(e),
            )
            return JSONResponse(status_code=e.status_code, content={
                "object": "error", "type": "invalid_request_error" if e.status_code == 400 else "upstream_error",
                "code": e.code, "message": str(e),
            })
        if is_backend_client_error(e):
            status_code = e.response.status_code
            if instance_info is None and not getattr(e, "_proxy_stats_recorded", False):
                stats_collector.record_error(
                    category=backend_error_category(e),
                    role="proxy",
                    node=None,
                    status_code=status_code,
                    endpoint=api,
                    message=summarize_error(e),
                )
            if instance_info is not None:
                pf = f"{pf_key}({instance_info.prefill_time:.2f}s)"
                dc = f"{instance_info.decoder.host}:{instance_info.decoder.port}({instance_info.decode_time:.2f}s)"
            else:
                pf = "N/A(0.00s)"
                dc = "N/A(0.00s)"
            input_log_value = input_tokens
            print(
                f"{request_receive_time} | {client_addr} | {request_kind} | "
                f"{time.time() - request_start_time:.2f}s | {status_code} | "
                f"I/O={input_log_value}/0 | "
                f"max_tokens={requested_max_tokens} | "
                f"P={pf} | D={dc} | ttft=N/A | backend_client_error",
                flush=True,
            )
            return backend_error_response(e)
        if instance_info is None and not getattr(e, "_proxy_stats_recorded", False):
            stats_collector.record_error(
                category=backend_error_category(e),
                role="proxy",
                node=None,
                status_code=backend_error_status(e),
                endpoint=api,
                message=summarize_error(e),
            )
        if instance_info is not None:
            pf = f"{pf_key}({instance_info.prefill_time:.2f}s)"
            dc = f"{instance_info.decoder.host}:{instance_info.decoder.port}({instance_info.decode_time:.2f}s)"
            ttft = "N/A"
        else:
            pf = "N/A(0.00s)"
            dc = "N/A(0.00s)"
            ttft = "N/A"
        input_log_value = input_tokens
        print(
            f"{request_receive_time} | {client_addr} | {request_kind} | "
            f"{time.time() - request_start_time:.2f}s | 500 | "
            f"I/O={input_log_value}/0 | "
            f"max_tokens={requested_max_tokens} | "
            f"P={pf} | D={dc} | ttft={ttft} | error: {e}",
            flush=True,
        )
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise
    finally:
        if failure_capture_token is not None:
            _active_failure_capture_context.reset(failure_capture_token)
        _active_client_request_id.reset(request_id_token)
        # Bug 5 fix: only decrement request_num if streaming hasn't started
        # (generate_stream's finally handles it when streaming is active)
        if not streaming_started:
            proxy_state.dec_request_num()


async def _handle_adjust_instances(adjust_mode: str, request: Request):
    auth_error = _authorize_instance_admin(request)
    if auth_error is not None:
        return auth_error
    try:
        req_data = await request.json()
        if not isinstance(req_data, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_instance_request",
                        "message": "The request body must be a JSON object.",
                    }
                },
            )
        instance_type = req_data.get("type", "")
        instances = req_data.get("instances", [])
        if isinstance(instances, str):
            instances = [instances]
        instances = trans_instances(instances)
        all_msg = f"{adjust_mode} {instance_type} instances: {[str(server) for server in instances]}."

        if instance_type not in [InstanceType.PREFILL, InstanceType.DECODE]:
            return {
                "error": f"Instance type {instance_type} is not supported. "
                f"Only support '{InstanceType.PREFILL}' and '{InstanceType.DECODE}'."
            }

        if adjust_mode == "add":
            added_nodes, waiting_nodes = await proxy_state.add_instances(instance_type, instances)
            if waiting_nodes:
                all_msg = (
                    f"{adjust_mode} {instance_type} instances: {added_nodes}. "
                    f"Instances {waiting_nodes} are waiting to be added."
                )
        elif adjust_mode == "remove":
            if instance_type == InstanceType.PREFILL:
                need_waiting = proxy_state.remove_prefillers(instances)
            else:
                need_waiting = proxy_state.remove_decoders(instances)

            if need_waiting:
                all_msg = f"Instances {instances} are isolated and waiting to be removed."
        return {
            "message": all_msg,
            "current_prefill_instances": [str(prefiller) for prefiller in proxy_state.prefillers],
            "current_decode_instances": [str(decoder) for decoder in proxy_state.decoders],
        }
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                    "message": "The request body must contain valid JSON.",
                }
            },
        )
    except Exception as e:
        logger.error(f"Failed to {adjust_mode} instances: {e}")
        raise


def _authorize_instance_admin(request: Request):
    if not INSTANCE_API_KEY:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "configuration_error",
                    "code": "instance_api_key_not_configured",
                    "message": "Instance administration is not configured.",
                }
            },
        )

    supplied_key = request.headers.get("x-api-key", "").strip()
    if not supplied_key:
        authorization = request.headers.get("authorization", "")
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer":
            supplied_key = value.strip()

    if not supplied_key or not secrets.compare_digest(supplied_key, INSTANCE_API_KEY):
        return JSONResponse(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "error": {
                    "type": "authentication_error",
                    "code": "invalid_instance_api_key",
                    "message": "A valid instance administration API key is required.",
                }
            },
        )
    return None


def trans_instances(instances: list[str]) -> list[ServerState]:
    server_list = []
    for instance in instances:
        h, p = instance.split(":")
        server_list.append(ServerState(h, int(p)))
    return server_list


async def _run_audited_endpoint(request: Request, api_kind: str, handler: Callable[[], Awaitable[Any]]):
    """Apply one audit lifecycle around a public inference endpoint."""
    audit = RequestAuditTrace(request, api_kind)
    if api_kind == "responses":
        stats_collector.record_responses_dsml_filter_request()
    token = _active_request_audit.set(audit)
    try:
        response = await handler()
    except asyncio.CancelledError as exc:
        audit.finish("disconnected", exception=exc, disconnect_phase="request_handler")
        raise
    except Exception as exc:
        audit.finish("failed", exception=exc)
        raise
    finally:
        _active_request_audit.reset(token)

    if isinstance(response, StreamingResponse):
        response.body_iterator = audit.wrap_stream(response.body_iterator)
        return response
    status = getattr(response, "status_code", 200)
    error_code = None
    if status >= 400:
        with suppress(TypeError, ValueError, json.JSONDecodeError):
            body = json.loads(response.body)
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                error_code = error["code"]
    if status >= 500:
        audit.finish("failed", http_status=status, error_code=error_code)
    elif status >= 400:
        audit.finish("rejected", http_status=status, error_code=error_code)
    else:
        audit.finish("accepted", http_status=status)
    return response


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _run_audited_endpoint(
        request, "completions", lambda: _handle_completions("/completions", request)
    )


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    async def handler():
        audit = _active_request_audit.get()
        body = await request.body()
        if audit is not None:
            audit.capture_body(body)
        try:
            request_data = json.loads(body)
            if audit is not None:
                audit.capture_request_json(request_data)
        except (TypeError, json.JSONDecodeError):
            return await _handle_completions("/chat/completions", _synthetic_request(request, body))
        return await _handle_completions("/chat/completions", request)

    return await _run_audited_endpoint(request, "chat_completions", handler)


@app.post("/responses")
@app.post("/v1/responses")
async def handle_responses_route(request: Request):
    return await _run_audited_endpoint(request, "responses", lambda: handle_responses(request))


@app.get("/v1/responses/{response_id}")
async def retrieve_response(response_id: str):
    """Retrieve a completed Responses object from the shared state store."""
    if not response_state_store.enabled:
        return responses_error(
            "Response retrieval requires RESPONSES_REDIS_URL.",
            "response_state_unavailable",
            503,
        )
    try:
        response = await response_state_store.get(response_id)
    except Exception:
        logger.exception("Failed to retrieve response state response_id=%s", response_id)
        return responses_error(
            "Response state is temporarily unavailable.",
            "response_state_unavailable",
            503,
        )
    if response is None:
        return responses_error(
            "The response does not exist or has expired.",
            "response_not_found",
            404,
        )
    return JSONResponse(content=_public_response_state(response))


@app.get("/health")
async def health():
    """Match the vLLM/Prefill liveness endpoint: HTTP 200 with an empty body."""
    return Response(status_code=200)


@app.get("/healthcheck")
async def healthcheck():
    metrics_snapshot = stats_collector.snapshot(proxy_state.prefillers, proxy_state.decoders)
    recent_backend_success = {
        "prefillers": [{
            "node": f"{node['host']}:{node['port']}",
            "last_success_age_seconds": node.get("last_success_age_seconds"),
            "last_success_kind": node.get("last_success_kind"),
        } for node in metrics_snapshot["prefill_nodes"]],
        "decoders": [{
            "node": f"{node['host']}:{node['port']}",
            "last_success_age_seconds": node.get("last_success_age_seconds"),
            "last_success_kind": node.get("last_success_kind"),
        } for node in metrics_snapshot["decode_nodes"]],
    }
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders),
        "ttft": metrics_snapshot["ttft"],
        "certain_kv_leaks": metrics_snapshot["certain_kv_leaks"],
        "suspected_kv_leaks": metrics_snapshot["suspected_kv_leaks"],
        "prefill_disconnect_drain": metrics_snapshot["prefill_disconnect_drain"],
        "decoder_observability": metrics_snapshot["decoder_observability"],
        "decoder_taint": metrics_snapshot["decoder_taint"],
        "healthcheck": metrics_snapshot["healthcheck"],
        "recent_backend_success": recent_backend_success,
    }


@app.get("/metrics")
async def metrics():
    return stats_collector.snapshot(proxy_state.prefillers, proxy_state.decoders)


@app.post("/instances/add")
async def handle_add_instances(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instances/remove")
async def handle_remove_instances(request: Request):
    return await _handle_adjust_instances("remove", request)


@app.post("/instance/add")
async def handle_add_instance_compat(request: Request):
    return await _handle_adjust_instances("add", request)


@app.post("/instance/remove")
async def handle_remove_instance_compat(request: Request):
    return await _handle_adjust_instances("remove", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port, access_log=False)
