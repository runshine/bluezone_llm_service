import asyncio
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path

import httpx
from starlette.requests import Request


SPEC = importlib.util.spec_from_file_location(
    "load_balance_proxy_server",
    Path(__file__).with_name("load_balance_proxy_server.py"),
)
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


def run(coro):
    return asyncio.run(coro)


async def _collect(iterator):
    return [item async for item in iterator]


def _request(path: str, body: bytes) -> Request:
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http", "method": "POST", "path": path,
            "raw_path": path.encode(), "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8077),
            "scheme": "http", "http_version": "1.1",
        },
        receive,
    )


def test_audit_invalid_json_is_indexed_and_archived(tmp_path):
    old = (PROXY.REQUEST_AUDIT_TRACE_DIR, PROXY.REQUEST_AUDIT_TRACE,
           PROXY.REQUEST_AUDIT_LISTENER_PORT, PROXY.REQUEST_AUDIT_SERVICE_ID)
    try:
        PROXY.REQUEST_AUDIT_TRACE_DIR = str(tmp_path)
        PROXY.REQUEST_AUDIT_TRACE = True
        PROXY.REQUEST_AUDIT_LISTENER_PORT = "8077"
        PROXY.REQUEST_AUDIT_SERVICE_ID = "test-8077"
        response = run(PROXY.handle_responses_route(_request("/v1/responses", b"{bad")))
        assert response.status_code == 400
        audit_files = list((tmp_path / "audit").rglob("events.jsonl"))
        assert len(audit_files) == 1
        events = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
        assert events[0]["event"] == "received"
        assert events[-1]["event"] == "terminal"
        assert events[-1]["final_state"] == "rejected"
        archived = list((tmp_path / "payload").rglob("rejected/req_*"))
        assert len(archived) == 1
        assert (archived[0] / "request.raw.bin").read_bytes() == b"{bad"
        assert (archived[0] / "terminal.json").is_file()
    finally:
        (PROXY.REQUEST_AUDIT_TRACE_DIR, PROXY.REQUEST_AUDIT_TRACE,
         PROXY.REQUEST_AUDIT_LISTENER_PORT, PROXY.REQUEST_AUDIT_SERVICE_ID) = old


def test_audit_stream_completion_has_one_terminal_event(tmp_path):
    old = (PROXY.REQUEST_AUDIT_TRACE_DIR, PROXY.REQUEST_AUDIT_TRACE,
           PROXY.REQUEST_AUDIT_LISTENER_PORT)
    try:
        PROXY.REQUEST_AUDIT_TRACE_DIR = str(tmp_path)
        PROXY.REQUEST_AUDIT_TRACE = True
        PROXY.REQUEST_AUDIT_LISTENER_PORT = "8077"
        audit = PROXY.RequestAuditTrace(_request("/v1/chat/completions", b"{}"), "chat_completions")

        async def source():
            yield b"data: ok\n\n"

        assert run(_collect(audit.wrap_stream(source()))) == [b"data: ok\n\n"]
        event_file = next((tmp_path / "audit").rglob("events.jsonl"))
        events = [json.loads(line) for line in event_file.read_text().splitlines()]
        assert [event["event"] for event in events].count("terminal") == 1
        assert events[-1]["final_state"] == "accepted"
    finally:
        (PROXY.REQUEST_AUDIT_TRACE_DIR, PROXY.REQUEST_AUDIT_TRACE,
         PROXY.REQUEST_AUDIT_LISTENER_PORT) = old


def test_responses_client_sse_capture_is_exact_and_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_TRACE", True)
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_PAYLOAD_MODE", "all")
    chunks = [b'data: {"type":"response.created"}\n\n',
              'data: {"type":"response.output_text.delta","delta":"中文｜DSML｜"}\n\n'.encode(),
              b'data: {"type":"response.completed"}\n\ndata: [DONE]\n\n']

    async def source():
        for chunk in chunks:
            yield chunk

    for enabled, api_kind in [(True, "responses"), (False, "responses"), (True, "chat_completions")]:
        monkeypatch.setattr(PROXY, "RESPONSES_SSE_TRACE", enabled)
        audit = PROXY.RequestAuditTrace(_request("/v1/responses", b"{}"), api_kind)
        assert run(_collect(audit.wrap_stream(source()))) == chunks
        path = Path(audit.directory) / "response.client.sse"
        if enabled and api_kind == "responses":
            assert path.read_bytes() == b"".join(chunks)
            assert path.stat().st_mode & 0o777 == 0o600
            assert "response.client.sse" in json.loads(
                (Path(audit.directory) / "manifest.json").read_text()
            )["payload_files"]
        else:
            assert not path.exists()


def test_responses_client_sse_capture_failure_preserves_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_TRACE_DIR", str(tmp_path))
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_TRACE", True)
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_PAYLOAD_MODE", "all")
    monkeypatch.setattr(PROXY, "RESPONSES_SSE_TRACE", True)
    audit = PROXY.RequestAuditTrace(_request("/v1/responses", b"{}"), "responses")
    (Path(audit.directory) / "response.client.sse").mkdir()

    async def source():
        yield b'data: {"type":"response.completed"}\n\n'

    assert run(_collect(audit.wrap_stream(source()))) == [
        b'data: {"type":"response.completed"}\n\n'
    ]
    assert audit._write_failed
    assert audit.fields["final_state"] == "accepted"


def test_decoder_stream_incomplete_is_classified_separately():
    error = PROXY.DecoderStreamIncompleteError(
        "Decoder SSE stream ended without data: [DONE]."
    )

    assert PROXY.backend_error_category(error) == "decoder_stream_incomplete"


def test_first_output_timeout_does_not_taint_decoder():
    assert not PROXY.should_taint_decoder_for_error(
        PROXY.PdFirstOutputTimeoutError("deadline exceeded")
    )
    assert not PROXY.should_taint_decoder_for_error(
        PROXY.DecoderStartProtocolError("protocol retry budget exhausted")
    )
    assert not PROXY.should_taint_decoder_for_error(
        httpx.HTTPStatusError(
            "client error",
            request=httpx.Request("POST", "http://decoder.test/v1/chat/completions"),
            response=httpx.Response(400),
        )
    )
    assert PROXY.should_taint_decoder_for_error(
        httpx.ConnectError("decoder unavailable")
    )


def test_decoder_failure_window_thresholds_and_5xx_classification():
    previous_state = PROXY.proxy_state
    previous_args = getattr(PROXY, "global_args", None)
    state = PROXY.ProxyState([("prefill", 6700)], [("decode", 6721)])
    decoder = state.decoders[0]
    PROXY.proxy_state = state
    PROXY.global_args = SimpleNamespace(
        decoder_taint_enabled=True,
        decoder_taint_failure_threshold=3,
        decoder_taint_failure_window_seconds=60.0,
        decoder_taint_http_5xx_threshold=2,
        decoder_taint_reset_on_success=True,
    )
    error = httpx.HTTPStatusError(
        "backend unavailable",
        request=httpx.Request("POST", "http://decoder.test/v1/chat/completions"),
        response=httpx.Response(503),
    )
    try:
        assert PROXY.classify_decoder_failure(error) == "http_5xx"
        assert PROXY.should_taint_decoder_for_error(error, decoder)[2] is False
        assert PROXY.should_taint_decoder_for_error(error, decoder)[2] is True
        metrics = state.snapshot_decoder_taint_metrics()
        assert metrics["decoder_failure_total"][0]["category"] == "http_5xx"
        assert metrics["decoder_taint_suppressed_total"][0]["value"] == 1
    finally:
        PROXY.proxy_state = previous_state
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args
        for server in [*state.prefillers, *state.decoders]:
            run(server.client.aclose())
            run(server.prefill_client.aclose())


def test_nonstream_decoder_request_has_a_total_deadline():
    async def request_handler(request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json={"choices": []}, request=request)

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        try:
            await PROXY.send_decode_request_with_retry(
                client,
                "/chat/completions",
                {},
                request_id="req-nonstream-timeout",
                max_retries=3,
                base_delay=0,
            )
        finally:
            await client.aclose()

    previous_timeout = PROXY.NONSTREAM_DECODER_TIMEOUT_SECONDS
    PROXY.NONSTREAM_DECODER_TIMEOUT_SECONDS = 0.01
    try:
        try:
            run(execute())
            assert False, "expected non-stream decoder deadline to fail"
        except TimeoutError:
            pass
    finally:
        PROXY.NONSTREAM_DECODER_TIMEOUT_SECONDS = previous_timeout


def test_decoder_start_protocol_marker_is_only_detected_before_real_output():
    assert PROXY.DECODER_START_PROTOCOL_MARKERS == {
        "begin\u2581of\u2581sentence",
        "<\uff5cbegin\u2581of\u2581sentence\uff5c>",
    }
    for marker_text in PROXY.DECODER_START_PROTOCOL_MARKERS:
        marker = PROXY.StreamChoiceInfo(reasoning=marker_text)
        assert PROXY.is_decoder_start_protocol_leak(marker, False)
        assert not PROXY.is_decoder_start_protocol_leak(marker, True)
        assert not PROXY.is_decoder_start_protocol_leak(
            PROXY.StreamChoiceInfo(reasoning=f"{marker_text} and normal reasoning"),
            False,
        )
        assert not PROXY.is_decoder_start_protocol_leak(
            PROXY.StreamChoiceInfo(reasoning=marker_text, text="unexpected text"),
            False,
        )
    assert PROXY.backend_error_category(PROXY.DecoderStartProtocolError("exhausted")) == (
        "decoder_start_protocol_recompute_exhausted"
    )


def test_client_disconnect_only_drains_while_prefill_is_pending():
    previous_stats = PROXY.stats_collector
    PROXY.stats_collector = PROXY.StatsCollector()
    try:
        prefill = PROXY.RequestLeakState(request_id="prefill-disconnect")
        PROXY.mark_client_disconnected(prefill)
        assert prefill.client_disconnected is True
        assert prefill.drain_after_prefill is True
        assert prefill.prefill_disconnect_draining is True
        assert prefill.disconnect_phase == "prefill_pending"

        decoder = PROXY.RequestLeakState(
            request_id="decoder-disconnect",
            pipeline_phase="decoder_running",
        )
        PROXY.mark_client_disconnected(decoder)
        assert decoder.client_disconnected is True
        assert decoder.drain_after_prefill is False
        assert decoder.prefill_disconnect_draining is False
        assert decoder.disconnect_phase == "decoder_running"
    finally:
        PROXY.stats_collector = previous_stats


def test_first_decoder_semantic_output_ignores_protocol_framing_and_marker():
    assert not PROXY._decoder_chunk_has_semantic_output(b"\n\n")
    assert not PROXY._decoder_chunk_has_semantic_output(b"data: [DONE]\n\n")
    marker = next(iter(PROXY.DECODER_START_PROTOCOL_MARKERS))
    marker_chunk = json.dumps({"choices": [{"delta": {"reasoning_content": marker}}]})
    assert not PROXY._decoder_chunk_has_semantic_output(
        f"data: {marker_chunk}\n\n".encode()
    )
    content_chunk = json.dumps({"choices": [{"delta": {"content": "ready"}}]})
    assert PROXY._decoder_chunk_has_semantic_output(
        f"data: {content_chunk}\n\n".encode()
    )


def test_prefill_disconnect_drain_stops_decoder_after_first_semantic_output():
    class Response:
        def json(self):
            return {"kv_transfer_params": {"remote_engine_id": "engine"}}

    class Node:
        def __init__(self, name):
            self.name = name
            self.client = object()

        def __str__(self):
            return self.name

    class State:
        def __init__(self, decoder):
            self.decoder = decoder
            self.events = []

        def release_prefiller(self, *args):
            self.events.append("release_prefiller")

        def release_prefiller_kv(self, *args):
            self.events.append("release_prefiller_kv")

        def select_decoder(self):
            self.events.append("select_decoder")
            return self.decoder

        def release_decoder(self, *args):
            self.events.append("release_decoder")

        def mark_backend_success(self, *args):
            self.events.append("prefill_success")

    async def execute():
        previous_state = PROXY.proxy_state
        previous_stream = PROXY.stream_service_response_with_retry
        previous_args = getattr(PROXY, "global_args", None)
        decoder = Node("decoder.test:6721")
        state = State(decoder)
        stream_closed = asyncio.Event()

        async def fake_stream(*args, **kwargs):
            try:
                payload = json.dumps({"choices": [{"delta": {"content": "ready"}}]})
                yield f"data: {payload}\n\n".encode()
                assert False, "drain must stop after the first semantic output"
            finally:
                stream_closed.set()

        PROXY.proxy_state = state
        PROXY.stream_service_response_with_retry = fake_stream
        PROXY.global_args = SimpleNamespace(max_retries=1, retry_delay=0)
        leak_state = PROXY.RequestLeakState(
            request_id="prefill-disconnect",
            client_disconnected=True,
            drain_after_prefill=True,
            prefill_disconnect_draining=True,
        )
        try:
            prefill_task = asyncio.create_task(asyncio.sleep(0, result=Response()))
            await PROXY.drain_prefill_after_client_disconnect(
                prefill_task,
                Node("prefill.test:6700"),
                1.0,
                "/chat/completions",
                {},
                leak_state.request_id,
                None,
                leak_state,
            )
            assert leak_state.first_decoder_output_received is True
            assert leak_state.drain_stop_reason == "first_decoder_output"
            assert leak_state.pipeline_phase == "terminal"
            assert stream_closed.is_set()
            assert state.events == [
                "release_prefiller",
                "prefill_success",
                "select_decoder",
                "release_prefiller_kv",
                "release_decoder",
            ]
        finally:
            PROXY.proxy_state = previous_state
            PROXY.stream_service_response_with_retry = previous_stream
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

    run(execute())


def test_decoder_start_protocol_marker_can_span_reasoning_sse_chunks():
    for marker in PROXY.DECODER_START_PROTOCOL_MARKERS:
        split_at = len(marker) // 2
        first = PROXY.StreamChoiceInfo(reasoning=marker[:split_at])
        second = PROXY.StreamChoiceInfo(reasoning=marker[split_at:])

        assert PROXY.decoder_start_protocol_marker_state("", first, False) == "prefix"
        assert (
            PROXY.decoder_start_protocol_marker_state(first.reasoning, second, False)
            == "marker"
        )


def test_decoder_start_protocol_marker_prefix_does_not_hide_normal_reasoning():
    prefix = PROXY.StreamChoiceInfo(reasoning="begin\u2581of\u2581")
    normal_continuation = PROXY.StreamChoiceInfo(reasoning="a normal explanation")

    assert PROXY.decoder_start_protocol_marker_state("", prefix, False) == "prefix"
    assert (
        PROXY.decoder_start_protocol_marker_state(
            prefix.reasoning, normal_continuation, False
        )
        == "not_marker"
    )
    assert (
        PROXY.decoder_start_protocol_marker_state(
            "", PROXY.StreamChoiceInfo(reasoning="begin\u2581of\u2581sentence", text="x"), False
        )
        == "not_marker"
    )


def test_decoder_protocol_retry_prompt_is_an_internal_new_user_turn():
    request = {
        "messages": [
            {"role": "user", "content": "original"},
            {"role": "assistant", "content": "tool planning"},
            {"role": "tool", "content": "tool result"},
        ]
    }
    original = json.loads(json.dumps(request))

    assert PROXY.append_decoder_start_protocol_retry_prompt(request)
    assert request["messages"][:-1] == original["messages"]
    assert request["messages"][-1] == {
        "role": "user",
        "content": PROXY.DECODER_START_PROTOCOL_RETRY_PROMPT,
    }


def test_decoder_protocol_retry_prompt_does_not_mutate_original_request_copy_source():
    request = {"messages": [{"role": "user", "content": "original"}]}
    retry_request = json.loads(json.dumps(request))

    assert PROXY.append_decoder_start_protocol_retry_prompt(retry_request)
    assert request == {"messages": [{"role": "user", "content": "original"}]}


def test_decoder_protocol_retry_prompt_can_be_disabled_from_cli():
    original_argv = sys.argv
    try:
        sys.argv = ["load_balance_proxy_server.py", "--no-enable-decoder-start-protocol-retry-prompt"]
        args = PROXY.parse_args()
    finally:
        sys.argv = original_argv
    assert args.enable_decoder_start_protocol_retry_prompt is False


def test_decoder_protocol_retry_prompt_is_enabled_by_default():
    original_argv = sys.argv
    try:
        sys.argv = ["load_balance_proxy_server.py"]
        args = PROXY.parse_args()
    finally:
        sys.argv = original_argv
    assert args.enable_decoder_start_protocol_retry_prompt is True


def test_decoder_start_protocol_recompute_default_allows_one_retry():
    original_argv = sys.argv
    try:
        sys.argv = ["load_balance_proxy_server.py"]
        args = PROXY.parse_args()
    finally:
        sys.argv = original_argv
    assert args.max_decoder_start_protocol_recompute_attempts == 1


def test_decoder_start_protocol_log_reason_is_short_marker_variant():
    assert PROXY.decoder_start_protocol_marker_variant("<wrapped>") == "wrapped"
    assert PROXY.decoder_start_protocol_marker_variant("begin\u2581of\u2581sentence") == "unwrapped"


def test_stats_collector_tracks_recomputes_by_original_nodes():
    collector = PROXY.StatsCollector()

    collector.record_prefill_recompute("prefill.test:6700")
    collector.record_prefill_recompute("prefill.test:6700")
    collector.record_decode_recompute("decode.test:6721")

    assert collector.pf_recomputes == {"prefill.test:6700": 2}
    assert collector.dc_recomputes == {"decode.test:6721": 1}


def test_stats_row_places_recompute_immediately_after_requests():
    collector = PROXY.StatsCollector()

    row = collector._fmt_row(
        "Prefill",
        "prefill.test:6700",
        3,
        10,
        1.0,
        120,
        recomputes=2,
    )

    assert "|     3 |         2 | input=" in row


def test_periodic_stats_prints_and_resets_recompute_counts(capsys):
    collector = PROXY.StatsCollector()
    collector.record_prefill("prefill.test:6700", 10, 1.0)
    collector.record_decode("decode.test:6721", 2, 1.0)
    collector.record_prefill_recompute("prefill.test:6700")
    collector.record_decode_recompute("decode.test:6721")
    collector.record_completed()

    collector.print_and_reset(
        120,
        prefiller_nodes=[SimpleNamespace(host="prefill.test", port=6700)],
        decoder_nodes=[SimpleNamespace(host="decode.test", port=6721)],
    )

    output = capsys.readouterr().out
    assert "|  Reqs | Recompute | Tokens" in output
    assert "|     1 | pf=1/dc=1 | in=10+out=2" in output
    assert collector.pf_recomputes == {}
    assert collector.dc_recomputes == {}
    assert collector.last_completed_period["total_requests"] == 1
    assert collector.last_completed_period["period_id"] == 1


def test_decoder_selection_prioritizes_active_requests_before_period_assignments():
    state = PROXY.ProxyState(
        [("prefill", 6700)],
        [("decode", port) for port in range(6721, 6725)],
    )
    decoders = state.decoders
    decoders[0].active_requests = 1
    decoders[0].period_assigned_requests = 0
    decoders[0].active_tokens = 1
    decoders[1].active_requests = 1
    decoders[1].period_assigned_requests = 0
    decoders[1].active_tokens = 1000000
    decoders[2].active_requests = 0
    decoders[2].period_assigned_requests = 999
    decoders[2].active_tokens = 999999999
    decoders[3].active_requests = 2
    decoders[3].period_assigned_requests = 0
    decoders[3].active_tokens = 0
    with state._lock:
        state.decoder_heap = [
            (state._calc_decoder_priority(server), state._next_counter(), server)
            for server in decoders
        ]
        import heapq
        heapq.heapify(state.decoder_heap)

    selected = state.select_decoder(token_count=10**9)
    assert selected is decoders[2]
    assert selected.active_requests == 1
    assert selected.period_assigned_requests == 1000
    assert selected.active_tokens == 999999999
    assert state._calc_decoder_priority(decoders[1]) == (1.0, 0.0)


def test_decoder_selection_uses_period_assignments_when_active_counts_tie():
    state = PROXY.ProxyState(
        [("prefill", 6700)],
        [("decode", port) for port in range(6721, 6724)],
    )
    decoders = state.decoders
    for decoder in decoders:
        decoder.active_requests = 2
    decoders[0].period_assigned_requests = 7
    decoders[1].period_assigned_requests = 3
    decoders[2].period_assigned_requests = 5
    with state._lock:
        state.decoder_heap = [
            (state._calc_decoder_priority(server), state._next_counter(), server)
            for server in decoders
        ]
        import heapq
        heapq.heapify(state.decoder_heap)

    selected = state.select_decoder()
    assert selected is decoders[1]
    assert selected.active_requests == 3
    assert selected.period_assigned_requests == 4


def test_decoder_selection_excludes_only_the_current_retry_when_possible():
    state = PROXY.ProxyState(
        [("prefill", 6700)],
        [("decode", 6721), ("decode", 6722)],
    )
    first, second = state.decoders

    selected = state.select_decoder(excluded_decoders={first})
    assert selected is second
    assert first not in state.tainted_decoders
    assert second not in state.tainted_decoders


def test_decoder_selection_falls_back_to_excluded_node_when_it_is_only_healthy_node():
    state = PROXY.ProxyState([("prefill", 6700)], [("decode", 6721)])
    decoder = state.decoders[0]

    assert state.select_decoder(excluded_decoders={decoder}) is decoder
    assert decoder not in state.tainted_decoders


def test_decoder_period_assignments_reset_without_releasing_active_requests():
    state = PROXY.ProxyState(
        [("prefill", 6700)],
        [("decode", 6721), ("decode", 6722)],
    )
    first, second = state.decoders
    first.active_requests = 1
    first.period_assigned_requests = 4
    second.active_requests = 0
    second.period_assigned_requests = 9
    with state._lock:
        state.decoder_heap = [
            (state._calc_decoder_priority(server), state._next_counter(), server)
            for server in state.decoders
        ]
        import heapq
        heapq.heapify(state.decoder_heap)

    state.reset_decoder_period_assignments()

    assert first.active_requests == 1
    assert second.active_requests == 0
    assert first.period_assigned_requests == 0
    assert second.period_assigned_requests == 0
    assert state.select_decoder() is second


def test_decoder_release_ignores_token_count():
    state = PROXY.ProxyState(
        [("prefill", 6700)],
        [("decode", 6721)],
    )
    decoder = state.select_decoder(token_count=123456)
    assert decoder.active_requests == 1
    assert decoder.active_tokens == 0
    state.release_decoder(decoder, token_count=987654)
    assert decoder.active_requests == 0
    assert decoder.active_tokens == 0


def test_prefill_payload_isolated_and_valid_utf8_json():
    original = {
        "model": "DeepSeekv4",
        "messages": [{
            "role": "user",
            "content": '中文 "quoted" \\\\ reasoning',
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"key":"value"}',
            },
        }],
        "stream": True,
        "max_tokens": 32,
    }
    original_copy = json.loads(json.dumps(original, ensure_ascii=False))
    prepared = PROXY.prepare_prefill_request_data(original)
    prepared["kv_transfer_params"] = {"do_remote_decode": True}
    prepared["stream"] = False
    prepared["max_tokens"] = 1
    body, body_hash = PROXY._serialize_json_payload(prepared)

    assert original == original_copy
    assert json.loads(body) == prepared
    assert len(body) > 0
    assert body_hash == hashlib.sha256(body).hexdigest()
    assert json.loads(body)["messages"][0]["content"].startswith("中文")


def test_prefill_http_request_has_explicit_body_length_and_close():
    captured = {}

    async def request_handler(request):
        body = await request.aread()
        captured["body"] = body
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={"kv_transfer_params": {"remote_engine_id": "engine-0"}},
            request=request,
        )

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://prefill.test/v1",
        )
        server = type("Prefiller", (), {"url": "http://prefill.test/v1", "__str__": lambda self: "prefill.test:6700"})()
        try:
            original = {
                "model": "DeepSeekv4",
                "messages": [{"role": "user", "content": "中文、引号：\"ok\""}],
                "stream": True,
                "max_tokens": 32,
            }
            response = await PROXY.send_request_to_service(
                client,
                server,
                "/chat/completions",
                original,
                "req-test-1",
                max_retries=1,
            )
            return response
        finally:
            await client.aclose()

    response = run(execute())
    assert response.status_code == 200
    assert captured["headers"]["content-type"] == "application/json"
    assert int(captured["headers"]["content-length"]) == len(captured["body"])
    assert captured["headers"]["connection"] == "close"
    payload = json.loads(captured["body"])
    assert payload["messages"][0]["content"] == "中文、引号：\"ok\""
    assert payload["max_tokens"] == 1
    assert payload["stream"] is False
    assert payload["kv_transfer_params"]["do_remote_decode"] is True


def test_invalid_client_json_returns_400():
    class Request:
        client = type("Client", (), {"host": "127.0.0.1", "port": 1234})()

        async def body(self):
            return b'{"messages":"unterminated}'

    PROXY.global_args = type(
        "Args",
        (),
        {"max_input_chars": 0},
    )()
    class ProxyState:
        def inc_request_num(self):
            pass

        def dec_request_num(self):
            pass

    PROXY.proxy_state = ProxyState()
    response = run(PROXY._handle_completions("/chat/completions", Request()))
    assert response.status_code == 400
    payload = json.loads(response.body)
    assert payload["code"] == "client_invalid_json"


def _set_failure_capture_context(request_id="req-capture-1"):
    return PROXY._active_failure_capture_context.set(
        PROXY.FailureCaptureContext(
            client_request_id="client-capture-1",
            response_id="resp_capture_1",
            api="/responses",
            request_kind="api=response|stream=true",
            client_request_body=b'{"model":"DeepSeekv4","input":"client original"}',
            normalized_backend_request={
                "model": "DeepSeekv4",
                "messages": [{"role": "user", "content": "client original"}],
            },
            received_at="2026-08-27 10:00:00",
        )
    )


def test_successful_prefill_does_not_create_failure_capture():
    async def request_handler(request):
        return httpx.Response(200, json={"kv_transfer_params": {}}, request=request)

    async def execute(capture_dir):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://prefill.test/v1",
        )
        server = type("Prefiller", (), {"url": "http://prefill.test/v1", "__str__": lambda self: "prefill.test:6700"})()
        try:
            return await PROXY.send_request_to_service(
                client,
                server,
                "/chat/completions",
                {"model": "DeepSeekv4", "messages": [{"role": "user", "content": "ok"}]},
                "req-success-no-capture",
                max_retries=1,
            )
        finally:
            await client.aclose()

    previous_enabled = PROXY.PREFILL_FAILURE_CAPTURE
    previous_dir = PROXY.PREFILL_FAILURE_CAPTURE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.PREFILL_FAILURE_CAPTURE = True
        PROXY.PREFILL_FAILURE_CAPTURE_DIR = temp_dir
        token = _set_failure_capture_context()
        try:
            response = run(execute(temp_dir))
        finally:
            PROXY._active_failure_capture_context.reset(token)
            PROXY.PREFILL_FAILURE_CAPTURE = previous_enabled
            PROXY.PREFILL_FAILURE_CAPTURE_DIR = previous_dir
        assert response.status_code == 200
        assert list(Path(temp_dir).iterdir()) == []


def test_prefill_400_writes_replayable_failure_capture_without_secret():
    async def request_handler(request):
        return httpx.Response(
            400,
            json={"error": {"message": "Unterminated string starting at: line 1 column 9 (char 8)"}},
            request=request,
        )

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://prefill.test/v1",
        )
        server = type("Prefiller", (), {"url": "http://prefill.test/v1", "__str__": lambda self: "prefill.test:6700"})()
        try:
            await PROXY.send_request_to_service(
                client,
                server,
                "/chat/completions",
                {"model": "DeepSeekv4", "messages": [{"role": "user", "content": "中文 \\\"quoted\\\""}]},
                "req-capture-400",
                max_retries=1,
            )
        finally:
            await client.aclose()

    previous_enabled = PROXY.PREFILL_FAILURE_CAPTURE
    previous_dir = PROXY.PREFILL_FAILURE_CAPTURE_DIR
    previous_key = os.environ.get("OPENAI_API_KEY")
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.PREFILL_FAILURE_CAPTURE = True
        PROXY.PREFILL_FAILURE_CAPTURE_DIR = temp_dir
        os.environ["OPENAI_API_KEY"] = "test-secret"
        token = _set_failure_capture_context()
        try:
            try:
                run(execute())
                assert False, "Expected Prefill 400"
            except httpx.HTTPStatusError:
                pass
        finally:
            PROXY._active_failure_capture_context.reset(token)
            PROXY.PREFILL_FAILURE_CAPTURE = previous_enabled
            PROXY.PREFILL_FAILURE_CAPTURE_DIR = previous_dir
            if previous_key is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = previous_key

        request_dir = Path(temp_dir) / "400" / "req-capture-400"
        attempt_dir = request_dir / "attempt-001"
        manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
        request_meta = json.loads((attempt_dir / "prefill.request.meta.json").read_text(encoding="utf-8"))
        body = (attempt_dir / "prefill.request.body").read_bytes()
        assert manifest["failure_kind"] == "prefill_invalid_json"
        assert manifest["attempts"] == ["attempt-001"]
        assert (request_dir / "client.request.body").read_bytes().startswith(b'{"model"')
        assert json.loads((request_dir / "backend.request.json").read_text(encoding="utf-8"))["messages"]
        assert len(body) == request_meta["content_length"]
        assert hashlib.sha256(body).hexdigest() == request_meta["body_sha256"]
        assert json.loads(body)["messages"][0]["content"].startswith("中文")
        saved_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in request_dir.rglob("*") if path.is_file())
        assert "test-secret" not in saved_text
        assert "--data-binary @prefill.request.body" in (attempt_dir / "replay.sh").read_text(encoding="utf-8")


def test_prefill_timeout_writes_transport_failure_capture():
    async def request_handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://prefill.test/v1",
        )
        server = type("Prefiller", (), {"url": "http://prefill.test/v1", "__str__": lambda self: "prefill.test:6700"})()
        try:
            await PROXY.send_request_to_service(
                client,
                server,
                "/chat/completions",
                {"model": "DeepSeekv4", "messages": [{"role": "user", "content": "timeout"}]},
                "req-capture-timeout",
                max_retries=1,
            )
        finally:
            await client.aclose()

    previous_enabled = PROXY.PREFILL_FAILURE_CAPTURE
    previous_dir = PROXY.PREFILL_FAILURE_CAPTURE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.PREFILL_FAILURE_CAPTURE = True
        PROXY.PREFILL_FAILURE_CAPTURE_DIR = temp_dir
        token = _set_failure_capture_context()
        try:
            try:
                run(execute())
                assert False, "Expected Prefill timeout"
            except httpx.ReadTimeout:
                pass
        finally:
            PROXY._active_failure_capture_context.reset(token)
            PROXY.PREFILL_FAILURE_CAPTURE = previous_enabled
            PROXY.PREFILL_FAILURE_CAPTURE_DIR = previous_dir
        manifest = json.loads(
            (Path(temp_dir) / "transport" / "req-capture-timeout" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["failure_kind"] == "prefill_transport_readtimeout"


def test_prefill_retry_writes_distinct_failure_attempts():
    async def request_handler(request):
        return httpx.Response(500, json={"error": "temporary failure"}, request=request)

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://prefill.test/v1",
        )
        server = type("Prefiller", (), {"url": "http://prefill.test/v1", "__str__": lambda self: "prefill.test:6700"})()
        try:
            await PROXY.send_request_to_service(
                client,
                server,
                "/chat/completions",
                {"model": "DeepSeekv4", "messages": [{"role": "user", "content": "retry"}]},
                "req-capture-retry",
                max_retries=2,
                base_delay=0,
            )
        finally:
            await client.aclose()

    previous_enabled = PROXY.PREFILL_FAILURE_CAPTURE
    previous_dir = PROXY.PREFILL_FAILURE_CAPTURE_DIR
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.PREFILL_FAILURE_CAPTURE = True
        PROXY.PREFILL_FAILURE_CAPTURE_DIR = temp_dir
        token = _set_failure_capture_context()
        try:
            try:
                run(execute())
                assert False, "Expected Prefill 500"
            except httpx.HTTPStatusError:
                pass
        finally:
            PROXY._active_failure_capture_context.reset(token)
            PROXY.PREFILL_FAILURE_CAPTURE = previous_enabled
            PROXY.PREFILL_FAILURE_CAPTURE_DIR = previous_dir
        request_dir = Path(temp_dir) / "500" / "req-capture-retry"
        manifest = json.loads((request_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["attempts"] == ["attempt-001", "attempt-002"]
        assert (request_dir / "attempt-001" / "prefill.request.body").is_file()
        assert (request_dir / "attempt-002" / "prefill.request.body").is_file()


def test_removed_background_drain_arguments_are_not_accepted():
    original_argv = sys.argv
    try:
        sys.argv = ["load_balance_proxy_server.py"]
        args = PROXY.parse_args()
    finally:
        sys.argv = original_argv
    assert not hasattr(args, "max_background_drain_seconds")
    assert not hasattr(args, "enable_background_drain")
    assert args.healthcheck_success_ttl_seconds == 60.0
    assert args.enable_periodic_backend_healthcheck is False
    assert args.decoder_first_response_observe_seconds == 30.0
    assert args.decoder_first_response_log_interval_seconds == 60.0


def test_periodic_backend_healthcheck_requires_explicit_startup_flag():
    original_argv = sys.argv
    try:
        sys.argv = ["load_balance_proxy_server.py", "--enable-periodic-backend-healthcheck"]
        args = PROXY.parse_args()
    finally:
        sys.argv = original_argv
    assert args.enable_periodic_backend_healthcheck is True


def test_tainted_backend_recovery_runs_when_periodic_probe_is_disabled():
    pf = PROXY.ServerState("recovery-prefill", 6700)
    dc = PROXY.ServerState("recovery-decode", 6721)
    try:
        assert PROXY.NodeListener._should_run_backend_checks(
            False, [pf], [], set(), set()
        ) is True
        assert PROXY.NodeListener._should_run_backend_checks(
            False, [], [dc], set(), set()
        ) is True
    finally:
        run(pf.client.aclose())
        run(pf.prefill_client.aclose())
        run(dc.client.aclose())
        run(dc.prefill_client.aclose())


def test_tainted_backend_recovery_skips_pending_removals_when_periodic_disabled():
    pf = PROXY.ServerState("pending-prefill", 6700)
    dc = PROXY.ServerState("pending-decode", 6721)
    try:
        assert PROXY.NodeListener._should_run_backend_checks(
            False, [pf], [dc], {pf}, {dc}
        ) is False
        # A recoverable node still enables the loop when another node is pending removal.
        assert PROXY.NodeListener._should_run_backend_checks(
            False, [pf], [dc], {pf}, set()
        ) is True
    finally:
        run(pf.client.aclose())
        run(pf.prefill_client.aclose())
        run(dc.client.aclose())
        run(dc.prefill_client.aclose())


def _health_listener_for_test(server):
    listener = object.__new__(PROXY.NodeListener)
    listener.proxy_state = type("Proxy", (), {"_lock": PROXY.threading.RLock()})()
    listener.health_stats = {}
    listener._stats_lock = PROXY.threading.Lock()
    return listener


def test_recent_business_success_skips_active_health_probe_until_ttl_expires():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        server = PROXY.ServerState("health-test", 6700)
        listener = _health_listener_for_test(server)
        PROXY.global_args = SimpleNamespace(healthcheck_success_ttl_seconds=60.0)
        server.last_success_at = PROXY.time.time() - 59.0
        assert listener._has_recent_business_success(server) is True
        server.last_success_at = PROXY.time.time() - 61.0
        assert listener._has_recent_business_success(server) is False
        PROXY.global_args = SimpleNamespace(healthcheck_success_ttl_seconds=0.0)
        server.last_success_at = PROXY.time.time()
        assert listener._has_recent_business_success(server) is False
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args
        run(server.client.aclose())
        run(server.prefill_client.aclose())


def test_healthcheck_summary_tracks_skips_probes_and_business_successes():
    listener = _health_listener_for_test(None)
    listener._record_health_probe("prefill.test:6700", "idle", True)
    listener._record_health_probe("decode.test:6721", "tainted", False)
    listener._record_health_probe("waiting.test:6722", "waiting", True)
    listener._record_health_skip("prefill.test:6700")
    listener.record_business_success("prefill.test:6700", "prefill_completed")
    listener.record_business_success("decode.test:6721", "decode_completed")
    summary = listener.get_healthcheck_summary()
    assert summary["probed_idle"] == 1
    assert summary["probed_tainted"] == 1
    assert summary["probed_waiting"] == 1
    assert summary["skipped_recent_success"] == 1
    assert summary["business_success"] == 2
    assert summary["business_success_by_kind"] == {
        "prefill_completed": 1,
        "decode_completed": 1,
    }


def test_backend_success_snapshot_is_per_node_and_reports_source():
    server = PROXY.ServerState("success-test", 6721)
    recorded = []
    state = object.__new__(PROXY.ProxyState)
    state._lock = PROXY.threading.RLock()
    state.prefillers = []
    state.decoders = [server]
    state.node_listener = type(
        "Listener", (), {"record_business_success": lambda _, key, kind: recorded.append((key, kind))}
    )()
    try:
        state.mark_backend_success(server, "decode_completed")
        snapshot = state.snapshot_backend_success()
        assert recorded == [("success-test:6721", "decode_completed")]
        assert snapshot["success-test:6721"]["last_success_kind"] == "decode_completed"
        assert snapshot["success-test:6721"]["last_success_age_seconds"] is not None
    finally:
        run(server.client.aclose())
        run(server.prefill_client.aclose())


def test_prefill_disconnect_drain_is_idempotent_and_has_no_deadline_watchdog():
    previous_stats = PROXY.stats_collector
    PROXY.stats_collector = PROXY.StatsCollector()
    try:
        leak_state = PROXY.RequestLeakState(request_id="req-prefill-disconnect")
        PROXY.mark_client_disconnected(leak_state)
        PROXY.mark_client_disconnected(leak_state)
        active = PROXY.stats_collector._kv_leak_snapshot_locked()["prefill_disconnect_drain"]
        assert active["active"] == 1
        assert active["active_by_decoder"] == {"unknown": 1}

        PROXY.finish_prefill_disconnect_drain(leak_state)
        PROXY.finish_prefill_disconnect_drain(leak_state)
        finished = PROXY.stats_collector._kv_leak_snapshot_locked()["prefill_disconnect_drain"]
        assert finished["active"] == 0
        assert finished["finished_total"] == 1
        assert finished["failed_total"] == 0
    finally:
        PROXY.stats_collector = previous_stats


def test_backend_model_name_rewrites_only_the_internal_backend_request():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = SimpleNamespace(backend_model_name="DeepSeekv4")
        request = {
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "hello"}],
        }
        normalized = PROXY.prepare_backend_request_data(request)
        assert request["model"] == "gpt-5.6-luna"
        assert normalized["model"] == "DeepSeekv4"
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_backend_model_name_is_optional():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = SimpleNamespace(backend_model_name=None)
        request = {"model": "gpt-5.6-luna"}
        assert PROXY.prepare_backend_request_data(request)["model"] == "gpt-5.6-luna"
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_decoder_first_response_observability_snapshot_tracks_active_attempts():
    collector = PROXY.StatsCollector()
    started_at = PROXY.time.time() - 2.0
    collector.record_decoder_attempt_started()
    collector.start_decoder_first_response_waiting(
        "req-1:decode-1:transport-1",
        "decode.test:6721",
        "waiting_response_headers",
        started_at,
    )
    collector.record_decoder_first_response_waiting(
        "req-1:decode-1:transport-1",
        "decode.test:6721",
        "waiting_first_sse",
        started_at,
    )
    snapshot = collector._decoder_observability_snapshot_locked()
    assert snapshot["attempts_started"] == 1
    assert snapshot["first_response_waiting_total"] == 1
    assert snapshot["first_response_waiting_active"] == 1
    assert snapshot["waiting_by_node"] == {"decode.test:6721": 1}
    assert snapshot["active_attempts"][0]["phase"] == "waiting_first_sse"

    collector.record_decoder_first_sse("req-1:decode-1:transport-1")
    collector.record_decoder_stream_eof_without_done()
    completed = collector._decoder_observability_snapshot_locked()
    assert completed["first_sse_received"] == 1
    assert completed["first_response_waiting_active"] == 1
    collector.record_decoder_first_valid_output("req-1:decode-1:transport-1")
    assert collector._decoder_observability_snapshot_locked()["first_response_waiting_active"] == 0
    assert completed["stream_eof_without_done_total"] == 1


def test_decoder_attempt_trace_records_safe_transport_stages():
    async def request_handler(request):
        return httpx.Response(
            200,
            content=b'data: {"id":"safe"}\\n\\ndata: [DONE]\\n\\n',
            request=request,
        )

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        try:
            chunks = []
            async for chunk in PROXY.stream_service_response_with_retry(
                client,
                "/chat/completions",
                {"secret_prompt": "must-not-appear-in-attempt-trace"},
                request_id="req-observe",
                decoder_node="decoder.test:6721",
                decoder_attempt=1,
                max_retries=1,
                base_delay=0,
            ):
                chunks.append(chunk)
            return chunks
        finally:
            await client.aclose()

    previous_enabled = PROXY.RESPONSES_WIRE_TRACE
    previous_dir = PROXY.RESPONSES_WIRE_TRACE_DIR
    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.RESPONSES_WIRE_TRACE = True
        PROXY.RESPONSES_WIRE_TRACE_DIR = temp_dir
        PROXY.stats_collector = PROXY.StatsCollector()
        PROXY.global_args = SimpleNamespace(
            decoder_first_response_observe_seconds=0,
            decoder_first_response_log_interval_seconds=1,
        )
        trace = PROXY.WireTrace("resp-observe")
        token = PROXY._active_wire_trace.set(trace)
        try:
            chunks = run(execute())
        finally:
            PROXY._active_wire_trace.reset(token)
            PROXY.RESPONSES_WIRE_TRACE = previous_enabled
            PROXY.RESPONSES_WIRE_TRACE_DIR = previous_dir
            PROXY.stats_collector = previous_stats
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

        assert b"".join(chunks).endswith(b"data: [DONE]\\n\\n")
        trace_text = (Path(temp_dir) / "resp-observe" / "decode.attempts.jsonl").read_text(
            encoding="utf-8"
        )
        records = [json.loads(line) for line in trace_text.splitlines()]
        assert [record["event"] for record in records] == [
            "started",
            "response_headers_received",
            "first_sse_chunk_received",
            "stream_eof",
        ]
        assert records[0]["decoder_node"] == "decoder.test:6721"
        assert records[0]["request_body_bytes"] > 0
        assert len(records[0]["request_body_sha256"]) == 64
        assert "must-not-appear-in-attempt-trace" not in trace_text


def test_decoder_stream_progress_records_submit_to_first_valid_output():
    async def request_handler(request):
        return httpx.Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ready"}}]}\n\n',
            request=request,
        )

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        progress = PROXY.DecoderStreamProgress()
        try:
            async for _ in PROXY.stream_service_response_with_retry(
                client,
                "/chat/completions",
                {},
                request_id="req-first-token-delay",
                decoder_node="decoder.test:6721",
                decoder_attempt=1,
                stream_progress=progress,
                max_retries=1,
                base_delay=0,
            ):
                progress.observe_valid_output()
            return progress
        finally:
            await client.aclose()

    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    PROXY.stats_collector = PROXY.StatsCollector()
    PROXY.global_args = SimpleNamespace(
        decoder_first_response_observe_seconds=0,
        decoder_first_response_log_interval_seconds=1,
    )
    try:
        progress = run(execute())
    finally:
        PROXY.stats_collector = previous_stats
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args

    assert progress.first_valid_output_at is not None
    assert progress.first_valid_output_delay is not None
    assert progress.first_valid_output_delay >= 0


def test_decoder_first_response_watchdog_marks_waiting_first_sse_and_cleans_up():
    class DelayedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.03)
            yield b"data: [DONE]\\n\\n"

        async def aclose(self):
            return None

    async def request_handler(request):
        return httpx.Response(200, stream=DelayedStream(), request=request)

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        try:
            return [
                chunk
                async for chunk in PROXY.stream_service_response_with_retry(
                    client,
                    "/chat/completions",
                    {},
                    request_id="req-watchdog",
                    decoder_node="decoder.test:6722",
                    decoder_attempt=2,
                    max_retries=1,
                    base_delay=0,
                )
            ]
        finally:
            await client.aclose()

    previous_enabled = PROXY.RESPONSES_WIRE_TRACE
    previous_dir = PROXY.RESPONSES_WIRE_TRACE_DIR
    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.RESPONSES_WIRE_TRACE = True
        PROXY.RESPONSES_WIRE_TRACE_DIR = temp_dir
        PROXY.stats_collector = PROXY.StatsCollector()
        PROXY.global_args = SimpleNamespace(
            decoder_first_response_observe_seconds=0.005,
            decoder_first_response_log_interval_seconds=0.005,
        )
        trace = PROXY.WireTrace("resp-watchdog")
        token = PROXY._active_wire_trace.set(trace)
        try:
            assert run(execute()) == [b"data: [DONE]\\n\\n"]
            snapshot = PROXY.stats_collector._decoder_observability_snapshot_locked()
        finally:
            PROXY._active_wire_trace.reset(token)
            PROXY.RESPONSES_WIRE_TRACE = previous_enabled
            PROXY.RESPONSES_WIRE_TRACE_DIR = previous_dir
            PROXY.stats_collector = previous_stats
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

        records = [
            json.loads(line)
            for line in (Path(temp_dir) / "resp-watchdog" / "decode.attempts.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        waiting = [record for record in records if record["event"] == "first_response_waiting"]
        assert waiting and waiting[0]["phase"] == "waiting_first_sse"
        assert snapshot["first_response_waiting_total"] == 1
        assert snapshot["first_response_waiting_active"] == 0


def test_pd_first_output_deadline_times_out_before_decoder_sse_and_cleans_up():
    class DelayedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            return None

    async def request_handler(request):
        return httpx.Response(200, stream=DelayedStream(), request=request)

    async def execute(deadline):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        try:
            return [
                chunk
                async for chunk in PROXY.stream_service_response_with_retry(
                    client,
                    "/chat/completions",
                    {},
                    request_id="req-first-output-timeout",
                    decoder_node="decoder.test:6722",
                    decoder_attempt=1,
                    first_prefill_sent_at=PROXY.time.monotonic(),
                    first_output_deadline_at=deadline,
                    max_retries=1,
                    base_delay=0,
                )
            ]
        finally:
            await client.aclose()

    previous_enabled = PROXY.RESPONSES_WIRE_TRACE
    previous_dir = PROXY.RESPONSES_WIRE_TRACE_DIR
    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    with tempfile.TemporaryDirectory() as temp_dir:
        PROXY.RESPONSES_WIRE_TRACE = True
        PROXY.RESPONSES_WIRE_TRACE_DIR = temp_dir
        PROXY.stats_collector = PROXY.StatsCollector()
        PROXY.global_args = SimpleNamespace(
            decoder_first_response_observe_seconds=0,
            decoder_first_response_log_interval_seconds=1,
        )
        trace = PROXY.WireTrace("resp-first-output-timeout")
        token = PROXY._active_wire_trace.set(trace)
        try:
            deadline = PROXY.time.monotonic() + 0.01
            try:
                run(execute(deadline))
                assert False, "expected first-output deadline to fail"
            except PROXY.PdFirstOutputTimeoutError:
                pass
            snapshot = PROXY.stats_collector._decoder_observability_snapshot_locked()
        finally:
            PROXY._active_wire_trace.reset(token)
            PROXY.RESPONSES_WIRE_TRACE = previous_enabled
            PROXY.RESPONSES_WIRE_TRACE_DIR = previous_dir
            PROXY.stats_collector = previous_stats
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

        records = [
            json.loads(line)
            for line in (
                Path(temp_dir) / "resp-first-output-timeout" / "decode.attempts.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        assert [record["event"] for record in records] == [
            "started",
            "response_headers_received",
            "first_valid_output_timeout",
        ]
        assert snapshot["first_response_waiting_active"] == 0


def test_empty_assistant_start_chunk_does_not_cancel_first_valid_output_deadline():
    class EmptyThenStallStream(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n'
            await asyncio.sleep(0.05)

        async def aclose(self):
            self.closed = True

    stream = EmptyThenStallStream()

    async def request_handler(request):
        return httpx.Response(200, stream=stream, request=request)

    async def execute(deadline):
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        try:
            progress = PROXY.DecoderStreamProgress()
            return [
                chunk
                async for chunk in PROXY.stream_service_response_with_retry(
                    client,
                    "/chat/completions",
                    {},
                    request_id="req-empty-start-timeout",
                    decoder_node="decoder.test:6722",
                    decoder_attempt=1,
                    first_prefill_sent_at=PROXY.time.monotonic(),
                    first_output_deadline_at=deadline,
                    stream_progress=progress,
                    max_retries=1,
                    base_delay=0,
                )
            ]
        finally:
            await client.aclose()

    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    PROXY.stats_collector = PROXY.StatsCollector()
    PROXY.global_args = SimpleNamespace(
        decoder_first_response_observe_seconds=0,
        decoder_first_response_log_interval_seconds=1,
        decoder_stream_idle_timeout_seconds=0.01,
    )
    try:
        try:
            run(execute(PROXY.time.monotonic() + 0.01))
            assert False, "expected first-valid-output deadline to fail"
        except PROXY.PdFirstOutputTimeoutError:
            pass
        assert stream.closed
        snapshot = PROXY.stats_collector._decoder_observability_snapshot_locked()
        assert snapshot["first_sse_received"] == 1
        assert snapshot["first_valid_output_received"] == 0
        assert snapshot["first_valid_output_timeout_total"] == 1
    finally:
        PROXY.stats_collector = previous_stats
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_valid_output_switches_to_decoder_idle_deadline():
    class OutputThenStallStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
            await asyncio.sleep(0.05)

        async def aclose(self):
            return None

    async def request_handler(request):
        return httpx.Response(200, stream=OutputThenStallStream(), request=request)

    async def execute():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(request_handler),
            base_url="http://decoder.test/v1",
        )
        progress = PROXY.DecoderStreamProgress()
        try:
            async for chunk in PROXY.stream_service_response_with_retry(
                client,
                "/chat/completions",
                {},
                request_id="req-idle-timeout",
                decoder_node="decoder.test:6722",
                decoder_attempt=1,
                first_output_deadline_at=PROXY.time.monotonic() + 1,
                stream_progress=progress,
                max_retries=1,
                base_delay=0,
            ):
                assert chunk
                progress.observe_valid_output()
        finally:
            await client.aclose()

    previous_stats = PROXY.stats_collector
    previous_args = getattr(PROXY, "global_args", None)
    PROXY.stats_collector = PROXY.StatsCollector()
    PROXY.global_args = SimpleNamespace(
        decoder_first_response_observe_seconds=0,
        decoder_first_response_log_interval_seconds=1,
        decoder_stream_idle_timeout_seconds=0.01,
    )
    try:
        try:
            run(execute())
            assert False, "expected idle deadline to fail"
        except PROXY.DecoderStreamIdleTimeoutError:
            pass
        snapshot = PROXY.stats_collector._decoder_observability_snapshot_locked()
        assert snapshot["first_valid_output_received"] == 1
        assert snapshot["stream_idle_timeout_total"] == 1
    finally:
        PROXY.stats_collector = previous_stats
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args
