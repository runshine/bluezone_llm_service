"""Native V4.1 image rendering, error boundaries and shared P/D transport."""
import asyncio
import base64
import copy
import importlib.util
import json
import io
import threading
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image
from starlette.requests import Request

SPEC = importlib.util.spec_from_file_location("multimodal_pd_tests", Path(__file__).with_name("load_balance_proxy_server.py"))
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

URL = "data:image/png;base64,AA=="
IMAGE = {"type": "image_url", "image_url": {"url": URL, "detail": "auto"}}


def real_image_part(*, corrupt=False):
    buffer = io.BytesIO()
    Image.new("RGB", (720, 200), "white").save(buffer, format="JPEG")
    data = buffer.getvalue()
    if corrupt:
        data = data[:int(len(data) * 0.3)]
    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


def test_corrupt_tool_image_replaced_without_mutating_history(caplog):
    good, bad = real_image_part(), real_image_part(corrupt=True)
    remote = {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}
    original = {"messages": [{"role": "tool", "tool_call_id": "call_image", "content": [
        {"type": "text", "text": "before"}, good, bad, remote,
        {"type": "text", "text": "after"}, copy.deepcopy(bad),
    ]}]}
    untouched = copy.deepcopy(original)
    prepared = P.prepare_backend_request_data(original)
    parts = prepared["messages"][0]["content"]
    assert original == untouched
    assert parts[0] == original["messages"][0]["content"][0]
    assert parts[1] == good and parts[3] == remote
    assert parts[4] == original["messages"][0]["content"][4]
    assert parts[2] == parts[5] == {"type": "text", "text": P.TOOL_IMAGE_FAILURE_TEXT}
    assert prepared["messages"][0]["tool_call_id"] == "call_image"
    assert P.prepare_backend_request_data(prepared) == prepared
    assert "message_index=0 content_index=2" in caplog.text
    assert bad["image_url"]["url"] not in caplog.text
    # Restoring the original history applies the same policy without changing it.
    assert P.prepare_backend_request_data(original) == prepared


def test_corrupt_user_image_still_rejected_by_upstream(monkeypatch):
    async def check():
        bad = real_image_part(corrupt=True)
        prepared = P.prepare_backend_request_data({"messages": [{"role": "user", "content": [bad]}]})
        assert prepared["messages"][0]["content"] == [bad]
        def handler(request):
            assert json.loads(request.content)["messages"] == prepared["messages"]
            return httpx.Response(400, json={"error": {"message": "Failed to load image"}})
        client = setup_counting(monkeypatch, handler)
        try:
            with pytest.raises(P.ResponsesRequestError) as err:
                await P.count_request_tokens(prepared, None)
            assert err.value.status_code == 400
        finally:
            await client.aclose()
    asyncio.run(check())


def test_replaced_only_tool_image_uses_text_counting(monkeypatch):
    async def check():
        def local(req, tokenizer):
            assert P.image_input_count(req["messages"]) == 0
            assert req["messages"][0]["content"][0]["text"] == P.TOOL_IMAGE_FAILURE_TEXT
            return 20, 20
        monkeypatch.setattr(P, "count_prefill_and_decode_tokens", local)
        # No remote client or node pool is needed after the last image is replaced.
        monkeypatch.setattr(P, "proxy_state", None)
        prepared = P.prepare_backend_request_data({"messages": [
            {"role": "tool", "content": [real_image_part(corrupt=True)]},
        ]})
        assert await P.count_request_tokens(prepared, None) == (20, 20)
    asyncio.run(check())


def request_data():
    return P.prepare_backend_request_data({
        "model": "DeepSeekv4.1", "max_tokens": 64,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "before"}, copy.deepcopy(IMAGE),
            {"type": "text", "text": "after"}, copy.deepcopy(IMAGE),
        ]}],
        "mm_processor_kwargs": {"max_image_tokens": 1024},
        "tools": [{"type": "function", "function": {"name": "inspect", "parameters": {"type": "object"}}}],
    })


def state_without_clients():
    state = object.__new__(P.ProxyState)
    state._lock = threading.RLock()
    state._tokenize_round_robin = {"prefill": 0, "decode": 0}
    state.prefillers = [SimpleNamespace(url="http://prefill:6700/v1", active_requests=0)]
    state.decoders = [SimpleNamespace(url="http://decode:6721/v1", active_requests=0)]
    state.tainted_prefillers = []
    state.tainted_decoders = []
    state.pending_remove_prefillers = []
    state.pending_remove_decoders = []
    return state


def setup_counting(monkeypatch, handler, *, timeout=1, concurrency=8):
    monkeypatch.setattr(P, "proxy_state", state_without_clients())
    monkeypatch.setattr(P, "global_args", SimpleNamespace(
        multimodal_tokenize_timeout_seconds=timeout, multimodal_max_image_bytes=20 * 1024 * 1024,
    ), raising=False)
    monkeypatch.setattr(P, "multimodal_tokenize_semaphore", asyncio.Semaphore(concurrency))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    monkeypatch.setattr(P, "multimodal_tokenize_client", client)
    return client


def test_counting_payload_and_parallel_roles(monkeypatch):
    async def check():
        seen = []
        both = asyncio.Event()
        async def handler(request):
            assert request.url.path == "/tokenize"
            body = json.loads(request.content)
            seen.append((request.url.host, body))
            if len(seen) == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 0.5)
            assert body["messages"] == original["messages"]
            assert body["tools"] == original["tools"]
            assert body["mm_processor_kwargs"] == original["mm_processor_kwargs"]
            assert body["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "high"}
            assert not any(key in body for key in ("reasoning_effort", "max_tokens", "stream", "kv_transfer_params"))
            return httpx.Response(200, json={"count": 218 if request.url.host == "prefill" else 220})
        client = setup_counting(monkeypatch, handler)
        original = request_data()
        original["kv_transfer_params"] = {"remote_block_ids": [1]}
        untouched = copy.deepcopy(original)
        try:
            assert await P.count_request_tokens(original, None) == (218, 220)
            assert original == untouched
            assert len(seen) == 2
            assert P.proxy_state.prefillers[0].active_requests == 0
        finally:
            await client.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("status,body,expected", [
    (400, {"error": URL}, 400), (422, {}, 400), (429, {}, 502),
    (500, {}, 502), (200, {}, 502), (200, {"count": True}, 502),
    (200, {"count": -1}, 502), (200, {"count": "218"}, 502),
    (200, [], 502),
])
def test_counting_errors_are_sanitized_and_not_retried(monkeypatch, status, body, expected):
    async def check():
        calls = []
        def handler(request):
            calls.append(request)
            return httpx.Response(status, json=body)
        client = setup_counting(monkeypatch, handler)
        try:
            with pytest.raises(P.ResponsesRequestError) as err:
                await P.count_request_tokens(request_data(), None)
            assert err.value.status_code == expected
            assert "Image tokenize" in str(err.value) and "node=http://" in str(err.value)
            assert URL not in str(err.value)
            assert len(calls) <= 2
            assert not P.proxy_state.tainted_prefillers and not P.proxy_state.tainted_decoders
        finally:
            await client.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("failure,expected", [("connect", 502), ("timeout", 504), ("queued", 504)])
def test_counting_deadline_includes_semaphore(monkeypatch, failure, expected):
    async def check():
        calls = []
        async def handler(request):
            calls.append(request)
            if failure == "connect":
                raise httpx.ConnectError("must not echo " + URL)
            await asyncio.Event().wait()
        client = setup_counting(monkeypatch, handler, timeout=0.02)
        if failure == "queued":
            monkeypatch.setattr(P, "multimodal_tokenize_semaphore", asyncio.Semaphore(0))
        try:
            with pytest.raises(P.ResponsesRequestError) as err:
                await P.count_request_tokens(request_data(), None)
            assert err.value.status_code == expected
            assert URL not in str(err.value)
            if failure == "queued":
                assert calls == []
        finally:
            await client.aclose()
    asyncio.run(check())


def test_counting_global_concurrency_and_cancellation(monkeypatch):
    async def check():
        active = peak = 0
        started = asyncio.Event()
        release = asyncio.Event()
        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            started.set()
            try:
                await release.wait()
                return httpx.Response(200, json={"count": 218})
            finally:
                active -= 1
        client = setup_counting(monkeypatch, handler, concurrency=2)
        tasks = [asyncio.create_task(P.count_request_tokens(request_data(), None)) for _ in range(4)]
        try:
            await started.wait()
            await asyncio.sleep(0)
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            release.set()
            assert await asyncio.gather(*tasks[1:]) == [(218, 218)] * 3
            assert peak <= 2 and active == 0
            assert P.multimodal_tokenize_semaphore._value == 2
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await client.aclose()
    asyncio.run(check())


def test_text_uses_only_local_counter(monkeypatch):
    monkeypatch.setattr(P, "count_prefill_and_decode_tokens", lambda req, tok: (11, 12))
    def fail(*args):
        raise AssertionError("Text must not select a counting node")
    monkeypatch.setattr(P, "proxy_state", SimpleNamespace(select_tokenize_node=fail))
    assert asyncio.run(P.count_request_tokens({"messages": [{"role": "user", "content": "hello"}]}, None)) == (11, 12)


@pytest.mark.parametrize("code", [400, 502, 504])
def test_common_handler_returns_count_error_without_dispatch(monkeypatch, code):
    monkeypatch.setattr(P, "global_args", SimpleNamespace(max_input_chars=0), raising=False)
    monkeypatch.setattr(P, "proxy_state", SimpleNamespace(inc_request_num=lambda: None, dec_request_num=lambda: None))
    async def fail(*args):
        raise P.ResponsesRequestError("Image tokenize decode node=http://d: failed.", "multimodal_tokenize_failed", code)
    async def dispatch(*args, **kwargs):
        raise AssertionError("Failed counting must not dispatch inference")
    monkeypatch.setattr(P, "count_request_tokens", fail)
    monkeypatch.setattr(P, "_handle_select_instance", dispatch)
    result = asyncio.run(P._handle_completions("/chat/completions", make_request("/chat/completions", request_data())))
    assert result.status_code == code
    assert json.loads(result.body)["code"] == "multimodal_tokenize_failed"


def test_nullable_tool_schema_remains_valid_content():
    req = request_data()
    req["messages"][0]["tools"] = [{"type": "function", "function": {
        "name": "inspect", "parameters": {"type": ["object", "null"]},
    }}]
    assert P.image_input_count(req["messages"]) == 2
    assert P.estimate_json_string_chars(req) > 0


def test_count_node_round_robin_excludes_unavailable():
    state = state_without_clients()
    p0 = state.prefillers[0]
    p1 = SimpleNamespace(url="http://p1/v1", active_requests=0)
    p2 = SimpleNamespace(url="http://p2/v1", active_requests=0)
    p3 = SimpleNamespace(url="http://p3/v1", active_requests=0)
    state.prefillers += [p1, p2, p3]
    state.tainted_prefillers = [p2]
    state.pending_remove_prefillers = [p3]
    assert [state.select_tokenize_node("prefill") for _ in range(4)] == [p0, p1, p0, p1]
    assert all(node.active_requests == 0 for node in state.prefillers)
    state.tainted_prefillers = state.prefillers
    with pytest.raises(P.ResponsesRequestError):
        state.select_tokenize_node("prefill")


@pytest.mark.parametrize("effort,expected", [
    (None, "high"), ("medium", "high"), ("unknown", "high"), ("ultra", "high"),
    ("Low", "low"), ("high", "max"), ("extra high", "max"), ("max", "max"), ("off", None),
])
def test_image_responses_reasoning(monkeypatch, effort, expected):
    original = {"model": "public", "input": [{"role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": URL, "detail": "low"},
    ]}], "reasoning": {"effort": effort}, "store": False}
    untouched = copy.deepcopy(original)
    chat, state = asyncio.run(P.responses_request_to_chat_request(original))
    assert state.backend_route == "pd" and state.contains_images
    assert chat.get("reasoning_effort") == expected
    payload = P.build_multimodal_tokenize_payload(chat)
    assert payload["chat_template_kwargs"].get("reasoning_effort") == expected
    assert payload["chat_template_kwargs"]["enable_thinking"] == (expected is not None)
    assert original == untouched


@pytest.mark.parametrize("part", [
    {"type": "input_image", "file_id": "file_1"},
    {"type": "image_url", "image_url": URL},
    {"type": "image_url", "image_url": {"url": "ftp://host/a"}},
    {"type": "image_url", "image_url": {"url": "http:///a"}},
    {"type": "image_url", "image_url": {"url": "http://["}},
    {"type": "input_image", "image_url": "data:text/plain;base64,AA=="},
    {"type": "input_image", "image_url": "data:image/png;base64,?"},
    {"type": "input_image", "image_url": URL, "detail": "bad"},
    {"type": "input_audio"}, {"type": "video"}, {"type": "input_file"},
    {"type": {}}, 123,
])
def test_invalid_content_is_not_silently_dropped(part):
    with pytest.raises(P.ResponsesRequestError):
        P.normalize_chat_messages_for_backend([{"role": "user", "content": [part]}])


@pytest.mark.parametrize("part_type", ["input_image", "image_url"])
@pytest.mark.parametrize("location", ["user", "tool"])
def test_original_detail_preserved_in_responses_and_tokenize(part_type, location):
    part = ({"type": "input_image", "image_url": URL, "detail": "original"}
            if part_type == "input_image" else
            {"type": "image_url", "image_url": {"url": URL, "detail": "original"}})
    items = ([{"role": "user", "content": [part]}] if location == "user" else [
        {"type": "function_call", "name": "inspect", "call_id": "image_call", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "image_call", "output": [part]},
    ])
    body = {"model": "public", "input": items, "store": False,
            "tools": [{"type": "function", "name": "inspect", "parameters": {"type": "object"}}]}
    untouched = copy.deepcopy(body)
    chat, state = asyncio.run(P.responses_request_to_chat_request(body))
    expected = {"type": "image_url", "image_url": {"url": URL, "detail": "original"}}
    images = [p for m in chat["messages"] if isinstance(m.get("content"), list)
              for p in m["content"] if p.get("type") == "image_url"]
    assert images == [expected]
    assert state.contains_images and state.backend_route == "pd"
    assert P.build_multimodal_tokenize_payload(chat)["messages"] == chat["messages"]
    assert P.prepare_prefill_request_data(chat)["messages"] == chat["messages"]
    assert body == untouched


def test_image_size_guard_and_text_character_limit(monkeypatch):
    monkeypatch.setattr(P, "global_args", SimpleNamespace(multimodal_max_image_bytes=1), raising=False)
    with pytest.raises(P.ResponsesRequestError) as error:
        P._responses_multimodal_content([{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}])
    assert error.value.code == "multimodal_image_too_large"
    assert P.estimate_json_string_chars([{"type": "text", "text": "abc"}, IMAGE]) == 7
    assert P.estimate_json_string_chars(URL) == len(URL)  # User text still counts.


@pytest.mark.parametrize("old_route", ["multimodal_direct", "pd_text", "pd"])
def test_previous_response_migration_keeps_images_and_tools(monkeypatch, old_route):
    previous = {"id": "resp_old", "model": "public", "_backend_route": old_route,
                "_backend_model": "old-vision-model", "_normalized_messages": [
                    {"role": "user", "content": [copy.deepcopy(IMAGE)]},
                    {"role": "assistant", "content": "a picture"},
                ], "_tool_registry": {"inspect": {"source_type": "function", "name": "inspect",
                                                  "parameters": {"type": "object"}}}}
    async def get(response_id):
        assert response_id == "resp_old"
        return previous
    monkeypatch.setattr(P.response_state_store, "get", get)
    monkeypatch.setattr(P, "RESPONSES_REDIS_URL", "redis://test")
    chat, state = asyncio.run(P.responses_request_to_chat_request({
        "model": "public", "previous_response_id": "resp_old", "input": "What color?", "store": False,
    }))
    assert state.backend_route == "pd" and state.contains_images
    assert state.backend_model != "old-vision-model"
    assert P.image_input_count(chat["messages"]) == 1
    assert state.tool_registry["inspect"]["name"] == "inspect"
    public = P._public_response_state(P._state_payload({"id": "resp_new", "output": []}, chat, state))
    assert not any(key.startswith("_") for key in public)
    assert previous["_backend_route"] == old_route


def test_recompute_preserves_images_and_recounts(monkeypatch):
    async def check():
        counts = []
        def handler(request):
            body = json.loads(request.content)
            counts.append(body)
            return httpx.Response(200, json={"count": 218 + len(body["messages"])})
        client = setup_counting(monkeypatch, handler)
        req = request_data()
        base = copy.deepcopy(req["messages"])
        try:
            await P.count_request_tokens(req, None)
            P.prepare_retry_request(req, None, "partial", True, base)
            assert P.image_input_count(req["messages"]) == 2
            assert await P.count_request_tokens(req, None) == (220, 220)
            assert len(counts) == 4
            assert counts[0]["messages"] == base
            assert counts[2]["messages"][-1] == {"role": "assistant", "content": "partial"}
            assert counts[0]["chat_template_kwargs"] == counts[2]["chat_template_kwargs"]
        finally:
            await client.aclose()
    asyncio.run(check())


def test_image_in_last_assistant_survives_recompute():
    req = {"messages": [{"role": "assistant", "content": [copy.deepcopy(IMAGE)]}]}
    P.prepare_retry_request(req, None, "partial", True)
    assert P.image_input_count(req["messages"]) == 1


def test_prefill_and_decoder_transport_keep_images_on_retry(monkeypatch):
    async def check():
        seen = {"prefill": [], "decode": []}
        def handler(request):
            role = request.url.host
            body = json.loads(request.content)
            seen[role].append(body)
            if len(seen[role]) == 1:
                return httpx.Response(503, json={"error": "retry"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
        req = request_data()
        monkeypatch.setattr(P, "proxy_state", SimpleNamespace(
            mark_backend_success=lambda *args: None,
        ))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://prefill/v1") as pc:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://decode/v1") as dc:
                await P.send_request_to_service(pc, SimpleNamespace(url="http://prefill/v1"), "/chat/completions", req, "req_mm", max_retries=2, base_delay=0)
                await P.send_decode_request_with_retry(dc, "/chat/completions", req, "req_mm", max_retries=2, base_delay=0)
        for bodies in seen.values():
            assert len(bodies) == 2 and bodies[0] == bodies[1]
            assert bodies[0]["messages"] == req["messages"]
            assert bodies[0]["reasoning_effort"] == "high"
        assert seen["prefill"][0]["max_tokens"] == 1
        assert seen["decode"][0]["max_tokens"] == 64
    asyncio.run(check())


def make_request(path, payload):
    raw = json.dumps(payload).encode()
    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}
    return Request({"type": "http", "method": "POST", "path": path,
                    "headers": [], "client": ("127.0.0.1", 1)}, receive)


@pytest.mark.parametrize("path", ["/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses"])
@pytest.mark.parametrize("images", [False, True])
def test_all_routes_reach_common_pd_handler(monkeypatch, path, images):
    seen = []
    async def handler(api, request, **kwargs):
        body = json.loads(await request.body())
        seen.append(body)
        assert P._has_multimodal_content(body) == images
        return P.JSONResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 218, "completion_tokens": 1}})
    monkeypatch.setattr(P, "_handle_completions", handler)
    monkeypatch.setattr(P, "REQUEST_AUDIT_TRACE", False)
    monkeypatch.setattr(P, "RESPONSES_WIRE_TRACE", False)
    payload = {"model": "public", "store": False}
    content = [copy.deepcopy(IMAGE)] if images else "hello"
    if path.endswith("responses"):
        payload["input"] = [{"role": "user", "content": content}]
        result = asyncio.run(P.handle_responses_route(make_request(path, payload)))
    else:
        payload["messages"] = [{"role": "user", "content": content}]
        result = asyncio.run(P.handle_chat_completions(make_request(path, payload)))
    assert result.status_code == 200 and len(seen) == 1
    assert path in {route.path for route in P.app.routes}
    assert not hasattr(P, "multimodal_client") and not hasattr(P, "_run_multimodal_backend")


def test_streaming_count_failure_has_terminal_event(monkeypatch):
    async def check():
        async def fail(request, chat, state, trace, raw):
            return P.responses_error("Image tokenize prefill node=http://p: timed out.", "multimodal_tokenize_timeout", 504)
        state = P.ResponseRequestState("resp_mm_error", "DeepSeekv4.1", 1, False, contains_images=True)
        chunks = [chunk async for chunk in P._responses_stream_lifecycle(
            make_request("/responses", {}), {"stream": True, "messages": []}, state,
            P.WireTrace(state.response_id), b"{}", backend_runner=fail,
        )]
        events = [json.loads(chunk.decode()[6:]) for chunk in chunks if chunk.startswith(b"data: {")]
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] in {"response.failed", "response.incomplete"}
        assert events[-2]["type"] == "error"
        assert events[-2]["error"]["code"] == "multimodal_tokenize_timeout"
    monkeypatch.setattr(P, "RESPONSES_WIRE_TRACE", False)
    asyncio.run(check())


def test_derived_traces_redact_but_raw_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "RESPONSES_WIRE_TRACE", True)
    monkeypatch.setattr(P, "RESPONSES_WIRE_TRACE_DIR", str(tmp_path))
    trace = P.WireTrace("resp_images")
    req = request_data()
    trace.write_json("request.raw.json", req)
    trace.write_json("prefill.request.json", req)
    assert URL in (tmp_path / "resp_images/request.raw.json").read_text()
    assert URL not in (tmp_path / "resp_images/prefill.request.json").read_text()
    redacted = P._redact_multimodal(req)
    assert P._redact_multimodal(redacted) == redacted


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("corrupt_tool", [False, True])
@pytest.mark.parametrize("detail", ["auto", "original"])
def test_complete_mock_pd_pipeline(monkeypatch, stream, recompute, corrupt_tool, detail):
    """Run the real common handler, schedulers, counting, KV and retry paths."""
    monkeypatch.setattr(sys, "argv", ["proxy", "--backend-model-name", "DeepSeekv4.1"])
    monkeypatch.setattr(P, "global_args", P.parse_args(), raising=False)
    monkeypatch.setattr(P, "NodeListener", lambda state: SimpleNamespace(record_business_success=lambda *args: None))
    monkeypatch.setattr(P, "stats_collector", P.StatsCollector())
    monkeypatch.setattr(P, "prefiller_max_model_len", 100000)
    monkeypatch.setattr(P, "decoder_max_model_len", 100000)
    monkeypatch.setattr(P, "REQUEST_AUDIT_TRACE", False)
    monkeypatch.setattr(P, "RESPONSES_WIRE_TRACE", False)
    monkeypatch.setattr(P, "PREFILL_FAILURE_CAPTURE", False)
    async def check():
        state = P.ProxyState([("prefill", 6700)], [("decode", 6721)])
        monkeypatch.setattr(P, "proxy_state", state)
        seen = {"tokenize": [], "prefill": [], "decode": []}
        kv = {"do_remote_prefill": True, "remote_block_ids": [4], "remote_engine_id": "P"}
        def handler(request):
            body = json.loads(request.content)
            for message in body["messages"]:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") == "image_url":
                            assert part["image_url"]["detail"] == detail
            if corrupt_tool:
                tool_messages = [m for m in body["messages"] if m["role"] == "tool"]
                assert len(tool_messages) == 1
                assert tool_messages[0]["content"] == [
                    {"type": "text", "text": P.TOOL_IMAGE_FAILURE_TEXT},
                ]
            if request.url.path == "/tokenize":
                seen["tokenize"].append(body)
                return httpx.Response(200, json={"count": 218 + len(body["messages"])})
            role = request.url.host
            seen[role].append(body)
            assert P.image_input_count(body["messages"]) == 2
            if role == "prefill":
                assert body["max_tokens"] == 1
                return httpx.Response(200, json={"kv_transfer_params": kv})
            assert body["kv_transfer_params"] == kv
            reason = "recomputed" if recompute and len(seen["decode"]) == 1 else "stop"
            choice = {"finish_reason": reason, "stop_reason": reason}
            usage = {"prompt_tokens": 218 + len(body["messages"]), "completion_tokens": 1}
            if stream:
                choice["delta"] = {"content": "partial" if reason == "recomputed" else "done"}
                wire = ("data: " + json.dumps({"choices": [choice], "usage": usage}) + "\n\n"
                        + "data: [DONE]\n\n").encode()
                return httpx.Response(200, content=wire, headers={"content-type": "text/event-stream"})
            choice["message"] = {"content": "partial" if reason == "recomputed" else "done"}
            return httpx.Response(200, json={"choices": [choice], "usage": usage})
        transport = httpx.MockTransport(handler)
        clients = []
        try:
            for node in [*state.prefillers, *state.decoders]:
                await node.client.aclose()
                await node.prefill_client.aclose()
                node.client = httpx.AsyncClient(transport=transport, base_url=node.url)
                node.prefill_client = httpx.AsyncClient(transport=transport, base_url=node.url)
                clients += [node.client, node.prefill_client]
            counting = httpx.AsyncClient(transport=transport, trust_env=False)
            clients.append(counting)
            monkeypatch.setattr(P, "multimodal_tokenize_client", counting)
            monkeypatch.setattr(P, "multimodal_tokenize_semaphore", asyncio.Semaphore(8))
            body = request_data()
            for message in body["messages"]:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        if part.get("type") == "image_url":
                            part["image_url"]["detail"] = detail
            if corrupt_tool:
                body["messages"].append({"role": "tool", "tool_call_id": "call_bad_image",
                                         "content": [real_image_part(corrupt=True)]})
            untouched = copy.deepcopy(body)
            body["stream"] = stream
            result = await P._handle_completions("/chat/completions", make_request("/chat/completions", body))
            if stream:
                wire = b"".join([part async for part in result.body_iterator])
                assert b"done" in wire and wire.endswith(b"data: [DONE]\n\n")
                assert b'"error"' not in wire
            else:
                assert result.status_code == 200
            expected = 2 if recompute else 1
            assert len(seen["prefill"]) == len(seen["decode"]) == expected
            assert len(seen["tokenize"]) == 2 * expected
            for role in ("prefill", "decode"):
                for sent in seen[role]:
                    assert sent["reasoning_effort"] == "high"
                    assert sent["messages"][0] == body["messages"][0]
            assert state.get_request_num() == 0
            assert state.prefillers[0].active_requests == state.decoders[0].active_requests == 0
            assert body["messages"] == untouched["messages"]
        finally:
            for client in clients:
                await client.aclose()
    asyncio.run(check())
