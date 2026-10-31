"""V4.1 reasoning contract and prompt parity, without contacting inference nodes."""

import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest
import httpx


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v41_proxy_tests", HERE / "load_balance_proxy_server.py")
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


def responses(extra):
    return asyncio.run(PROXY.responses_request_to_chat_request({
        "model": "public-model", "input": "hello", "store": False, **extra,
    }))


@pytest.mark.parametrize("raw,expected,effective", [
    ("light", "low", "low"), ("LOW", "low", "low"), ("minimal", "low", "low"),
    (" medium ", "high", "medium"), ("high", "max", "high"),
    ("xhigh", "max", "xhigh"), ("extra_high", "max", "xhigh"),
    ("Extra High", "max", "xhigh"), ("max", "max", "max"),
    (None, "low", "low"), ("", "low", "low"), ("  ", "low", "low"),
    ("future-value", "high", "medium"), ("ultra", "high", "medium"),
])
def test_codex_mapping_and_original_request(raw, expected, effective):
    request = {"reasoning": {"effort": raw}}
    original = copy.deepcopy(request)
    chat, state = responses(request)
    assert request == original
    assert chat["reasoning_effort"] == expected
    assert chat["chat_template_kwargs"]["enable_thinking"] is True
    assert state.reasoning_effort == effective
    assert state.reasoning_settings["raw_effort"] == raw
    if raw is None or not raw.strip():
        assert state.reasoning_settings["default_reason"] == "missing_effort"
        assert state.reasoning_settings["fallback_reason"] is None
    elif raw in {"future-value", "ultra"}:
        assert state.reasoning_settings["fallback_reason"] == "unknown_effort"
        assert state.reasoning_settings["default_reason"] is None
    for _ in range(3):
        chat = PROXY.prepare_backend_request_data(chat, reasoning_settings=state.reasoning_settings)
        assert chat["reasoning_effort"] == expected
    assert "reasoning_settings" not in chat
    assert "source" not in chat
    assert "reasoning" not in chat


@pytest.mark.parametrize("extra", [{}, {"thinking": {"type": "enabled"}}, {"reasoning": {"effort": None}}])
def test_default_is_low(extra):
    chat, state = responses(extra)
    assert chat["reasoning_effort"] == "low"
    assert state.reasoning_effort == "low"
    assert state.reasoning_settings["default_reason"] == "missing_effort"


@pytest.mark.parametrize("raw,expected", [
    ("low", "low"), ("high", "high"), ("max", "max"),
    ("medium", "high"), ("xhigh", "high"), ("ultra", "high"),
    (None, "low"), ("", "low"), (" HIGH ", "high"),
])
def test_native_chat_semantics(raw, expected):
    request = {"model": "gpt-client-alias", "reasoning_effort": raw}
    original = copy.deepcopy(request)
    backend = PROXY.prepare_backend_request_data(request)
    assert backend["reasoning_effort"] == expected
    assert backend["chat_template_kwargs"]["enable_thinking"] is True
    assert request == original


@pytest.mark.parametrize("alias", ["none", "off", "disabled", "disable", " OFF "])
def test_legacy_disable_and_explicit_conflict(alias):
    for source in ("responses", "chat"):
        request = {"reasoning_effort": alias}
        backend = PROXY.normalize_reasoning_effort_for_backend(request, source=source)
        assert "reasoning_effort" not in backend
        assert backend["chat_template_kwargs"] == {"enable_thinking": False}
        with pytest.raises(PROXY.ResponsesRequestError, match="enabled"):
            PROXY.resolve_reasoning_settings({**request, "thinking": {"type": "enabled"}}, source=source)


@pytest.mark.parametrize("switch", [
    {"thinking": {"type": "disabled"}},
    {"chat_template_kwargs": {"thinking": False}},
    {"chat_template_kwargs": {"enable_thinking": False}},
])
def test_disabled_ignores_strength_and_removes_duplicate_parameters(switch):
    request = {"reasoning_effort": "max", **copy.deepcopy(switch)}
    request.setdefault("chat_template_kwargs", {}).update(reasoning_effort="low", drop_thinking=False)
    original = copy.deepcopy(request)
    backend = PROXY.normalize_reasoning_effort_for_backend(request)
    assert request == original
    assert "thinking" not in backend
    assert "reasoning_effort" not in backend
    assert backend["chat_template_kwargs"] == {"enable_thinking": False, "drop_thinking": False}
    assert PROXY._infer_thinking_mode(backend) == "chat"


@pytest.mark.parametrize("payload,code", [
    ({"thinking": True}, "invalid_thinking"),
    ({"thinking": None}, "invalid_thinking"),
    ({"thinking": {"type": "bad"}}, "invalid_thinking"),
    ({"thinking": {"type": []}}, "invalid_thinking"),
    ({"thinking": {"type": "enabled", "extra": 1}}, "invalid_thinking"),
    ({"chat_template_kwargs": []}, "invalid_thinking"),
    ({"chat_template_kwargs": {"enable_thinking": 1}}, "invalid_thinking"),
    ({"chat_template_kwargs": {"thinking": {"type": "disabled"}}}, "invalid_thinking"),
    ({"thinking": {"type": "disabled"}, "chat_template_kwargs": {"enable_thinking": True}}, "conflicting_reasoning_parameters"),
    ({"chat_template_kwargs": {"thinking": False, "enable_thinking": True}}, "conflicting_reasoning_parameters"),
    ({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "max"}}, "conflicting_reasoning_parameters"),
    ({"reasoning_effort": "none", "chat_template_kwargs": {"reasoning_effort": "high"}}, "conflicting_reasoning_parameters"),
])
def test_invalid_switches_and_conflicts(payload, code):
    with pytest.raises(PROXY.ResponsesRequestError) as exc:
        PROXY.resolve_reasoning_settings(payload, source="chat")
    assert exc.value.status_code == 400
    assert exc.value.code == code


@pytest.mark.parametrize("raw", [1, True, [], {}])
def test_invalid_effort_types_are_not_unknown_strings(raw):
    with pytest.raises(PROXY.ResponsesRequestError) as exc:
        responses({"reasoning": {"effort": raw}})
    assert exc.value.code == "invalid_reasoning_effort"
    for request in ({"reasoning_effort": raw}, {"chat_template_kwargs": {"reasoning_effort": raw}}):
        with pytest.raises(PROXY.ResponsesRequestError) as exc:
            PROXY.resolve_reasoning_settings(request, source="chat")
        assert exc.value.code == "invalid_reasoning_effort"


def test_template_effort_is_native_and_merged_after_codex_mapping():
    chat, state = responses({
        "reasoning": {"effort": "medium"},
        "chat_template_kwargs": {"reasoning_effort": "high", "drop_thinking": False},
    })
    assert chat["reasoning_effort"] == "high"
    assert chat["chat_template_kwargs"] == {"enable_thinking": True, "drop_thinking": False}
    assert state.reasoning_settings["raw_template_effort"] == "high"
    with pytest.raises(PROXY.ResponsesRequestError):
        responses({"reasoning": {"effort": "high"}, "chat_template_kwargs": {"reasoning_effort": "high"}})
    backend = PROXY.prepare_backend_request_data({"chat_template_kwargs": {"reasoning_effort": "low"}})
    assert backend["reasoning_effort"] == "low"


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("effort,expected", [("low", "low"), ("medium", "high"), ("high", "max")])
def test_pd_wire_and_transport_and_protocol_retries_preserve_settings(monkeypatch, stream, enabled, effort, expected):
    from types import SimpleNamespace
    monkeypatch.setattr(PROXY, "global_args", SimpleNamespace(decoder_first_response_observe_seconds=0), raising=False)
    monkeypatch.setattr(PROXY, "PREFILL_FAILURE_CAPTURE", False)
    captured = {"prefill": [], "decode": []}

    async def handle(request):
        phase = request.url.host.split(".")[0]
        captured[phase].append(json.loads(await request.aread()))
        if len(captured[phase]) == 1:
            raise httpx.ConnectError("temporary transport failure", request=request)
        if phase == "prefill":
            return httpx.Response(200, json={"kv_transfer_params": {"remote_engine_id": "test-engine"}})
        if stream:
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    async def execute():
        chat, state = await PROXY.responses_request_to_chat_request({
            "model": "public-model", "input": "hello", "store": False, "stream": stream,
            "reasoning": {"effort": effort}, "thinking": {"type": "enabled" if enabled else "disabled"},
        })
        chat = PROXY.prepare_backend_request_data(chat, reasoning_settings=state.reasoning_settings)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://prefill.test/v1/") as pclient, httpx.AsyncClient(transport=httpx.MockTransport(handle), base_url="http://decode.test/v1/") as dclient:
            for recompute in range(2):
                if recompute:
                    assert PROXY.append_decoder_start_protocol_retry_prompt(chat)
                result = await PROXY.send_request_to_service(pclient, SimpleNamespace(url="http://prefill.test/v1"), "/chat/completions", chat, "test-reasoning", max_retries=2, base_delay=0)
                chat["kv_transfer_params"] = result.json()["kv_transfer_params"]
                if stream:
                    chunks = [chunk async for chunk in PROXY.stream_service_response_with_retry(
                        dclient, "/chat/completions", chat, "test-reasoning", "decode.test:6721", recompute + 1,
                        max_retries=2, base_delay=0,
                    )]
                    assert b"ok" in b"".join(chunks)
                else:
                    result = await PROXY.send_decode_request_with_retry(dclient, "/chat/completions", chat, "test-reasoning", max_retries=2, base_delay=0)
                    assert result.status_code == 200

    asyncio.run(execute())
    for phase, requests in captured.items():
        assert len(requests) == 3  # First transport attempt, retry, protocol recompute.
        for payload in requests:
            assert payload["chat_template_kwargs"]["enable_thinking"] is enabled
            assert payload.get("reasoning_effort") == (expected if enabled else None)
            assert "thinking" not in payload
            assert "reasoning_settings" not in payload
            assert "effective_client_effort" not in payload
            if phase == "prefill":
                assert payload["max_tokens"] == 1
                assert payload["stream"] is False
                assert payload["kv_transfer_params"]["do_remote_decode"] is True
            else:
                assert payload["max_tokens"] == 16384
                assert payload["stream"] is stream
                assert payload["kv_transfer_params"]["remote_engine_id"] == "test-engine"


@pytest.fixture(scope="module")
def reference_encoder():
    # Optional serving-source override allows the same suite to check the
    # exact installed backend encoder without importing vLLM or loading NPUs.
    path = Path(os.environ.get("V41_REFERENCE_ENCODER", HERE.parent / "weights/encoding/encoding.py"))
    spec = importlib.util.spec_from_file_location("reference_v41", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
CONVERSATIONS = [
    [{"role": "user", "content": "hello"}],
    [{"role": "system", "content": "Be concise"}, {"role": "user", "content": "hello"}],
    [{"role": "user", "content": "old"}, {"role": "assistant", "reasoning": "old thought", "content": "answer"}, {"role": "user", "content": "next"}],
    [{"role": "user", "content": "lookup"}, {"role": "assistant", "content": "", "reasoning_content": "lookup first", "tool_calls": [{"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"value"}'}}]}, {"role": "tool", "tool_call_id": "a", "content": "found"}],
]


class RecordingTokenizer:
    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        self.prompt = text
        return list(text.encode("utf-8"))


@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("conversation", CONVERSATIONS)
@pytest.mark.parametrize("drop_thinking", [True, False])
def test_prompt_matches_independent_v41_reference(reference_encoder, effort, enabled, conversation, drop_thinking):
    request = {
        "messages": conversation,
        "reasoning_effort": effort,
        "thinking": {"type": "enabled" if enabled else "disabled"},
        "chat_template_kwargs": {"drop_thinking": drop_thinking},
    }
    if any(m.get("tool_calls") for m in conversation):
        request["tools"] = [TOOL]
    backend = PROXY.prepare_backend_request_data(request)
    # Independently mimic the serving tokenizer's documented normalization,
    # rather than reusing the proxy's message construction helper.
    messages = copy.deepcopy(backend["messages"])
    for msg in messages:
        if "reasoning" in msg:
            msg["reasoning_content"] = msg["reasoning"]
    if backend.get("tools"):
        system = next((m for m in messages if m["role"] == "system"), None)
        if system is None:
            system = {"role": "system", "content": ""}
            messages.insert(0, system)
        system["tools"] = copy.deepcopy(backend["tools"])
    expected = reference_encoder.encode_messages(
        messages, thinking_mode="thinking" if enabled else "chat",
        reasoning_effort=effort if enabled else None, drop_thinking=drop_thinking,
    )
    tokenizer = RecordingTokenizer()
    prefill, decode = PROXY.count_prefill_and_decode_tokens(backend, tokenizer)
    assert tokenizer.prompt == expected
    assert prefill == decode == len(expected.encode("utf-8"))
    if enabled:
        assert f"Reasoning Effort: {dict(low=25, high=50, max=100)[effort]} " in expected
    else:
        assert "Reasoning Effort:" not in expected


def test_pure_completion_count_unchanged_and_prefill_uses_v41():
    class PlainTokenizer:
        def encode(self, text, **kwargs):
            return list(text)
    request = PROXY.prepare_backend_request_data({"prompt": "hello"})
    prefill, decode = PROXY.count_prefill_and_decode_tokens(request, PlainTokenizer())
    assert decode == len("hello")
    assert prefill > decode
    assert "Reasoning Effort: 25 " in PROXY._encode_dsv4_prompt_for_count(request, True)


def test_no_old_encoder_fallback(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("encoding failed")
    monkeypatch.setattr(PROXY, "encode_dsv4_messages", broken)
    with pytest.raises(ValueError, match="DeepSeek-V4.1"):
        PROXY.count_tokens({"messages": [{"role": "user", "content": "hello"}]}, RecordingTokenizer())


def test_multimodal_parameters_use_pd_reasoning(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(PROXY, "global_args", SimpleNamespace(
        multimodal_backend_url="http://unused/v1", multimodal_backend_model="vision",
    ), raising=False)
    chat, state = responses({
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "describe"},
            {"type": "input_image", "image_url": "https://example.com/test.png"},
        ]}],
        "reasoning": {"effort": "medium"},
        "thinking": {"type": "disabled"},
    })
    assert state.backend_route == "pd"
    assert state.contains_images
    assert state.reasoning_settings["thinking_enabled"] is False
    assert "reasoning_effort" not in chat
    assert "thinking" not in chat
    assert chat["chat_template_kwargs"]["enable_thinking"] is False


@pytest.mark.parametrize("path", ["/responses", "/v1/responses"])
def test_responses_route_preserves_raw_body_and_records_reasoning_trace(monkeypatch, tmp_path, path):
    from starlette.requests import Request
    monkeypatch.setattr(PROXY, "RESPONSES_WIRE_TRACE", True)
    monkeypatch.setattr(PROXY, "RESPONSES_WIRE_TRACE_DIR", str(tmp_path))
    original = {"model": "public-model", "input": "hello", "store": False, "reasoning": {"effort": "unknown-effort"}}
    raw = json.dumps(original).encode()
    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}
    request = Request({"type": "http", "method": "POST", "path": path, "headers": [], "client": ("127.0.0.1", 1234)}, receive)
    async def runner(request, chat, state, trace, raw_body):
        assert raw_body == raw
        assert state.reasoning_settings["source"] == "responses"
        assert chat["reasoning_effort"] == "high"
        assert "raw_effort" not in chat
        return PROXY.JSONResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    response = asyncio.run(PROXY.handle_responses(request, backend_runner=runner))
    assert response.status_code == 200
    result = json.loads(response.body)
    directory = tmp_path / result["id"]
    assert json.loads((directory / "request.raw.json").read_text()) == original
    metadata = json.loads((directory / "routing.decision.json").read_text())["reasoning"]
    assert metadata["raw_effort"] == "unknown-effort"
    assert metadata["effective_client_effort"] == "medium"
    assert metadata["backend_effort"] == "high"
    assert metadata["fallback_reason"] == "unknown_effort"


@pytest.mark.parametrize("malformed", [[], False, 1, "high"])
def test_invalid_reasoning_container_is_rejected(malformed):
    with pytest.raises(PROXY.ResponsesRequestError) as exc:
        responses({"reasoning": malformed})
    assert exc.value.code == "invalid_reasoning_effort"


@pytest.mark.parametrize("path,source,expected", [
    ("/responses", "responses", "max"), ("/v1/responses", "responses", "max"),
    ("/chat/completions", "chat", "high"), ("/v1/chat/completions", "chat", "high"),
])
def test_registered_route_aliases_keep_entry_semantics(monkeypatch, path, source, expected):
    monkeypatch.setattr(PROXY, "REQUEST_AUDIT_TRACE", False)
    monkeypatch.setattr(PROXY, "RESPONSES_WIRE_TRACE", False)
    captured = []
    async def handle(api, request, log_api=None, *, response_state=None, **kwargs):
        body = json.loads(await request.body())
        settings = response_state.reasoning_settings if response_state else None
        backend = PROXY.prepare_backend_request_data(body, reasoning_settings=settings)
        captured.append(backend)
        return PROXY.JSONResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
    monkeypatch.setattr(PROXY, "_handle_completions", handle)
    body = {"model": "public-model", "store": False}
    if source == "responses":
        body.update(input="hello", reasoning={"effort": "high"})
    else:
        body.update(messages=[{"role": "user", "content": "hello"}], reasoning_effort="high")
    async def execute():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=PROXY.app), base_url="http://test") as client:
            return await client.post(path, json=body)
    response = asyncio.run(execute())
    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0]["reasoning_effort"] == expected
