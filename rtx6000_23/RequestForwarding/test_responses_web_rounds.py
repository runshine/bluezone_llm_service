"""Responses web rounds: terminal validation, client handoff, and classified failures."""

import asyncio
import copy
import json

import pytest

from test_responses_adapter import (
    PROXY, run, _sse_payload, _tool_search_declaration, _tool_search_output,
)


class NullTrace:
    def __init__(self, *_args):
        pass

    def write(self, *_args):
        pass

    def write_json(self, *_args):
        pass


class MemoryStore:
    enabled = True

    def __init__(self):
        self.responses = {}

    async def put(self, response):
        self.responses[response["id"]] = copy.deepcopy(response)

    async def get(self, response_id):
        return copy.deepcopy(self.responses.get(response_id))


def request(payload):
    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    return PROXY.Request({
        "type": "http", "method": "POST", "path": "/v1/responses",
        "headers": [], "query_string": b"", "scheme": "http",
        "server": ("127.0.0.1", 8077), "client": ("127.0.0.1", 12345),
    }, receive)


def call(name, call_id, arguments="{}"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def chat_response(calls=None, content="", reasoning=""):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_content"] = reasoning
    return {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}]}


async def chat_stream(response):
    choice = response["choices"][0]
    delta = copy.deepcopy(choice["message"])
    for index, tool in enumerate(delta.get("tool_calls", [])):
        tool["index"] = index
    yield f'data: {json.dumps({"choices": [{"delta": delta}]})}\n\n'.encode()
    yield f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": choice["finish_reason"]}]})}\n\n'.encode()
    yield b"data: [DONE]\n\n"


async def decode_response(response):
    if isinstance(response, PROXY.JSONResponse):
        return json.loads(response.body), []
    wire = [chunk async for chunk in response.body_iterator]
    events = [_sse_payload(chunk) for chunk in wire if _sse_payload(chunk) is not None]
    completed = [e["response"] for e in events if e["type"] == "response.completed"]
    if completed:
        assert len(completed) == 1
        assert wire[-1] == b"data: [DONE]\n\n"
        return completed[0], events
    assert b"data: [DONE]\n\n" not in wire
    assert [e["type"] for e in events].count("response.incomplete") == 1
    assert [e["type"] for e in events][-2:] == ["error", "response.incomplete"]
    assert events[-2]["error"]["code"] == events[-1]["response"]["incomplete_details"]["reason"]
    return {"error": events[-2]["error"]}, events


@pytest.mark.parametrize("ending", ["both", "no_done", "no_finish", "neither"])
def test_web_stream_requires_both_terminal_signals(ending):
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        if ending in ("both", "no_done"):
            yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        if ending in ("both", "no_finish"):
            yield b"data: [DONE]\n\n"

    async def check():
        state = PROXY.ResponseRequestState("resp_terminal", "test", 1, False)
        result = PROXY.StreamingResponse(PROXY._convert_web_chat_stream_to_responses_stream(
            chunks(), {"messages": []}, None, state, None,
        ))
        body, events = await decode_response(result)
        if ending == "both":
            assert body["status"] == "completed"
        else:
            assert body["error"]["code"] == "decoder_stream_error"
            assert not any(e["type"] == "response.output_item.done" for e in events)

    run(check())


@pytest.mark.parametrize("failure,code", [
    ("bad_json", "decoder_stream_error"),
    ("bad_shape", "decoder_stream_error"),
    ("read_error", "decoder_stream_error"),
    ("upstream_error", "pd_first_output_timeout"),
    ("unknown_code", "decoder_stream_error"),
])
def test_web_stream_failure_code_and_safe_message(failure, code):
    secret = "sensitive-image-or-tool-arguments"

    async def chunks():
        if failure == "bad_json":
            yield b"data: {broken\n\n"
        elif failure == "bad_shape":
            yield b"data: []\n\n"
        elif failure == "read_error":
            raise OSError(secret)
        else:
            yield f'data: {json.dumps({"error": {"code": code if failure == "upstream_error" else "unknown", "message": secret}})}\n\n'.encode()

    async def check():
        state = PROXY.ResponseRequestState("resp_error", "test", 1, False)
        result = PROXY.StreamingResponse(PROXY._convert_web_chat_stream_to_responses_stream(
            chunks(), {"messages": []}, None, state, None,
        ))
        body, _ = await decode_response(result)
        assert body["error"]["code"] == code
        assert secret not in json.dumps(body)

    run(check())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("web_first", [False, True])
def test_mixed_web_and_client_tools_pause_persist_and_resume(stream, web_first, monkeypatch):
    store = MemoryStore()
    monkeypatch.setattr(PROXY, "response_state_store", store)
    monkeypatch.setattr(PROXY, "WireTrace", NullTrace)
    client_calls = [
        call("__codex_tool_search", "search1", '{"query":"js"}'),
        call("lookup", "lookup1", '{"key":"x"}'),
        call("__codex_custom_node_repl_js", "js1", '{"input":"1 + 1"}'),
    ]
    web = call("__codex_web_search_web_search", "web1", '{"query":"docs"}')
    all_calls = [web, *client_calls] if web_first else [*client_calls, web]
    initial = chat_response(all_calls, "Searching tools.", "Need both sources.")
    web_count = 0
    backend_count = 0

    async def search(_args):
        nonlocal web_count
        web_count += 1
        return {"results": ["web evidence"]}

    async def backend(_request, chat, *_args):
        nonlocal backend_count
        backend_count += 1
        if backend_count == 1:
            result = initial
        else:
            assert backend_count == 2
            calls = [c["id"] for m in chat["messages"] for c in m.get("tool_calls", [])]
            outputs = [m["tool_call_id"] for m in chat["messages"] if m["role"] == "tool"]
            assert calls == ["web1", "search1", "lookup1", "js1"]
            assert outputs == calls
            assert web_count == 1
            result = chat_response(content="All tools completed.")
        if chat.get("stream"):
            return PROXY.StreamingResponse(chat_stream(result), media_type="text/event-stream")
        return PROXY.JSONResponse(result)

    monkeypatch.setattr(PROXY, "execute_web_search", search)

    async def check():
        result = await PROXY.handle_responses(request({
            "model": "test", "store": True, "stream": stream, "input": "Use tools.",
            "tools": [
                _tool_search_declaration(), {"type": "web_search"},
                {"type": "function", "name": "lookup", "parameters": {"type": "object"}},
                {"type": "namespace", "name": "node_repl", "tools": [{
                    "type": "custom", "name": "js", "format": {"type": "text"},
                }]},
            ],
        }), backend_runner=backend)
        body, _ = await decode_response(result)
        assert backend_count == 1 and web_count == 1
        assert [i["call_id"] for i in body["output"] if "call_id" in i] == ["search1", "lookup1", "js1"]
        stored = store.responses[body["id"]]
        messages = stored["_normalized_messages"]
        assert [c["id"] for m in messages for c in m.get("tool_calls", [])] == ["web1", "search1", "lookup1", "js1"]
        assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == ["web1"]
        assert sum(m.get("content") == "Searching tools." for m in messages) == 1
        assert sum(m.get("reasoning_content") == "Need both sources." for m in messages) == 1
        result = await PROXY.handle_responses(request({
            "model": "test", "store": True, "stream": stream, "previous_response_id": body["id"],
            "input": [
                _tool_search_output("search1", tools=[]),
                {"type": "function_call_output", "call_id": "lookup1", "output": "found"},
                {"type": "custom_tool_call_output", "call_id": "js1", "output": "2"},
            ],
        }), backend_runner=backend)
        final, _ = await decode_response(result)
        assert final["output"][-1]["content"][0]["text"] == "All tools completed."
        assert backend_count == 2 and web_count == 1

    run(check())


@pytest.mark.parametrize("invalid,code", [
    ("undeclared", "undeclared_tool"),
    ("arguments", "invalid_function_arguments"),
    ("missing_id", "incomplete_function_call"),
    ("missing_name", "incomplete_function_call"),
    ("duplicate_id", "incomplete_function_call"),
    ("malformed_call", "incomplete_function_call"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_all_mixed_calls_validate_before_web_side_effects(invalid, code, stream, monkeypatch):
    monkeypatch.setattr(PROXY, "WireTrace", NullTrace)
    web = call("__codex_web_search_web_search", "web1", '{"query":"docs"}')
    client = call("lookup", "lookup1")
    if invalid == "undeclared":
        client["function"]["name"] = "not_declared"
    elif invalid == "arguments":
        client["function"]["arguments"] = "{broken"
    elif invalid == "missing_id":
        client.pop("id")
    elif invalid == "missing_name":
        client["function"].pop("name")
    elif invalid == "duplicate_id":
        client["id"] = "web1"
    else:
        client = "invalid"
    response = chat_response([web, client])

    async def search(_args):
        raise AssertionError("No tool may execute before the whole batch validates")

    async def backend(*_args):
        if stream:
            # Malformed list entries cannot be indexed by the test stream helper.
            async def chunks():
                delta = copy.deepcopy(response["choices"][0]["message"])
                for index, tool in enumerate(delta["tool_calls"]):
                    if isinstance(tool, dict):
                        tool["index"] = index
                yield f'data: {json.dumps({"choices": [{"delta": delta, "finish_reason": "tool_calls"}]})}\n\n'.encode()
                yield b"data: [DONE]\n\n"
            return PROXY.StreamingResponse(chunks())
        return PROXY.JSONResponse(response)

    monkeypatch.setattr(PROXY, "execute_web_search", search)

    async def check():
        result = await PROXY.handle_responses(request({
            "model": "test", "store": False, "stream": stream, "input": "Use tools.",
            "tools": [{"type": "web_search"}, {"type": "function", "name": "lookup"}],
        }), backend_runner=backend)
        body, _ = await decode_response(result)
        expected = "decoder_stream_error" if stream and invalid == "malformed_call" else code
        assert body["error"]["code"] == expected
        if not stream:
            assert result.status_code == 502

    run(check())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("failure,code,status", [
    ("backend", "rate_limit_exceeded", 429),
    ("invalid_json", "invalid_decoder_response", 502),
    ("invalid_shape", "invalid_decoder_response", 502),
    ("connection", "decoder_stream_error", 502),
    ("unexpected", "proxy_internal_error", 500),
])
def test_web_continuation_preserves_error_classification(stream, failure, code, status, monkeypatch):
    monkeypatch.setattr(PROXY, "WireTrace", NullTrace)
    secret = "sensitive-image-or-tool-arguments"
    backend_count = 0

    async def search(_args):
        return {"results": []}

    async def backend(_request, chat, *_args):
        nonlocal backend_count
        backend_count += 1
        if backend_count == 1:
            response = chat_response([call("__codex_web_search_web_search", "web1", '{"query":"docs"}')])
            return PROXY.StreamingResponse(chat_stream(response)) if stream else PROXY.JSONResponse(response)
        assert chat["stream"] is False
        if failure == "backend":
            return PROXY.JSONResponse({"error": {"code": code, "message": secret}}, status_code=status)
        if failure == "invalid_json":
            response = PROXY.JSONResponse({})
            response.body = b"{broken"
            return response
        if failure == "invalid_shape":
            return PROXY.JSONResponse([])
        if failure == "connection":
            raise OSError(secret)
        raise RuntimeError(secret)

    monkeypatch.setattr(PROXY, "execute_web_search", search)

    async def check():
        result = await PROXY.handle_responses(request({
            "model": "test", "store": False, "stream": stream, "input": "Search.",
            "tools": [{"type": "web_search"}],
        }), backend_runner=backend)
        body, _ = await decode_response(result)
        assert body["error"]["code"] == code
        assert secret not in json.dumps(body)
        if not stream:
            assert result.status_code == status
        assert backend_count == 2

    run(check())


@pytest.mark.parametrize("stream", [False, True])
def test_web_store_failure_never_reports_success(stream, monkeypatch):
    class BrokenStore(MemoryStore):
        async def put(self, _response):
            raise OSError("private-store-address")

    monkeypatch.setattr(PROXY, "response_state_store", BrokenStore())
    monkeypatch.setattr(PROXY, "WireTrace", NullTrace)

    async def backend(*_args):
        response = chat_response(content="Answer.")
        return PROXY.StreamingResponse(chat_stream(response)) if stream else PROXY.JSONResponse(response)

    async def check():
        result = await PROXY.handle_responses(request({
            "model": "test", "store": True, "stream": stream, "input": "Hello.",
            "tools": [{"type": "web_search"}],
        }), backend_runner=backend)
        body, events = await decode_response(result)
        assert body["error"]["code"] == "response_state_unavailable"
        assert "private-store-address" not in json.dumps(body)
        if stream:
            assert not any(e["type"] == "response.output_item.done" for e in events)
        else:
            assert result.status_code == 503

    run(check())


@pytest.mark.parametrize("stream", [False, True])
def test_pure_web_rounds_continue_and_tool_failure_remains_a_result(stream, monkeypatch):
    monkeypatch.setattr(PROXY, "WireTrace", NullTrace)
    backend_count = 0
    web_count = 0

    async def search(arguments):
        nonlocal web_count
        web_count += 1
        assert json.loads(arguments) == {"query": "docs"}
        if web_count == 1:
            raise PROXY.WebToolError("Search service unavailable.")
        return {"results": ["evidence"]}

    async def backend(_request, chat, *_args):
        nonlocal backend_count
        backend_count += 1
        if backend_count > 1:
            assert chat["stream"] is False
            calls = [c["id"] for m in chat["messages"] for c in m.get("tool_calls", [])]
            results = [m for m in chat["messages"] if m["role"] == "tool"]
            assert calls == [m["tool_call_id"] for m in results]
            assert json.loads(results[0]["content"])["error"]["code"] == "web_tool_error"
        if backend_count <= 2:
            response = chat_response([
                call("__codex_web_search_web_search", f"web{backend_count}", '{"query":"docs"}')
            ], reasoning=f"Round {backend_count}.")
        else:
            assert backend_count == 3
            response = chat_response(content="Recovered search result.")
            response["choices"][0]["message"]["tool_calls"] = None
        if chat.get("stream"):
            return PROXY.StreamingResponse(chat_stream(response))
        return PROXY.JSONResponse(response)

    monkeypatch.setattr(PROXY, "execute_web_search", search)

    async def check():
        result = await PROXY.handle_responses(request({
            "model": "test", "stream": stream, "store": False,
            "input": "Search.", "tools": [{"type": "web_search"}],
        }), backend_runner=backend)
        body, _ = await decode_response(result)
        assert body["output"][-1]["content"][0]["text"] == "Recovered search result."
        assert backend_count == 3 and web_count == 2

    run(check())


def test_web_stream_collection_propagates_cancellation():
    async def chunks():
        raise asyncio.CancelledError()
        yield b""  # Make this an async iterator.

    with pytest.raises(asyncio.CancelledError):
        run(PROXY._collect_chat_stream_response(chunks()))


def test_web_stream_collection_closes_source_on_parse_failure():
    closed = False

    async def chunks():
        nonlocal closed
        try:
            yield b"data: {broken\n\n"
            raise AssertionError("Must not continue after invalid SSE")
        finally:
            closed = True

    async def check():
        with pytest.raises(PROXY.ResponsesBackendError):
            await PROXY._collect_chat_stream_response(chunks())
        assert closed

    run(check())
