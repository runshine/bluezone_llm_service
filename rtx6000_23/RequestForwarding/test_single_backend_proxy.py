import asyncio
import base64
import copy
import json
import os

import httpx
import load_balance_proxy_server as RESPONSES
import single_backend_proxy_server as SINGLE


def run(coro):
    return asyncio.run(coro)


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        return None


def _runner(handler):
    client = httpx.AsyncClient(
        base_url="http://single-vllm.test/v1",
        transport=httpx.MockTransport(handler),
    )
    stats = SINGLE.SingleBackendStats()
    return (
        SINGLE.SingleVllmBackendRunner(client, "DeepSeekv4", 2, 0, stats),
        client,
        stats,
    )


def _responses_request(body: dict):
    encoded = json.dumps(body).encode()

    async def receive_once():
        return {"type": "http.request", "body": encoded, "more_body": False}

    return RESPONSES.Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8078),
    }, receive_once)


def test_single_multimodal_arguments_are_normalized_and_validated():
    args = SINGLE.parse_args([
        "--backend-url", "http://single-vllm.test",
        "--backend-model-name", "qwen",
        "--multimodal-backend-url", "http://vision-vllm.test/",
        "--multimodal-backend-model", "vision-qwen",
    ])
    assert args.backend_url == "http://single-vllm.test/v1"
    assert args.multimodal_backend_url == "http://vision-vllm.test/v1"
    assert args.multimodal_backend_model == "vision-qwen"
    assert args.multimodal_max_image_bytes == 20 * 1024 * 1024

    try:
        SINGLE.parse_args([
            "--backend-url", "http://single-vllm.test/v1",
            "--backend-model-name", "qwen",
            "--multimodal-backend-url", "http://vision-vllm.test/v1",
        ])
    except ValueError as exc:
        assert "must be provided together" in str(exc)
    else:
        raise AssertionError("expected incomplete multimodal configuration to fail")


def test_single_responses_input_image_uses_existing_runner_and_forced_model():
    captured = {}

    async def handler(request):
        captured.update(json.loads((await request.aread()).decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "blank image"}, "finish_reason": "stop"}]},
            request=request,
        )

    async def execute():
        runner, client, _stats = _runner(handler)
        previous_args = getattr(RESPONSES, "global_args", None)
        RESPONSES.global_args = type("Args", (), {
            "responses_sse_heartbeat_seconds": 0,
            "multimodal_backend_url": "http://vision-vllm.test/v1",
            "multimodal_backend_model": "vision-qwen",
            "multimodal_max_image_bytes": 1024 * 1024,
        })()
        try:
            return await RESPONSES.handle_responses(
                _responses_request({
                    "model": "client-visible-model",
                    "input": [{"role": "user", "content": [
                        {"type": "input_text", "text": "Describe this image."},
                        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
                    ]}],
                    "store": False,
                }),
                backend_runner=runner,
            )
        finally:
            await client.aclose()
            if previous_args is None:
                delattr(RESPONSES, "global_args")
            else:
                RESPONSES.global_args = previous_args

    result = run(execute())
    assert result.status_code == 200
    assert captured["model"] == "DeepSeekv4"
    content = captured["messages"][0]["content"]
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
    }


def test_single_responses_input_image_enforces_configured_size_limit():
    previous_args = getattr(RESPONSES, "global_args", None)
    RESPONSES.global_args = type("Args", (), {
        "multimodal_backend_url": "http://vision-vllm.test/v1",
        "multimodal_backend_model": "vision-qwen",
        "multimodal_max_image_bytes": 3,
    })()
    try:
        try:
            run(RESPONSES.responses_request_to_chat_request({
                "model": "client-visible-model",
                "input": [{"role": "user", "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64," + base64.b64encode(b"four").decode(),
                }]}],
            }))
        except RESPONSES.ResponsesRequestError as exc:
            assert exc.code == "multimodal_image_too_large"
        else:
            raise AssertionError("expected oversized image to be rejected")
    finally:
        if previous_args is None:
            delattr(RESPONSES, "global_args")
        else:
            RESPONSES.global_args = previous_args


def test_single_backend_forces_model_and_returns_nonstream_chat_json():
    captured = {}

    async def handler(request):
        captured["json"] = json.loads((await request.aread()).decode())
        return httpx.Response(
            200,
            json={"id": "chatcmpl_single", "choices": [{"message": {"content": "ok"}}]},
            request=request,
        )

    async def execute():
        runner, client, stats = _runner(handler)
        try:
            response = await runner.request_chat(
                {"model": "client-alias", "messages": [{"role": "user", "content": "hi"}]}
            )
            return response, stats
        finally:
            await client.aclose()

    response, stats = run(execute())
    assert response.status_code == 200
    assert json.loads(response.body)["id"] == "chatcmpl_single"
    assert captured["json"]["model"] == "DeepSeekv4"
    assert stats.response_headers_received == 1


def test_single_backend_preserves_upstream_stream_chunks():
    source_chunks = [b"data: first\n\n", b"data: second\n\n", b"data: [DONE]\n\n"]

    async def handler(request):
        return httpx.Response(200, stream=ChunkStream(source_chunks), request=request)

    async def execute():
        runner, client, stats = _runner(handler)
        try:
            response = await runner.request_chat({"model": "x", "stream": True, "messages": []})
            received = [chunk async for chunk in response.body_iterator]
            return received, stats
        finally:
            await client.aclose()

    received, stats = run(execute())
    assert received == source_chunks
    assert stats.completed_streams == 1
    assert stats.first_sse_received == 1
    assert stats.active_streams == 0


def test_single_backend_retries_pre_header_5xx_but_not_4xx():
    calls = []

    async def handler(request):
        calls.append(request)
        status = 503 if len(calls) == 1 else 200
        return httpx.Response(status, json={"id": "chatcmpl_retry"}, request=request)

    async def execute():
        runner, client, stats = _runner(handler)
        try:
            response = await runner.request_chat({"model": "x", "messages": []})
            return response, stats
        finally:
            await client.aclose()

    response, stats = run(execute())
    assert response.status_code == 200
    assert len(calls) == 2
    assert stats.response_headers_received == 2

    calls.clear()

    async def client_error(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"code": "bad_request"}}, request=request)

    async def execute_client_error():
        runner, client, _stats = _runner(client_error)
        try:
            return await runner.request_chat({"model": "x", "messages": []})
        finally:
            await client.aclose()

    response = run(execute_client_error())
    assert response.status_code == 400
    assert len(calls) == 1


def test_single_backend_returns_last_5xx_and_transport_error_after_retries():
    server_calls = []

    async def always_unavailable(request):
        server_calls.append(request)
        return httpx.Response(503, json={"error": {"code": "unavailable"}}, request=request)

    async def execute_server_error():
        runner, client, stats = _runner(always_unavailable)
        try:
            return await runner.request_chat({"model": "x", "messages": []}), stats
        finally:
            await client.aclose()

    response, stats = run(execute_server_error())
    assert response.status_code == 503
    assert len(server_calls) == 2
    assert stats.errors == 1

    transport_calls = []

    async def transport_error(request):
        transport_calls.append(request)
        raise httpx.ConnectError("unreachable", request=request)

    async def execute_transport_error():
        runner, client, stats = _runner(transport_error)
        try:
            return await runner.request_chat({"model": "x", "messages": []}), stats
        finally:
            await client.aclose()

    response, stats = run(execute_transport_error())
    assert response.status_code == 502
    assert json.loads(response.body)["error"]["code"] == "upstream_transport_error"
    assert len(transport_calls) == 2
    assert stats.errors == 1


def test_single_backend_retries_stream_5xx_before_returning_a_body():
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, stream=ChunkStream([b"unavailable"]), request=request)
        return httpx.Response(200, stream=ChunkStream([b"data: [DONE]\n\n"]), request=request)

    async def execute():
        runner, client, stats = _runner(handler)
        try:
            response = await runner.request_chat({"model": "x", "stream": True, "messages": []})
            return [chunk async for chunk in response.body_iterator], stats
        finally:
            await client.aclose()

    chunks, stats = run(execute())
    assert chunks == [b"data: [DONE]\n\n"]
    assert len(calls) == 2
    assert stats.response_headers_received == 2


def test_single_backend_closes_upstream_stream_when_client_stops_reading():
    closed = []

    class ClosingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            await asyncio.sleep(60)

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        return httpx.Response(200, stream=ClosingStream(), request=request)

    async def execute():
        runner, client, stats = _runner(handler)
        try:
            response = await runner.request_chat({"model": "x", "stream": True, "messages": []})
            iterator = response.body_iterator
            assert await anext(iterator) == b"data: first\n\n"
            await iterator.aclose()
            return stats
        finally:
            await client.aclose()

    stats = run(execute())
    assert closed == [True]
    assert stats.active_streams == 0


def test_responses_handler_uses_injected_single_backend_runner():
    async def backend_runner(_request, chat_request, _state, _trace, _raw_body):
        assert chat_request["model"] == "client-visible-model"
        return RESPONSES.JSONResponse(
            content={
                "choices": [{"message": {"content": "single response"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        )

    async def receive_once():
        return {
            "type": "http.request",
            "body": b'{"model":"client-visible-model","input":"hello","store":false}',
            "more_body": False,
        }

    request = RESPONSES.Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8078),
    }, receive_once)
    result = run(RESPONSES.handle_responses(request, backend_runner=backend_runner))
    payload = json.loads(result.body)
    assert result.status_code == 200
    assert payload["object"] == "response"
    assert payload["output"][0]["content"][0]["text"] == "single response"


def test_single_backend_responses_custom_tool_uses_forced_model():
    captured = {}

    async def handler(request):
        captured.update(json.loads((await request.aread()).decode()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"tool_calls": [{
                "id": "call_patch",
                "type": "function",
                "function": {
                    "name": "__codex_custom_apply_patch",
                    "arguments": '{"input":"*** Begin Patch"}',
                },
            }]}, "finish_reason": "tool_calls"}]},
            request=request,
        )

    async def execute():
        runner, client, _stats = _runner(handler)
        try:
            return await RESPONSES.handle_responses(
                _responses_request({
                    "model": "client-visible-model",
                    "input": "Apply the patch.",
                    "tools": [{"type": "custom", "name": "apply_patch"}],
                    "store": False,
                }),
                backend_runner=runner,
            )
        finally:
            await client.aclose()

    result = run(execute())
    payload = json.loads(result.body)
    assert captured["model"] == "DeepSeekv4"
    assert payload["output"][0]["type"] == "custom_tool_call"
    assert payload["output"][0]["name"] == "apply_patch"
    assert payload["output"][0]["input"] == "*** Begin Patch"


def test_single_backend_responses_previous_response_id_uses_state_store():
    class MemoryStateStore:
        enabled = True

        def __init__(self):
            self.responses = {}

        async def get(self, response_id):
            return self.responses.get(response_id)

        async def put(self, response):
            self.responses[response["id"]] = response

        async def put_conversation(self, _conversation_id, _response_id):
            return None

    calls = []

    async def backend_runner(_request, chat_request, _state, _trace, _raw_body):
        calls.append(copy.deepcopy(chat_request))
        return RESPONSES.JSONResponse(content={
            "choices": [{"message": {"content": f"turn-{len(calls)}"}, "finish_reason": "stop"}],
        })

    previous_store = RESPONSES.response_state_store
    previous_default = os.environ.get("RESPONSES_DEFAULT_STORE")
    try:
        RESPONSES.response_state_store = MemoryStateStore()
        os.environ["RESPONSES_DEFAULT_STORE"] = "true"
        first = run(RESPONSES.handle_responses(
            _responses_request({"model": "client-visible-model", "input": "Remember 42."}),
            backend_runner=backend_runner,
        ))
        first_payload = json.loads(first.body)
        second = run(RESPONSES.handle_responses(
            _responses_request({
                "model": "client-visible-model",
                "previous_response_id": first_payload["id"],
                "input": "What should you remember?",
            }),
            backend_runner=backend_runner,
        ))
    finally:
        RESPONSES.response_state_store = previous_store
        if previous_default is None:
            os.environ.pop("RESPONSES_DEFAULT_STORE", None)
        else:
            os.environ["RESPONSES_DEFAULT_STORE"] = previous_default

    assert second.status_code == 200
    assert [message["role"] for message in calls[1]["messages"]] == [
        "user", "assistant", "user"
    ]
    assert calls[1]["messages"][-1]["content"] == "What should you remember?"


def test_responses_stream_uses_injected_single_backend_runner():
    called = []

    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"streamed"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def backend_runner(_request, _chat_request, _state, _trace, _raw_body):
        called.append(True)
        return RESPONSES.StreamingResponse(chunks(), media_type="text/event-stream")

    async def receive_once():
        return {
            "type": "http.request",
            "body": b'{"model":"client-visible-model","input":"hello","stream":true,"store":false}',
            "more_body": False,
        }

    request = RESPONSES.Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8078),
    }, receive_once)

    async def execute():
        result = await RESPONSES.handle_responses(request, backend_runner=backend_runner)
        return [chunk async for chunk in result.body_iterator]

    wire = run(execute())
    assert called == [True]
    assert any(b'"type": "response.output_text.delta"' in chunk for chunk in wire)
    assert wire[-1] == b"data: [DONE]\n\n"


def test_web_search_continuation_uses_injected_runner():
    calls = []

    async def backend_runner(_request, chat_request, _state, _trace, _raw_body):
        calls.append(copy.deepcopy(chat_request))
        return RESPONSES.JSONResponse(
            content={"choices": [{"message": {"content": "web result"}, "finish_reason": "stop"}]}
        )

    async def fake_web_search(_arguments):
        return {"results": [{"title": "result"}]}

    class Trace:
        def write(self, *_args, **_kwargs):
            pass

        def write_json(self, *_args, **_kwargs):
            pass

    previous_web_search = RESPONSES.execute_web_search
    RESPONSES.execute_web_search = fake_web_search
    try:
        state = RESPONSES.ResponseRequestState(
            "resp_web", "DeepSeekv4", 1, False,
            tool_registry={"web_search": {"source_type": "web_search", "execution": "proxy"}},
        )
        initial = {
            "choices": [{"message": {"content": "", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query":"test"}'},
            }]}}]
        }
        result = run(RESPONSES._run_proxy_web_tool_rounds(
            initial,
            {"model": "DeepSeekv4", "stream": False, "messages": []},
            None,
            state,
            Trace(),
            backend_runner,
        ))
    finally:
        RESPONSES.execute_web_search = previous_web_search

    assert result["choices"][0]["message"]["content"] == "web result"
    assert len(calls) == 1
    assert calls[0]["messages"][-1]["role"] == "tool"


def test_streaming_web_search_continuation_uses_injected_runner():
    calls = []

    async def initial_chunks():
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_web","type":"function","function":{"name":"__codex_web_search","arguments":"{\\"query\\":\\"test\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def backend_runner(_request, chat_request, _state, _trace, _raw_body):
        calls.append(copy.deepcopy(chat_request))
        return RESPONSES.JSONResponse(content={
            "choices": [{"message": {"content": "web answer"}, "finish_reason": "stop"}],
        })

    async def fake_web_search(_arguments):
        return {"results": [{"title": "result"}]}

    class Trace:
        def write(self, *_args, **_kwargs):
            pass

        def write_json(self, *_args, **_kwargs):
            pass

    previous_web_search = RESPONSES.execute_web_search
    RESPONSES.execute_web_search = fake_web_search
    try:
        state = RESPONSES.ResponseRequestState(
            "resp_stream_web", "DeepSeekv4", 1, False,
            tool_registry={"__codex_web_search": {
                "source_type": "web_search", "execution": "proxy",
            }},
        )

        async def collect():
            return [
                chunk async for chunk in RESPONSES._convert_web_chat_stream_to_responses_stream(
                    initial_chunks(), {"model": "DeepSeekv4", "stream": True, "messages": []},
                    None, state, Trace(), emit_created=False, backend_runner=backend_runner,
                )
            ]

        wire = run(collect())
    finally:
        RESPONSES.execute_web_search = previous_web_search

    assert len(calls) == 1
    assert calls[0]["stream"] is False
    assert calls[0]["messages"][-1]["role"] == "tool"
    assert wire[-1] == b"data: [DONE]\n\n"
