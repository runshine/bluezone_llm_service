"""Strict repairs for Decoder artifacts; no generated code is executed."""

import copy
import json

import pytest

from test_responses_adapter import PROXY, run, _codex_functions_exec_request, _sse_payload
from test_responses_web_rounds import chat_response, chat_stream, call, NullTrace


PATCH = "*** Begin Patch\n*** Add File: example.txt\n+hello\n*** End Patch"
COMMAND = {"cmd": "pwd", "workdir": "/workspace", "yield_time_ms": 10000,
           "max_output_tokens": 3000}
DESCRIPTION = "declare const tools: { exec_command(args: { cmd: string }): Promise<unknown>; };"


def request_body():
    body = _codex_functions_exec_request()
    body["store"] = False
    body["input"][0]["tools"][0]["tools"][0]["description"] = DESCRIPTION
    return body


def declared_command():
    return {"type": "function", "name": "exec_command",
            "parameters": copy.deepcopy(PROXY.CODEX_EXEC_COMMAND_PARAMETERS)}


def request_user_input_body():
    body = request_body()
    body["input"][0]["tools"][0]["tools"].append({
        "type": "function",
        "name": "request_user_input",
        "description": "Request one user choice.",
        "parameters": {
            "type": "object",
            "properties": {"questions": {"type": "array"}},
            "required": ["questions"],
            "additionalProperties": False,
        },
    })
    return body


def state_for(body=None):
    return run(PROXY.responses_request_to_chat_request(body or request_body()))


@pytest.mark.parametrize("mode", ["normal", "stream", "web", "web_stream"])
@pytest.mark.parametrize("alias,input_value", [
    ("__codex_patch", PATCH),
    ("__codex_apply_patch", PATCH),
    ("codex_apply_patch", PATCH),
    ("__codex_builtin_apply_patch", PATCH),
    ("__codex_mcp__exec", "text(1);"),
    ("__codex_mcp__codegen_tool_exec", "text(2);"),
    ("codex_functions.exec", "text(3);"),
    ("__codex_mcp_exec", "text(4);"),
    ("__codex_mcp__codex_namespace_functions_exec", "text(5);"),
    ("__codex_exec_code", "text(6);"),
    ("codex.exec_command", json.dumps(COMMAND)),
    ("__codex_exec_command", json.dumps(COMMAND)),
    ("__codex_mcp__exec_command", json.dumps(COMMAND)),
])
def test_alias_roundtrip_across_conversion_paths(mode, alias, input_value):
    async def collect():
        chat, state = await PROXY.responses_request_to_chat_request(request_body())
        upstream = chat_response([call(alias, "call_preserved", json.dumps({"input": input_value}))])
        before = copy.deepcopy(upstream)
        events = []
        if mode == "normal":
            final = PROXY.chat_result_to_response_object(upstream, chat, state)
        elif mode == "web":
            canonical = await PROXY._run_proxy_web_tool_rounds(
                upstream, chat, None, state, None,
            )
            final = PROXY.chat_result_to_response_object(canonical, chat, state)
        else:
            iterator = (
                PROXY.convert_chat_stream_to_responses_stream(chat_stream(upstream), state, chat)
                if mode == "stream" else
                PROXY._convert_web_chat_stream_to_responses_stream(
                    chat_stream(upstream), chat, None, state, NullTrace(),
                )
            )
            events = [_sse_payload(chunk) async for chunk in iterator
                      if _sse_payload(chunk) is not None]
            assert events[-1]["type"] == "response.completed"
            final = events[-1]["response"]
        assert upstream == before
        return final

    final = run(collect())
    item = final["output"][0]
    assert item["call_id"] == "call_preserved"
    assert item["type"] == "custom_tool_call"
    assert item["name"] == "exec" and item["namespace"] == "functions"
    if alias in PROXY.CODEX_PATCH_DECODER_ALIASES:
        assert item["input"] == f"await tools.apply_patch({json.dumps(PATCH, ensure_ascii=False)});"
    elif alias in PROXY.CODEX_EXEC_COMMAND_DECODER_ALIASES:
        assert item["input"] == "text(await tools.exec_command(" + json.dumps(COMMAND) + "));"
    else:
        assert item["input"] == input_value
    assert alias not in json.dumps(final)


@pytest.mark.parametrize("alias,value", [
    ("__codex_patch", PATCH), ("__codex_exec_command", json.dumps(COMMAND)),
])
@pytest.mark.parametrize("stream", [False, True])
def test_declared_direct_target_has_priority(alias, value, stream):
    body = request_body()
    body["tools"] = ([{"type": "custom", "name": "apply_patch"}]
                     if alias == "__codex_patch" else [declared_command()])

    async def collect():
        chat, state = await PROXY.responses_request_to_chat_request(body)
        upstream = chat_response([call(alias, "call_direct", json.dumps({"input": value}))])
        if not stream:
            return PROXY.chat_result_to_response_object(upstream, chat, state)
        events = [_sse_payload(chunk) async for chunk in
                  PROXY.convert_chat_stream_to_responses_stream(chat_stream(upstream), state, chat)
                  if _sse_payload(chunk) is not None]
        assert events[-1]["type"] == "response.completed"
        return events[-1]["response"]

    item = run(collect())["output"][0]
    assert item["call_id"] == "call_direct"
    if alias in PROXY.CODEX_PATCH_DECODER_ALIASES:
        assert item["name"] == "apply_patch" and item["input"] == PATCH
    else:
        assert item["name"] == "exec_command"
        assert json.loads(item["arguments"]) == COMMAND


def test_request_user_input_namespace_alias_requires_unique_declared_target():
    body = request_user_input_body()
    _, state = state_for(body)
    arguments = json.dumps({"questions": []})
    spec, normalized, deferred = PROXY._normalize_decoder_tool_call(
        "codex_namespace_functions_request_user_input", arguments, "call_input", state,
    )
    assert spec["source_name"] == "request_user_input"
    assert spec["namespace"] == "functions"
    assert json.loads(normalized) == {"questions": []}
    assert deferred is False

    _, undeclared_state = state_for()
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(
            "codex_namespace_functions_request_user_input", arguments, "call_input",
            undeclared_state,
        )
    assert error.value.code == "undeclared_tool"


@pytest.mark.parametrize("alias,args", [
    ("__codex_patch", {"input": "*** Begin Patch\n+unfinished"}),
    ("__codex_patch", {"input": PATCH.replace("\n", "\\n")}),
    ("__codex_patch", {"input": PATCH, "extra": 1}),
    ("__codex_exec_command", {"input": "const result = await tools."}),
    ("__codex_exec_command", {"input": json.dumps({"input": json.dumps(COMMAND)})}),
    ("__codex_exec_command", {"input": json.dumps(dict(COMMAND, extra=True))}),
    ("__codex_exec_command", {"input": json.dumps(dict(COMMAND, yield_time_ms=True))}),
    ("__codex_exec_command", {"input": json.dumps(dict(COMMAND, workdir=[]))}),
    ("__codex_exec_command", {"input": json.dumps(dict(COMMAND, cmd=" "))}),
    ("__codex_exec_command", {"input": {"cmd": "pwd"}}),
    ("__codex_mcp__exec", {"input": "<tool_call>text(1)</tool_call>"}),
    ("__codex_mcp__exec", {"input": "text(1)", "extra": 1}),
    ("__codex_mcp__codegen_tool_exec", {"input": False}),
    ("__codex_mcp__codegen_tool_exec", {"input": ""}),
])
def test_bad_alias_arguments_preserve_precise_stream_failure(alias, args):
    async def collect():
        chat, state = await PROXY.responses_request_to_chat_request(request_body())
        upstream = chat_response([call(alias, "bad", json.dumps(args))])
        with pytest.raises(PROXY.ResponsesRequestError) as error:
            PROXY.chat_result_to_response_object(upstream, chat, state)
        assert error.value.code == "invalid_function_arguments"
        return [_sse_payload(chunk) async for chunk in
                PROXY.convert_chat_stream_to_responses_stream(chat_stream(upstream), state, chat)
                if _sse_payload(chunk) is not None]

    events = run(collect())
    assert events[-1]["type"] == "response.incomplete"
    assert events[-1]["response"]["incomplete_details"]["reason"] == "invalid_function_arguments"
    assert next(e for e in events if e["type"] == "error")["error"]["code"] == "invalid_function_arguments"
    assert not any(e["type"] in {"response.completed", "response.output_item.done"} for e in events)


@pytest.mark.parametrize("alias,value", [
    ("__codex_patch", PATCH), ("__codex_exec_command", json.dumps(COMMAND)),
    ("__codex_mcp__exec", "text(1);"),
    ("__codex_mcp__codegen_tool_exec", "text(1);"),
])
def test_alias_requires_declared_target_and_real_call_id(alias, value):
    _, state = state_for({"model": "DSv4", "input": "hello", "store": False})
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(alias, json.dumps({"input": value}), "call", state)
    assert error.value.code == "undeclared_tool"
    chat, state = state_for()
    upstream = chat_response([call(alias, None, json.dumps({"input": value}))])
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY.chat_result_to_response_object(upstream, chat, state)
    assert error.value.code == "incomplete_function_call"


def test_nested_command_requires_documented_contract_and_checks_direct_schema():
    body = _codex_functions_exec_request()
    body["input"][0]["tools"][0]["tools"][0]["description"] = "Run JavaScript."
    _, state = state_for(body)
    args = json.dumps({"input": json.dumps(COMMAND)})
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call("__codex_exec_command", args, "call", state)
    assert error.value.code == "undeclared_tool"
    body["tools"] = [declared_command()]
    body["tools"][0]["parameters"]["properties"]["cmd"] = {"enum": ["only_this"]}
    _, state = state_for(body)
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call("__codex_exec_command", args, "call", state)
    assert error.value.code == "invalid_function_arguments"


@pytest.mark.parametrize("target", ["exec_command", "apply_patch"])
def test_ambiguous_direct_target_does_not_fall_back_to_exec(target):
    body = request_body()
    tool = declared_command() if target == "exec_command" else {
        "type": "custom", "name": "apply_patch",
    }
    body["tools"] = [tool, {"type": "namespace", "name": "functions",
                            "tools": [copy.deepcopy(tool)]}]
    _, state = state_for(body)
    alias = "__codex_exec_command" if target == "exec_command" else "__codex_patch"
    value = json.dumps(COMMAND) if target == "exec_command" else PATCH
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(alias, json.dumps({"input": value}), "call", state)
    assert error.value.code == "undeclared_tool"


def test_command_schema_never_loads_external_references():
    body = request_body()
    tool = declared_command()
    tool["parameters"]["properties"]["cmd"] = {"$ref": "https://invalid.example/schema"}
    body["tools"] = [tool]
    _, state = state_for(body)
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(
            "__codex_exec_command", json.dumps({"input": json.dumps(COMMAND)}), "call", state,
        )
    assert error.value.code == "invalid_function_arguments"


def test_command_rejection_is_logged_once(caplog):
    _, state = state_for()
    with pytest.raises(PROXY.ResponsesRequestError):
        PROXY._normalize_decoder_tool_call(
            "__codex_exec_command", '{"input":"pwd"}', "call", state,
        )
    assert caplog.text.count("codex_nested_tool_rejected") == 1


def test_stream_repairs_only_after_complete_arguments_and_logs_summary(caplog):
    async def collect():
        chat, state = await PROXY.responses_request_to_chat_request(request_body())
        arguments = json.dumps({"input": json.dumps(COMMAND)})

        async def chunks():
            for index, part in enumerate([arguments[:15], arguments[15:]]):
                delta = {"index": 0, "function": {"arguments": part}}
                if index == 0:
                    delta.update(id="fragmented", type="function")
                    delta["function"]["name"] = "__codex_exec_command"
                payload = {"choices": [{"delta": {"tool_calls": [delta]},
                                        "finish_reason": "tool_calls" if index else None}]}
                yield f"data: {json.dumps(payload)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return [_sse_payload(chunk) async for chunk in
                PROXY.convert_chat_stream_to_responses_stream(chunks(), state, chat)
                if _sse_payload(chunk) is not None]

    with caplog.at_level("INFO"):
        events = run(collect())
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["call_id"] == "fragmented"
    assert "conversion_type=command_json" in caplog.text
    assert "mapped_tool=__codex_custom_functions_exec" in caplog.text
    assert COMMAND["workdir"] not in caplog.text


@pytest.mark.parametrize("alias", ["__codex_tool", "__codex_functions_placeholder",
                                  "__codex_bash", "codex_dummy", "__codex_builtin_tool",
                                  'codex_internal_context source="goal', "__codex_mcp__unknown"])
def test_unknown_or_ambiguous_semantics_remain_rejected(alias):
    _, state = state_for()
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(alias, '{"input":"text(1);"}', "call", state)
    assert error.value.code == "undeclared_tool"
