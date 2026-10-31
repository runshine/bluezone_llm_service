import asyncio
from contextlib import suppress
import importlib.util
import json
import os
import time

import pytest


SPEC = importlib.util.spec_from_file_location(
    "load_balance_proxy_server",
    "load_balance_proxy_server.py",
)
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


def run(coro):
    return asyncio.run(coro)


def test_string_input_and_parameters():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "instructions": "Be concise.",
                "input": "Hello",
                "max_output_tokens": 32,
                "reasoning": {"effort": "medium"},
            }
        )
    )
    assert chat["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    assert chat["max_tokens"] == 16384
    assert chat["reasoning_effort"] == "high"
    assert state.reasoning_effort == "medium"
    assert state.response_id.startswith("resp_")


def test_responses_minimal_reasoning_effort_maps_to_backend_low():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": "Think briefly.",
                "reasoning": {"effort": "minimal"},
            }
        )
    )

    backend = PROXY.prepare_backend_request_data(chat)

    assert state.reasoning_effort == "low"
    assert state.reasoning_settings["raw_effort"] == "minimal"
    assert chat["reasoning_effort"] == "low"
    assert backend["reasoning_effort"] == "low"


def test_responses_keeps_public_model_while_backend_model_is_forced():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = type("Args", (), {"backend_model_name": "DeepSeekv4"})()
        chat, state = run(
            PROXY.responses_request_to_chat_request(
                {"model": "gpt-5.6-luna", "input": "Hello"}
            )
        )
        backend = PROXY.prepare_backend_request_data(chat)
        assert state.model == "gpt-5.6-luna"
        assert backend["model"] == "DeepSeekv4"
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_responses_default_output_budget_is_agent_safe():
    chat, _ = run(PROXY.responses_request_to_chat_request({"model": "DSv4", "input": "Use a tool."}))
    assert chat["max_tokens"] == 16384


def test_responses_rejects_object_discriminator_fields_without_invalid_json():
    cases = [
        (
            {"input": [{"role": "user", "content": [{"type": {"bad": True}}]}]},
            "invalid_input_content",
        ),
        (
            {"input": [{"type": {"bad": True}, "content": "x"}]},
            "invalid_input",
        ),
        (
            {"input": [{"role": {"bad": True}, "content": "x"}]},
            "invalid_input_role",
        ),
        (
            {"tools": [{"type": {"bad": True}, "name": "tool"}]},
            "invalid_tool",
        ),
        (
            {"tool_choice": {"type": {"bad": True}, "name": "tool"}},
            "invalid_tool_choice",
        ),
        (
            {"reasoning": {"effort": {"bad": True}}},
            "invalid_reasoning_effort",
        ),
    ]
    for extra, expected_code in cases:
        request = {"model": "DSv4", "input": "Hello", **extra}
        try:
            run(PROXY.responses_request_to_chat_request(request))
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == expected_code
        except TypeError as exc:
            raise AssertionError("Object discriminator must not raise TypeError") from exc
        else:
            raise AssertionError(f"Expected {expected_code}")


def test_responses_preserves_output_budget_above_16k():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {"model": "DSv4", "input": "Use a tool.", "max_output_tokens": 20000}
        )
    )
    assert chat["max_tokens"] == 20000


def test_input_items_and_tools():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {"role": "developer", "content": [{"type": "input_text", "text": "Rules"}]},
                    {"role": "user", "content": [{"type": "input_text", "text": "Call weather"}]},
                ],
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {"type": "object"},
                    }
                ],
            }
        )
    )
    assert chat["messages"][0]["role"] == "developer"
    assert chat["messages"][0]["tools"][0]["function"]["name"] == "get_weather"
    assert chat["messages"][1]["content"] == [{"type": "text", "text": "Call weather"}]


def _tool_search_declaration():
    return {
        "type": "tool_search",
        "execution": "client",
        "description": "Find relevant project tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _tool_search_call(call_id="call_search", arguments=None):
    return {
        "id": "tsc_search",
        "type": "tool_search_call",
        "execution": "client",
        "call_id": call_id,
        "status": "completed",
        "arguments": arguments if arguments is not None else {"query": "shipping ETA", "limit": 5},
    }


def _tool_search_output(call_id="call_search", tools=None):
    return {
        "type": "tool_search_output",
        "execution": "client",
        "call_id": call_id,
        "status": "completed",
        "tools": tools if tools is not None else [{
            "type": "function",
            "name": "get_shipping_eta",
            "description": "Look up an order ETA.",
            "defer_loading": True,
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }],
    }


def test_client_tool_search_declaration_becomes_internal_chat_function():
    chat, state = run(PROXY.responses_request_to_chat_request({
        "model": "gpt-5.6-luna",
        "input": "Find an ETA tool.",
        "tools": [_tool_search_declaration()],
    }))

    assert chat["tools"] == [{
        "type": "function",
        "function": {
            "name": "__codex_tool_search",
            "description": "Find relevant project tools.",
            "parameters": _tool_search_declaration()["parameters"],
        },
    }]
    assert state.tool_registry["__codex_tool_search"]["source_type"] == "tool_search"
    assert state.tool_registry["__codex_tool_search"]["execution"] == "client"


def test_tool_search_rejects_hosted_or_unspecified_execution():
    for execution in (None, "server", "other"):
        declaration = _tool_search_declaration()
        if execution is None:
            declaration.pop("execution")
        else:
            declaration["execution"] = execution
        try:
            run(PROXY.responses_request_to_chat_request({
                "model": "DSv4", "input": "Find a tool.", "tools": [declaration],
            }))
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == "unsupported_tool_execution"
            assert exc.status_code == 400
        else:
            raise AssertionError("Non-client tool_search must fail closed")


def test_unsupported_tool_search_execution_returns_400_before_backend_dispatch():
    async def check():
        body = json.dumps({
            "model": "DSv4",
            "input": "Find a tool.",
            "tools": [{**_tool_search_declaration(), "execution": "server"}],
        }).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        request = PROXY.Request({
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8077),
        }, receive)
        backend_called = False

        async def backend(*args):
            nonlocal backend_called
            backend_called = True
            raise AssertionError("Unsupported tool_search must not dispatch")

        response = await PROXY.handle_responses(request, backend_runner=backend)
        assert response.status_code == 400
        assert json.loads(response.body)["error"]["code"] == "unsupported_tool_execution"
        assert backend_called is False

    run(check())


def test_deferred_tools_are_not_exposed_until_search_output_loads_them():
    chat, state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4",
        "input": "Find a tool.",
        "tools": [
            _tool_search_declaration(),
            {
                "type": "function",
                "name": "deferred_lookup",
                "description": "Deferred lookup.",
                "defer_loading": True,
                "parameters": {"type": "object"},
            },
        ],
    }))
    assert [tool["function"]["name"] for tool in chat["tools"]] == [
        "__codex_tool_search"
    ]
    assert "deferred_lookup" not in state.tool_registry


def test_tool_search_decoder_call_is_returned_as_responses_item():
    chat, state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4", "input": "Find a tool.", "tools": [_tool_search_declaration()],
    }))
    response = PROXY.chat_result_to_response_object({
        "choices": [{"message": {"tool_calls": [{
            "id": "call_search",
            "type": "function",
            "function": {
                "name": "__codex_tool_search",
                "arguments": '{"query":"shipping ETA","limit":5}',
            },
        }]}}],
    }, chat, state)

    assert response["output"] == [{
        "id": response["output"][0]["id"],
        "type": "tool_search_call",
        "execution": "client",
        "call_id": "call_search",
        "status": "completed",
        "arguments": {"query": "shipping ETA", "limit": 5},
    }]
    saved = PROXY._state_payload(response, chat, state)
    assert saved["_normalized_messages"][-1]["tool_calls"][0]["function"]["name"] == (
        "__codex_tool_search"
    )


def test_tool_search_output_loads_tools_and_restores_chat_history():
    search_call = _tool_search_call()
    search_output = _tool_search_output()
    chat, state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4",
        "input": [search_call, search_output, {"role": "user", "content": "Use it."}],
        "tools": [_tool_search_declaration()],
    }))

    assert [tool["function"]["name"] for tool in chat["tools"]] == [
        "__codex_tool_search", "get_shipping_eta"
    ]
    assert state.tool_registry["get_shipping_eta"]["source_name"] == "get_shipping_eta"
    assistant_call, tool_result, user_message = chat["messages"][-3:]
    assert assistant_call["tool_calls"][0]["id"] == "call_search"
    assert assistant_call["tool_calls"][0]["function"]["name"] == "__codex_tool_search"
    assert tool_result == {
        "role": "tool",
        "tool_call_id": "call_search",
        "content": "Loaded tools: get_shipping_eta",
    }
    assert user_message["content"] == "Use it."


def test_tool_search_accepts_empty_and_namespace_results():
    empty_chat, _ = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4",
        "input": [_tool_search_call(), _tool_search_output(tools=[])],
        "tools": [_tool_search_declaration()],
    }))
    assert empty_chat["messages"][-1]["content"] == "Tool search completed with no tools loaded."
    assert [tool["function"]["name"] for tool in empty_chat["tools"]] == [
        "__codex_tool_search"
    ]

    namespace_result = _tool_search_output(tools=[{
        "type": "namespace",
        "name": "crm",
        "description": "CRM actions.",
        "tools": [{
            "type": "function",
            "name": "get_customer",
            "description": "Find a customer.",
            "defer_loading": True,
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        }],
    }])
    namespace_chat, namespace_state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4",
        "input": [_tool_search_call(), namespace_result],
        "tools": [_tool_search_declaration()],
    }))
    assert namespace_chat["tools"][-1]["function"]["name"] == (
        "__codex_namespace_crm_get_customer"
    )
    assert namespace_chat["messages"][-1]["content"] == "Loaded tools: crm.get_customer"
    assert namespace_state.tool_registry["__codex_namespace_crm_get_customer"]["namespace"] == "crm"


def test_tool_search_arguments_must_be_an_object():
    chat, state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4", "input": "Find a tool.", "tools": [_tool_search_declaration()],
    }))
    try:
        PROXY.chat_result_to_response_object({
            "choices": [{"message": {"tool_calls": [{
                "id": "call_bad_search",
                "type": "function",
                "function": {
                    "name": "__codex_tool_search",
                    "arguments": '["not", "an", "object"]',
                },
            }]}}],
        }, chat, state)
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_tool_search_call"
    else:
        raise AssertionError("tool_search arguments must be an object")


@pytest.mark.parametrize("namespace", [None, "node_repl"])
def test_tool_search_custom_result_round_trip_and_state_restore(namespace, monkeypatch):
    custom = {
        "type": "custom", "name": "js", "description": "Execute JavaScript.",
        "defer_loading": True,
        "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
    }
    loaded_tool = (
        {"type": "namespace", "name": namespace, "tools": [custom]}
        if namespace else custom
    )
    request = {
        "model": "DSv4", "store": False,
        "input": [_tool_search_call(), _tool_search_output(tools=[loaded_tool])],
        "tools": [_tool_search_declaration()],
    }
    original = json.dumps(request, sort_keys=True)
    chat, state = run(PROXY.responses_request_to_chat_request(request))
    assert json.dumps(request, sort_keys=True) == original
    backend_name = "__codex_custom_node_repl_js" if namespace else "__codex_custom_js"
    assert chat["tools"][-1]["function"]["name"] == backend_name
    assert state.tool_registry[backend_name]["format"] == custom["format"]
    assert chat["messages"][-1]["content"] == (
        "Loaded tools: node_repl.js" if namespace else "Loaded tools: js"
    )

    response = PROXY.chat_result_to_response_object({
        "choices": [{"message": {"tool_calls": [{
            "id": "call_js", "type": "function",
            "function": {"name": backend_name, "arguments": '{"input":"1 + 1"}'},
        }]}}],
    }, chat, state)
    item = response["output"][0]
    assert item["type"] == "custom_tool_call"
    assert item["name"] == "js"
    assert item["input"] == "1 + 1"
    assert item.get("namespace") == namespace
    stored = PROXY._state_payload(response, chat, state)

    class MemoryStore:
        enabled = True

        async def get(self, response_id):
            assert response_id == state.response_id
            return stored

    monkeypatch.setattr(PROXY, "response_state_store", MemoryStore())
    next_chat, next_state = run(PROXY.responses_request_to_chat_request({
        "model": "DSv4", "store": False, "previous_response_id": state.response_id,
        "input": [{"type": "custom_tool_call_output", "call_id": "call_js", "output": "2"}],
    }))
    assert next_chat["tools"] == chat["tools"]
    assert next_state.tool_registry[backend_name] == state.tool_registry[backend_name]
    assert next_chat["messages"][-2]["tool_calls"][0]["function"] == {
        "name": backend_name, "arguments": '{"input": "1 + 1"}',
    }
    assert next_chat["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_js", "content": "2",
    }


@pytest.mark.parametrize("status", [None, "missing"])
@pytest.mark.parametrize("stored_history", [False, True])
def test_tool_search_call_optional_status(status, stored_history, monkeypatch):
    call = _tool_search_call()
    if status == "missing":
        call.pop("status")
    else:
        call["status"] = status
    original = json.dumps(call, sort_keys=True)
    request = {
        "model": "DSv4", "store": False, "tools": [_tool_search_declaration()],
        "input": [call, _tool_search_output()],
    }
    if stored_history:
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4", "store": False, "input": "Find tools.",
            "tools": [_tool_search_declaration()],
        }))
        stored = PROXY._state_payload({"output": [call]}, chat, state)

        class MemoryStore:
            enabled = True

            async def get(self, response_id):
                assert response_id == state.response_id
                return stored

        monkeypatch.setattr(PROXY, "response_state_store", MemoryStore())
        request["previous_response_id"] = state.response_id
        request["input"] = [_tool_search_output()]

    chat, _ = run(PROXY.responses_request_to_chat_request(request))
    assert chat["messages"][-2]["tool_calls"][0]["id"] == "call_search"
    assert chat["messages"][-1]["content"] == "Loaded tools: get_shipping_eta"
    assert json.dumps(call, sort_keys=True) == original


@pytest.mark.parametrize("status", ["in_progress", "failed", "", False, [], {}])
def test_tool_search_call_rejects_explicit_invalid_status(status):
    call = {**_tool_search_call(), "status": status}
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        run(PROXY.responses_request_to_chat_request({
            "model": "DSv4", "tools": [_tool_search_declaration()],
            "input": [call, _tool_search_output()],
        }))
    assert error.value.code == "invalid_tool_search_call"
    assert error.value.status_code == 400


def test_tool_search_previous_response_restores_loaded_tools_and_accepts_output():
    previous_search_state = {
        "id": "resp_search",
        "model": "DSv4",
        "output": [_tool_search_call()],
        "_normalized_messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_search",
                "type": "function",
                "function": {
                    "name": "__codex_tool_search",
                    "arguments": '{"query":"shipping ETA","limit":5}',
                },
            }],
        }],
        "_tool_registry": {
            "__codex_tool_search": {
                "source_type": "tool_search",
                "source_name": "tool_search",
                "backend_name": "__codex_tool_search",
                "description": "Find relevant project tools.",
                "parameters": _tool_search_declaration()["parameters"],
                "execution": "client",
            },
        },
    }

    class MemoryStateStore:
        enabled = True

        async def get(self, response_id):
            assert response_id == "resp_search"
            return previous_search_state

    previous_store = PROXY.response_state_store
    try:
        PROXY.response_state_store = MemoryStateStore()
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "previous_response_id": "resp_search",
            "input": [_tool_search_output(), {"role": "user", "content": "Use the ETA tool."}],
        }))
    finally:
        PROXY.response_state_store = previous_store

    assert [tool["function"]["name"] for tool in chat["tools"]] == [
        "__codex_tool_search", "get_shipping_eta"
    ]
    assert chat["messages"][-2]["tool_call_id"] == "call_search"
    assert "Loaded tools: get_shipping_eta" in chat["messages"][-2]["content"]
    assert "get_shipping_eta" in state.tool_registry


def test_tool_search_loaded_tool_is_callable_on_the_next_previous_response_turn():
    class SearchStateStore:
        enabled = True

        async def get(self, response_id):
            assert response_id == "resp_search"
            return {
                "id": "resp_search",
                "model": "DSv4",
                "output": [_tool_search_call()],
                "_normalized_messages": [{
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "__codex_tool_search",
                            "arguments": '{"query":"shipping ETA"}',
                        },
                    }],
                }],
                "_tool_registry": {
                    "__codex_tool_search": {
                        "source_type": "tool_search",
                        "source_name": "tool_search",
                        "backend_name": "__codex_tool_search",
                        "parameters": _tool_search_declaration()["parameters"],
                    },
                },
            }

    store = SearchStateStore()
    previous_store = PROXY.response_state_store
    try:
        PROXY.response_state_store = store
        _, search_state = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "previous_response_id": "resp_search",
            "input": [_tool_search_output()],
        }))
        search_response = {
            "id": search_state.response_id,
            "output": [{
                "type": "function_call",
                "id": "fc_eta",
                "call_id": "call_eta",
                "name": "get_shipping_eta",
                "arguments": '{"order_id":"order_42"}',
                "status": "completed",
            }],
        }
        stored = PROXY._state_payload(search_response, {
            "messages": [
                {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_search", "type": "function",
                    "function": {"name": "__codex_tool_search", "arguments": '{"query":"shipping ETA"}'},
                }]},
                {"role": "tool", "tool_call_id": "call_search", "content": "Loaded tools: get_shipping_eta"},
            ]
        }, search_state)

        class LoadedToolStateStore:
            enabled = True

            async def get(self, response_id):
                assert response_id == search_state.response_id
                return stored

        PROXY.response_state_store = LoadedToolStateStore()
        next_chat, _ = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "previous_response_id": search_state.response_id,
            "input": [{
                "type": "function_call_output",
                "call_id": "call_eta",
                "output": "ETA: tomorrow",
            }],
        }))
    finally:
        PROXY.response_state_store = previous_store

    assert "get_shipping_eta" in [tool["function"]["name"] for tool in next_chat["tools"]]
    assert next_chat["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_eta", "content": "ETA: tomorrow"
    }


def test_tool_search_rejects_unmatched_missing_duplicate_and_unsupported_results():
    cases = [
        ([{**_tool_search_output(), "call_id": "unknown"}], "tool_search_call_not_found"),
        ([_tool_search_call()], "missing_tool_search_output"),
        ([_tool_search_output(), _tool_search_call()], "invalid_tool_search_output"),
        ([_tool_search_call(), _tool_search_output(), _tool_search_output()], "invalid_tool_search_output"),
        ([_tool_search_call(), _tool_search_output(tools=[{"type": "mcp", "server_label": "x"}])], "unsupported_tool_type"),
        ([_tool_search_call(), _tool_search_output(tools=[{"type": "function", "name": "bad", "parameters": []}])], "invalid_tool"),
    ]
    for input_items, expected_code in cases:
        try:
            run(PROXY.responses_request_to_chat_request({
                "model": "DSv4",
                "input": input_items,
                "tools": [_tool_search_declaration()],
            }))
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == expected_code
            assert exc.status_code == 400
        else:
            raise AssertionError(f"Expected {expected_code}")


def test_tool_search_rejects_conflicting_loaded_schema():
    conflicting_output = _tool_search_output(tools=[{
        "type": "function",
        "name": "lookup",
        "description": "Different schema.",
        "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}},
    }])
    try:
        run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": [_tool_search_call(), conflicting_output],
            "tools": [
                _tool_search_declaration(),
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Original schema.",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            ],
        }))
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "tool_schema_conflict"
    else:
        raise AssertionError("Conflicting loaded schema must fail closed")


def test_compaction_trigger_is_rejected_before_message_conversion():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [
                        {"role": "developer", "content": "Keep the facts."},
                        {"type": "compaction_trigger"},
                        {"role": "user", "content": "Continue."},
                    ],
                }
            )
        )
    except PROXY.UnsupportedResponsesCompactionError as exc:
        assert exc.code == "unsupported_compaction"
        assert exc.status_code == 400
        assert exc.input_item_index == 1
        assert exc.item_type == "compaction_trigger"
        assert str(exc) == "Remote compaction is not supported by this proxy."
    else:
        raise AssertionError("Remote compaction must fail closed")


def test_any_compaction_trigger_shape_is_rejected_without_leaking_fields():
    for item in [
        {"type": "compaction_trigger", "data": "unexpected"},
        {"type": "compaction_trigger", "encrypted_content": "secret"},
    ]:
        try:
            run(PROXY.responses_request_to_chat_request({"model": "DSv4", "input": [item]}))
        except PROXY.UnsupportedResponsesCompactionError as exc:
            assert exc.code == "unsupported_compaction"
            assert exc.item_type == "compaction_trigger"
            assert "secret" not in str(exc.diagnostic_payload())
        else:
            raise AssertionError("Every compaction trigger shape must fail closed")


def test_multiple_compaction_triggers_fail_on_first_trigger():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [
                        {"role": "user", "content": "before"},
                        {"type": "compaction_trigger"},
                        {"type": "compaction_trigger"},
                    ],
                }
            )
        )
    except PROXY.UnsupportedResponsesCompactionError as exc:
        assert exc.input_item_index == 1
    else:
        raise AssertionError("Repeated compaction triggers must fail closed")


def test_compaction_trigger_text_is_not_treated_as_control_item():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {"model": "DSv4", "input": "The word compaction_trigger is documentation."}
        )
    )
    assert chat["messages"][-1]["content"] == "The word compaction_trigger is documentation."


def test_compaction_trigger_with_unexpected_fields_remains_rejected():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{"type": "compaction_trigger", "data": "unexpected"}],
                }
            )
        )
    except PROXY.UnsupportedResponsesInputItemError as exc:
        assert exc.item_type == "compaction_trigger"
        assert exc.item_keys == ["data", "type"]
    else:
        raise AssertionError("Extended compaction trigger must be rejected")


def test_codex_additional_tools_are_sent_to_chat_backend():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "tool_choice": "auto",
                "input": [
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "functions",
                                "tools": [
                                    {
                                        "type": "function",
                                        "name": "exec_command",
                                        "description": "Run a command.",
                                        "parameters": {"type": "object"},
                                    }
                                ],
                            }
                        ],
                    },
                    {"role": "user", "content": "Inspect the workspace."},
                ],
            }
        )
    )
    assert [message["role"] for message in chat["messages"]] == ["system", "user"]
    assert chat["messages"][0]["tools"][0]["function"]["name"] == (
        "__codex_namespace_functions_exec_command"
    )
    assert chat["tools"] == chat["messages"][0]["tools"]
    assert chat["tool_choice"] == "auto"
    assert "__codex_namespace_functions_exec_command" in state.tool_registry


def test_codex_additional_tools_requires_tools_array():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{"type": "additional_tools", "role": "developer"}],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_tools"
    else:
        raise AssertionError("Expected invalid additional_tools to fail")


def test_codex_assistant_output_text_is_accepted_on_next_turn():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "你好啊！我是 Codex。"},
                        ],
                    },
                    {"role": "user", "content": "你是什么模型"},
                ],
            }
        )
    )
    assert chat["messages"] == [
        {"role": "assistant", "content": [{"type": "text", "text": "你好啊！我是 Codex。"}]},
        {"role": "user", "content": "你是什么模型"},
    ]


def test_codex_agent_message_is_preserved_as_assistant_history():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "type": "agent_message",
                        "id": "msg_agent_1",
                        "author": "agent",
                        "recipient": "user",
                        "content": [
                            {"type": "output_text", "text": "Completed the previous step."},
                        ],
                    },
                    {"role": "user", "content": "Continue."},
                ],
            }
        )
    )

    assert chat["messages"] == [
        {"role": "assistant", "content": [{"type": "text", "text": "Completed the previous step."}]},
        {"role": "user", "content": "Continue."},
    ]


def test_assistant_history_ignores_encrypted_content_part():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Completed the previous step."},
                            {"type": "encrypted_content", "encrypted_content": "secret"},
                        ],
                    },
                    {"role": "user", "content": "Continue."},
                ],
            }
        )
    )

    assert chat["messages"] == [
        {"role": "assistant", "content": [{"type": "text", "text": "Completed the previous step."}]},
        {"role": "user", "content": "Continue."},
    ]


def test_agent_message_ignores_encrypted_content_part():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "type": "agent_message",
                        "id": "msg_agent_1",
                        "author": "agent",
                        "recipient": "user",
                        "content": [
                            {"type": "output_text", "text": "Completed the previous step."},
                            {"type": "encrypted_content", "encrypted_content": "secret"},
                        ],
                    },
                ],
            }
        )
    )

    assert chat["messages"] == [
        {"role": "assistant", "content": [{"type": "text", "text": "Completed the previous step."}]},
    ]


def test_user_encrypted_content_part_remains_rejected():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{
                        "role": "user",
                        "content": [{"type": "encrypted_content", "encrypted_content": "secret"}],
                    }],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "unsupported_input_type"
    else:
        raise AssertionError("user encrypted_content must not be silently dropped")


def test_output_text_remains_invalid_for_user_messages():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{
                        "role": "user",
                        "content": [{"type": "output_text", "text": "not user input"}],
                    }],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "unsupported_input_type"
    else:
        raise AssertionError("output_text must not be accepted as user input")


def test_unknown_roleless_input_item_has_safe_failure_diagnostic(tmp_path):
    async def check_handler():
        body = json.dumps({
            "model": "DSv4",
            "input": [{
                "type": "item_reference",
                "id": "item_123",
                "content": "must-not-appear",
                "arguments": "also-must-not-appear",
                "encrypted_content": "secret",
            }],
        }).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        request = PROXY.Request({
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8077),
        }, receive)
        response = await PROXY.handle_responses(request)

        assert response.status_code == 400
        assert response.headers["x-proxy-diagnostic-id"].startswith("rejected_")
        error = json.loads(response.body)["error"]
        assert error["code"] == "unsupported_input_role"
        assert "item_reference" in error["message"]
        diagnostic_path = (
            tmp_path / response.headers["x-proxy-diagnostic-id"] / "input.rejected.json"
        )
        diagnostic_text = diagnostic_path.read_text()
        diagnostic = json.loads(diagnostic_text)
        assert diagnostic["input_item_index"] == 0
        assert diagnostic["item_type"] == "item_reference"
        assert diagnostic["has_encrypted_content"] is True
        assert "must-not-appear" not in diagnostic_text
        assert "also-must-not-appear" not in diagnostic_text
        assert '"secret"' not in diagnostic_text

    previous_enabled = PROXY.RESPONSES_WIRE_TRACE
    previous_dir = PROXY.RESPONSES_WIRE_TRACE_DIR
    try:
        PROXY.RESPONSES_WIRE_TRACE = True
        PROXY.RESPONSES_WIRE_TRACE_DIR = str(tmp_path)
        run(check_handler())
    finally:
        PROXY.RESPONSES_WIRE_TRACE = previous_enabled
        PROXY.RESPONSES_WIRE_TRACE_DIR = previous_dir


def test_handle_responses_rejects_remote_compaction_before_backend_dispatch():
    async def check_handler():
        backend_called = False

        async def backend(*_args):
            nonlocal backend_called
            backend_called = True
            raise AssertionError("remote compaction must not reach P/D")

        body = json.dumps({
            "model": "DSv4",
            "stream": True,
            "input": [
                {"role": "user", "content": "before"},
                {"type": "compaction_trigger", "extra": "ignored safely"},
            ],
        }).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        request = PROXY.Request({
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8077),
        }, receive)
        response = await PROXY.handle_responses(request, backend_runner=backend)

        assert response.status_code == 400
        error = json.loads(response.body)["error"]
        assert error == {
            "type": "invalid_request_error",
            "code": "unsupported_compaction",
            "message": "Remote compaction is not supported by this proxy.",
        }
        assert backend_called is False

    run(check_handler())


def test_function_call_output():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {"type": "function_call_output", "call_id": "call_1", "output": "22C"},
                ],
            }
        )
    )
    assert chat["messages"][0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "22C",
    }


def test_function_call_input_and_tool_choice():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Beijing"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "Sunny",
                    },
                ],
                "tools": [{
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object"},
                }],
                "tool_choice": {"type": "function", "name": "get_weather"},
            }
        )
    )
    assistant = next(message for message in chat["messages"] if message["role"] == "assistant")
    tool = next(message for message in chat["messages"] if message["role"] == "tool")
    assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
    assert tool["tool_call_id"] == "call_1"
    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_invalid_function_arguments_are_rejected_before_backend():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "exec_command",
                        "arguments": '{"message": "',
                    }],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_function_arguments"
    else:
        raise AssertionError("Truncated function arguments must be rejected")


def test_prepare_backend_request_rejects_invalid_tool_arguments():
    request = {
        "model": "DSv4",
        "messages": [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '{"cmd": '},
            }],
        }],
    }
    try:
        PROXY.prepare_backend_request_data(request)
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_function_arguments"
    else:
        raise AssertionError("Invalid tool arguments must not reach the backend")


def test_tool_choice_without_tools_is_not_forwarded_to_backend():
    chat, _ = run(
        PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": "hello",
            "tool_choice": "auto",
        })
    )
    assert "tool_choice" not in chat
    assert "tools" not in chat


def test_explicit_tool_choice_without_tools_is_rejected_locally():
    try:
        run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": "hello",
            "tool_choice": "required",
        }))
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "tool_choice_requires_tools"
    else:
        raise AssertionError("Expected tool_choice without tools to fail")


def test_function_tool_openai_compatibility_and_schema_validation():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": "Call weather",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather.",
                            "parameters": {"type": "object"},
                        },
                        "strict": True,
                    }
                ],
            }
        )
    )
    assert chat["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather.",
                "parameters": {"type": "object"},
                "strict": True,
            },
        }
    ]

    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": "Call weather",
                    "tools": [
                        {
                            "type": "function",
                            "name": "bad",
                            "parameters": [],
                        }
                    ],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_tool"
    else:
        raise AssertionError("Expected invalid function schema to fail")


def test_custom_and_web_search_tools_are_normalized():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": "Inspect the repository and search the web.",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply a patch.",
                        "format": {"type": "grammar", "syntax": "lark"},
                    },
                    {"type": "web_search_preview"},
                ],
                "tool_choice": {"type": "custom", "name": "apply_patch"},
            }
        )
    )
    assert [tool["function"]["name"] for tool in chat["tools"]] == [
        "__codex_custom_apply_patch",
        "__codex_web_search_web_search",
    ]
    assert chat["tools"][0]["function"]["parameters"]["required"] == ["input"]
    assert chat["tool_choice"] == {
        "type": "function",
        "function": {"name": "__codex_custom_apply_patch"},
    }
    assert state.tool_registry["__codex_custom_apply_patch"]["execution"] == "client"
    assert state.tool_registry["__codex_web_search_web_search"]["execution"] == "proxy"


def test_namespace_tools_are_flattened_for_pd_and_restored():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": "Create a file.",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "description": "Local tools.",
                    "tools": [{
                        "type": "function",
                        "name": "exec_command",
                        "description": "Run a command.",
                        "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                    }],
                }],
            }
        )
    )
    assert chat["tools"][0]["function"]["name"] == "__codex_namespace_functions_exec_command"
    assert state.tool_registry["__codex_namespace_functions_exec_command"]["namespace"] == "functions"
    assert chat.get("tool_choice") is None
    choice_chat, _ = run(
        PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": "Run it.",
            "tools": [{
                "type": "namespace", "name": "functions", "tools": [{
                    "type": "function", "name": "exec_command", "parameters": {"type": "object"}
                }]
            }],
            "tool_choice": {"type": "namespace", "name": "functions.exec_command"},
        })
    )
    assert choice_chat["tool_choice"]["function"]["name"] == "__codex_namespace_functions_exec_command"
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "__codex_namespace_functions_exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
        }]}}]},
        chat,
        state,
    )
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][0]["name"] == "exec_command"


def test_custom_tool_output_is_restored_to_chat_tool_message():
    chat, _ = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": [
                    {
                        "type": "custom_tool_call",
                        "id": "ctc_1",
                        "call_id": "call_1",
                        "name": "apply_patch",
                        "input": "*** Begin Patch",
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_1",
                        "output": "patched",
                    },
                ],
            }
        )
    )
    assert chat["messages"][0]["tool_calls"][0]["function"]["name"] == "apply_patch"
    assert chat["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"input": "*** Begin Patch"}'
    assert chat["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "patched",
    }


def test_custom_tool_response_is_formatted_with_original_name():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {
                "model": "DSv4",
                "input": "Apply it.",
                "tools": [{"type": "custom", "name": "apply_patch"}],
            }
        )
    )
    response = PROXY.chat_result_to_response_object(
        {
            "choices": [{"message": {"tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "__codex_custom_apply_patch",
                    "arguments": '{"input":"*** Begin Patch"}',
                },
            }]}}],
        },
        chat,
        state,
    )
    assert response["output"] == [{
        "id": response["output"][0]["id"],
        "type": "custom_tool_call",
        "call_id": "call_1",
        "name": "apply_patch",
        "input": "*** Begin Patch",
        "status": "completed",
    }]


def _codex_functions_exec_request():
    return {
        "model": "DSv4",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{
                        "type": "custom",
                        "name": "exec",
                        "description": "Run JavaScript with nested Codex tools. exec_command(args: { cmd: string }): Promise<unknown>;",
                    }],
                }],
            },
            {"role": "user", "content": "Edit the workspace."},
        ],
    }


def _codex_functions_wait_request():
    return {
        "model": "DSv4",
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "namespace",
                    "name": "functions",
                    "tools": [{
                        "type": "function",
                        "name": "wait",
                        "description": "Wait for a yielded exec cell.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "cell_id": {"type": "string"},
                                "yield_time_ms": {"type": "number"},
                                "max_tokens": {"type": "number"},
                                "terminate": {"type": "boolean"},
                            },
                            "required": ["cell_id"],
                            "additionalProperties": False,
                        },
                    }],
                }],
            },
            {"role": "user", "content": "Wait for the command."},
        ],
    }


def test_functions_wait_alias_maps_to_declared_wait_tool():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_wait_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_wait_alias",
            "type": "function",
            "function": {
                "name": "__codex_functions_wait",
                "arguments": '{"cell_id":"x","yield_time_ms":1000,"max_tokens":1000}',
            },
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "function_call"
    assert call["name"] == "wait"
    assert call["namespace"] == "functions"
    assert json.loads(call["arguments"]) == {
        "cell_id": "x",
        "yield_time_ms": 1000,
        "max_tokens": 1000,
    }


def test_functions_wait_alias_requires_declared_wait_and_valid_arguments():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for arguments, code in (
        ('{"cell_id":123}', "undeclared_tool"),
        ('{"cell_id":"x","unknown":true}', "undeclared_tool"),
    ):
        try:
            PROXY.chat_result_to_response_object(
                {"choices": [{"message": {"tool_calls": [{
                    "id": "call_wait_rejected",
                    "type": "function",
                    "function": {
                        "name": "__codex_functions_wait",
                        "arguments": arguments,
                    },
                }]}}]},
                chat,
                state,
            )
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == code
        else:
            raise AssertionError("Undeclared wait alias must be rejected")

    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_wait_request()))
    for arguments in (
        '{"cell_id":123}',
        '{"cell_id":"x","unknown":true}',
        '{"cell_id":"x","terminate":"false"}',
    ):
        try:
            PROXY.chat_result_to_response_object(
                {"choices": [{"message": {"tool_calls": [{
                    "id": "call_wait_invalid",
                    "type": "function",
                    "function": {
                        "name": "__codex_functions_wait",
                        "arguments": arguments,
                    },
                }]}}]},
                chat,
                state,
            )
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == "invalid_function_arguments"
        else:
            raise AssertionError("Malformed wait alias must be rejected")


def test_bare_apply_patch_is_wrapped_through_declared_functions_exec():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_patch",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "*** Begin Patch\n*** End Patch"},
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["call_id"] == "call_patch"
    assert call["input"].startswith("await tools.apply_patch(")
    restored = PROXY._response_output_to_assistant_message(response, state.tool_registry)
    function = restored["tool_calls"][0]["function"]
    assert function["name"] == "__codex_custom_functions_exec"
    assert json.loads(function["arguments"])["input"] == call["input"]


def test_bare_exec_is_restored_to_declared_functions_exec():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_exec",
            "type": "function",
            "function": {"name": "exec", "arguments": "await tools.apply_patch('*** Begin Patch');"},
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["input"] == "await tools.apply_patch('*** Begin Patch');"


def test_bare_exec_command_is_wrapped_through_declared_functions_exec():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_exec",
            "type": "function",
            "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'},
        }]}}]},
        chat,
        state,
    )
    assert response["output"][0]["name"] == "exec"
    assert response["output"][0]["input"] == 'text(await tools.exec_command({"cmd": "pwd"}));'


def test_bare_codex_builtin_tools_are_wrapped_through_declared_functions_exec():
    cases = {
        "write_stdin": {"session_id": 1, "chars": "ok"},
        "update_plan": {"plan": []},
        "update_goal": {"status": "complete"},
        "create_goal": {"objective": "audit"},
        "get_goal": {},
        "view_image": {"path": "/tmp/image.png"},
    }
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for tool_name, arguments in cases.items():
        response = PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }]}}]},
            chat,
            state,
        )
        call = response["output"][0]
        assert call["type"] == "custom_tool_call"
        assert call["name"] == "exec"
        assert call["namespace"] == "functions"
        assert call["input"] == (
            f"await tools.{tool_name}("
            + json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
            + ");"
        )


def test_undeclared_decoder_tool_is_rejected():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    try:
        PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": "call_unknown",
                "type": "function",
                "function": {"name": "__codec_unknown_functions_exec", "arguments": "{}"},
            }]}}]},
            chat,
            state,
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "undeclared_tool"
    else:
        raise AssertionError("Undeclared Decoder tool must be rejected")


def test_codec_functions_exec_alias_maps_to_declared_custom_tool():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_alias",
            "type": "function",
            "function": {
                "name": "__codec_custom_functions_exec",
                "arguments": '{"input":"await tools.update_plan({\\"plan\\":[]});"}',
            },
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["input"] == 'await tools.update_plan({"plan":[]});'


def test_codec_functions_exec_alias_requires_valid_custom_input():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    try:
        PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": "call_alias",
                "type": "function",
                "function": {"name": "__codec_custom_functions_exec", "arguments": "{}"},
            }]}}]},
            chat,
            state,
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_function_arguments"
    else:
        raise AssertionError("Alias without functions.exec input must be rejected")


def test_codec_functions_exec_short_alias_maps_to_declared_custom_tool():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for index, tool_name in enumerate((
        "__codec_functions_exec",
        "codex_exec",
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
    )):
        response = PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": f"call_alias_{index}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": '{"input":"await tools.get_goal({});"}',
                },
            }]}}]},
            chat,
            state,
        )
        assert response["output"][0]["input"] == "await tools.get_goal({});"


def test_new_functions_exec_aliases_keep_strict_input_validation():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for index, tool_name in enumerate((
        "__codex_volcengine_functions_exec",
        "codex custom functions exec",
        "__codex_functions.exec",
        "__codex_mem_functions_exec",
    )):
        try:
            PROXY.chat_result_to_response_object(
                {"choices": [{"message": {"tool_calls": [{
                    "id": f"call_bad_alias_{index}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": '{"x":"1"}'},
                }]}}]},
                chat,
                state,
            )
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == "invalid_function_arguments"
        else:
            raise AssertionError(f"{tool_name} must reject arguments without input")


def test_exact_exec_command_aliases_are_wrapped_through_declared_functions_exec():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for tool_name in (
        "__codec_exec_command",
        "__codex_custom_exec_command",
        "codex_exec_command",
        "__codex_exec_command",
    ):
        response = PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": f"call_{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": '{"cmd":"pwd"}'},
            }]}}]},
            chat,
            state,
        )
        call = response["output"][0]
        assert call["name"] == "exec"
        assert call["namespace"] == "functions"
        assert call["input"] == 'text(await tools.exec_command({"cmd": "pwd"}));'


def test_unrelated_exec_like_names_remain_rejected():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    for index, tool_name in enumerate((
        "__exec_dummy",
        "exec_query",
        "__codex_functions_exec_placeholder",
    )):
        try:
            PROXY.chat_result_to_response_object(
                {"choices": [{"message": {"tool_calls": [{
                    "id": f"call_rejected_{index}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": "{}"},
                }]}}]},
                chat,
                state,
            )
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == "undeclared_tool"
        else:
            raise AssertionError(f"Unexpectedly accepted {tool_name}")


@pytest.mark.parametrize("alias", ["__codex_custom_exec_command", "codex_exec_command"])
def test_custom_exec_command_input_alias_maps_only_valid_functions_exec_input(alias):
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_alias",
            "type": "function",
            "function": {
                "name": alias,
                "arguments": '{"input":"await tools.get_goal({});"}',
            },
        }]}}]},
        chat,
        state,
    )
    assert response["output"][0]["input"] == "await tools.get_goal({});"

    for arguments in (
        '{"input":"<tool_call>await tools.get_goal({});</tool_call>"}',
        '{"cmd":"pwd","extra":true}',
        '{"input":""}',
        '{"input":false}',
        '{"input":"text(1);","cmd":"pwd"}',
    ):
        try:
            PROXY.chat_result_to_response_object(
                {"choices": [{"message": {"tool_calls": [{
                    "id": "call_rejected",
                    "type": "function",
                    "function": {"name": alias, "arguments": arguments},
                }]}}]},
                chat,
                state,
            )
        except PROXY.ResponsesRequestError as exc:
            assert exc.code == "invalid_function_arguments"
        else:
            raise AssertionError("Malformed exact exec-command alias must be rejected")


def test_namespace_update_goal_alias_is_wrapped_through_declared_functions_exec():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_goal",
            "type": "function",
            "function": {
                "name": "__codex_namespace_functions_update_goal",
                "arguments": '{"status":"blocked"}',
            },
        }]}}]},
        chat,
        state,
    )
    assert response["output"][0]["input"] == 'await tools.update_goal({"status":"blocked"});'


def test_decoder_codex_custom_containment_maps_unique_declared_tool():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_contained",
            "type": "function",
            "function": {
                "name": "decoder_prefix___codex_custom_functions_exec__suffix",
                "arguments": '{"input":"await tools.get_goal({});"}',
            },
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["input"] == "await tools.get_goal({});"


def test_ambiguous_custom_functions_containment_is_passed_through():
    state = PROXY.ResponseRequestState(
        "resp_ambiguous", "DSv4", 1, False,
        tool_registry={
            "__codex_custom_functions_a": {"source_type": "function", "source_name": "a"},
            "__codex_custom_functions_b": {"source_type": "function", "source_name": "b"},
        },
    )
    spec, arguments, nested_wrapped = PROXY._normalize_decoder_tool_call(
        "__codex_custom_functions_a__codex_custom_functions_b", "{}", "call_ambiguous", state,
    )
    assert spec["source_type"] == "function"
    assert spec["source_name"] == "__codex_custom_functions_a__codex_custom_functions_b"
    assert arguments == "{}"
    assert nested_wrapped is False


def test_custom_functions_exec_suffix_maps_unique_declared_tool():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    response = PROXY.chat_result_to_response_object(
        {"choices": [{"message": {"tool_calls": [{
            "id": "call_passthrough",
            "type": "function",
            "function": {
                "name": "__codecode_custom_functions_exec",
                "arguments": '{"input":"text(1);"}',
            },
        }]}}]},
        chat,
        state,
    )
    call = response["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["input"] == "text(1);"


def test_custom_functions_exec_suffix_requires_unique_declared_tool():
    state = PROXY.ResponseRequestState(
        "resp_suffix_ambiguous", "DSv4", 1, False,
        tool_registry={
            "__codex_custom_functions_exec": {
                "source_type": "custom", "source_name": "exec",
            },
            "__other_custom_functions_exec": {
                "source_type": "function", "source_name": "other_exec",
            },
        },
    )
    # An ambiguous exec suffix may no longer bypass declared-tool validation.
    with pytest.raises(PROXY.ResponsesRequestError) as error:
        PROXY._normalize_decoder_tool_call(
            "__codecode_custom_functions_exec", "{}", "call_suffix_ambiguous", state,
        )
    assert error.value.code == "undeclared_tool"


def test_bare_exec_command_requires_object_arguments():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    try:
        PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "id": "call_exec",
                "type": "function",
                "function": {"name": "exec_command", "arguments": '"pwd"'},
            }]}}]},
            chat,
            state,
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "invalid_function_arguments"
    else:
        raise AssertionError("Expected scalar exec_command arguments to fail")


def test_bare_apply_patch_requires_decoder_call_id():
    chat, state = run(PROXY.responses_request_to_chat_request(_codex_functions_exec_request()))
    try:
        PROXY.chat_result_to_response_object(
            {"choices": [{"message": {"tool_calls": [{
                "type": "function",
                "function": {"name": "apply_patch", "arguments": "*** Begin Patch"},
            }]}}]},
            chat,
            state,
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "incomplete_function_call"
    else:
        raise AssertionError("Expected missing nested-tool call id to fail")


def test_external_url_validation_blocks_private_hosts():
    for url in ("http://127.0.0.1/x", "http://localhost/x", "file:///etc/passwd"):
        try:
            PROXY._validate_external_url(url)
        except PROXY.WebToolError:
            pass
        else:
            raise AssertionError(f"Expected URL to be blocked: {url}")


def test_web_search_has_configurable_provider():
    old_url = PROXY.CODEX_WEB_SEARCH_URL
    PROXY.CODEX_WEB_SEARCH_URL = None
    try:
        # The implementation has a controlled default provider; network tests
        # are kept out of the unit suite, but the configuration path is exact.
        assert PROXY.CODEX_WEB_SEARCH_URL is None
    finally:
        PROXY.CODEX_WEB_SEARCH_URL = old_url


def test_proxy_client_does_not_change_pd_client_configuration():
    old_http = PROXY.CODEX_HTTP_PROXY
    old_https = PROXY.CODEX_HTTPS_PROXY
    PROXY.CODEX_HTTP_PROXY = "http://10.43.208.68:33000"
    PROXY.CODEX_HTTPS_PROXY = PROXY.CODEX_HTTP_PROXY
    try:
        client = PROXY._external_http_client()
        assert client._trust_env is False
        assert client._mounts
        run(client.aclose())
    finally:
        PROXY.CODEX_HTTP_PROXY = old_http
        PROXY.CODEX_HTTPS_PROXY = old_https


def test_web_tool_arguments_accept_direct_page_url(monkeypatch=None):
    original = PROXY.fetch_external_url
    async def fake_fetch(url):
        assert url == "https://example.com"
        return {"url": url, "status_code": 200, "content_type": "text/html", "body": "Example Domain"}
    PROXY.fetch_external_url = fake_fetch
    try:
        result = run(PROXY.execute_web_search({"url": "https://example.com"}))
        assert result["type"] == "open_page"
        assert result["page"]["body"] == "Example Domain"
    finally:
        PROXY.fetch_external_url = original


def test_function_call_is_restored_for_previous_response():
    response = {
        "id": "resp_tool",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"Beijing"}',
                "status": "completed",
            }
        ],
    }
    payload = PROXY._state_payload(
        response,
        {"messages": [{"role": "user", "content": "Call weather"}]},
    )
    assert payload["_normalized_messages"][-1] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Beijing"}',
                },
            }
        ],
    }


def test_responses_store_defaults_to_enabled():
    previous = os.environ.get("RESPONSES_DEFAULT_STORE")
    try:
        os.environ.pop("RESPONSES_DEFAULT_STORE", None)
        assert PROXY._responses_store_default() is True
        os.environ["RESPONSES_DEFAULT_STORE"] = "false"
        assert PROXY._responses_store_default() is False
    finally:
        if previous is None:
            os.environ.pop("RESPONSES_DEFAULT_STORE", None)
        else:
            os.environ["RESPONSES_DEFAULT_STORE"] = previous


def test_previous_response_id_restores_stored_messages_and_default_store():
    class MemoryStateStore:
        enabled = True

        async def get(self, response_id):
            return {
                "id": response_id,
                "model": "DSv4",
                "_normalized_messages": [
                    {"role": "user", "content": "Remember 42."},
                    {"role": "assistant", "content": "I will remember 42."},
                ],
                "_tool_registry": {},
            }

    previous_store = PROXY.response_state_store
    previous_default = os.environ.get("RESPONSES_DEFAULT_STORE")
    try:
        PROXY.response_state_store = MemoryStateStore()
        os.environ["RESPONSES_DEFAULT_STORE"] = "true"
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "previous_response_id": "resp_previous",
            "input": "What number did I ask you to remember?",
        }))
        assert state.store is True
        assert state.previous_response_id == "resp_previous"
        assert [message["role"] for message in chat["messages"]] == [
            "user", "assistant", "user"
        ]
        assert chat["messages"][-1]["content"] == "What number did I ask you to remember?"
    finally:
        PROXY.response_state_store = previous_store
        if previous_default is None:
            os.environ.pop("RESPONSES_DEFAULT_STORE", None)
        else:
            os.environ["RESPONSES_DEFAULT_STORE"] = previous_default


def test_conversation_restores_stored_messages_without_json_decoding_response_id():
    class MemoryStateStore:
        enabled = True

        async def get_conversation(self, conversation_id):
            assert conversation_id == "conv_saved"
            return "resp_previous"

        async def get(self, response_id):
            assert response_id == "resp_previous"
            return {
                "id": response_id,
                "model": "DSv4",
                "_normalized_messages": [
                    {"role": "user", "content": "Remember amber-71."},
                    {"role": "assistant", "content": "I will remember amber-71."},
                ],
                "_tool_registry": {},
            }

    previous_store = PROXY.response_state_store
    try:
        PROXY.response_state_store = MemoryStateStore()
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "conversation": "conv_saved",
            "input": "What did I ask you to remember?",
        }))
        assert state.previous_response_id == "resp_previous"
        assert [message["role"] for message in chat["messages"]] == ["user", "assistant", "user"]
        assert chat["messages"][-1]["content"] == "What did I ask you to remember?"
    finally:
        PROXY.response_state_store = previous_store


def test_response_state_store_reads_raw_conversation_response_id():
    store = PROXY.ResponseStateStore()
    previous_url = PROXY.RESPONSES_REDIS_URL

    async def command(*parts):
        assert parts == ("GET", f"{PROXY.RESPONSES_KEY_PREFIX}conversation:conv_raw")
        return "resp_raw_value"

    try:
        PROXY.RESPONSES_REDIS_URL = "redis://127.0.0.1:6379/0"
        store._command = command
        assert run(store.get_conversation("conv_raw")) == "resp_raw_value"
    finally:
        PROXY.RESPONSES_REDIS_URL = previous_url


def test_internal_conversation_json_error_is_not_reported_as_client_invalid_json():
    class BrokenStateStore:
        enabled = True

        async def get_conversation(self, _conversation_id):
            raise json.JSONDecodeError("corrupt state", "resp_raw_value", 0)

    async def check_handler():
        body = json.dumps({"model": "DSv4", "conversation": "conv_broken", "input": "hello"}).encode()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        request = PROXY.Request({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/v1/responses",
            "raw_path": b"/v1/responses", "query_string": b"", "headers": [],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8077),
        }, receive)
        response = await PROXY.handle_responses(request)
        assert response.status_code == 503
        assert json.loads(response.body)["error"]["code"] == "response_state_unavailable"

    previous_store = PROXY.response_state_store
    try:
        PROXY.response_state_store = BrokenStateStore()
        run(check_handler())
    finally:
        PROXY.response_state_store = previous_store


def test_retrieve_response_hides_proxy_state_and_reports_missing_response():
    class MemoryStateStore:
        enabled = True

        def __init__(self):
            self.responses = {
                "resp_saved": {
                    "id": "resp_saved",
                    "object": "response",
                    "status": "completed",
                    "output": [],
                    "_normalized_messages": [{"role": "user", "content": "private"}],
                    "_tool_registry": {"internal": {"source_name": "private"}},
                }
            }

        async def get(self, response_id):
            return self.responses.get(response_id)

    previous_store = PROXY.response_state_store
    try:
        PROXY.response_state_store = MemoryStateStore()
        response = run(PROXY.retrieve_response("resp_saved"))
        payload = json.loads(response.body)
        assert response.status_code == 200
        assert payload["id"] == "resp_saved"
        assert "_normalized_messages" not in payload
        assert "_tool_registry" not in payload

        missing = run(PROXY.retrieve_response("resp_missing"))
        assert missing.status_code == 404
        assert json.loads(missing.body)["error"]["code"] == "response_not_found"
    finally:
        PROXY.response_state_store = previous_store


def test_response_object_conversion():
    chat, state = run(
        PROXY.responses_request_to_chat_request(
            {"model": "DSv4", "input": "Hello"}
        )
    )
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    converted = PROXY.chat_result_to_response_object(response, chat, state)
    assert converted["object"] == "response"
    assert converted["output"][0]["type"] == "message"
    assert converted["output"][0]["content"][0]["text"] == "Hi"
    assert converted["usage"]["total_tokens"] == 7


def test_ds_v4_reasoning_is_response_output_item():
    response = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "17 x 19 = 323",
                "reasoning": "Compute 17 * 19.",
            },
            "finish_reason": "stop",
        }]
    }
    state = PROXY.ResponseRequestState("resp_reasoning", "DSv4", 1, False)
    converted = PROXY.chat_result_to_response_object(response, {"messages": []}, state)
    assert [item["type"] for item in converted["output"]] == ["reasoning", "message"]
    assert converted["output"][0]["summary"][0]["text"] == "Compute 17 * 19."


def test_ds_v4_reasoning_stream_is_before_text():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"reasoning":"Think first"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":"Answer"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_reasoning_stream", "DSv4", 1, False)
        result = []
        async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}
        ):
            result.append(chunk.decode())
        return result

    events = []
    for block in "\n".join(run(collect())).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    types = [event["type"] for event in events]
    assert types.index("response.output_item.added") < types.index(
        "response.reasoning_summary_part.added"
    )
    assert types.index("response.reasoning_summary_text.done") < types.index(
        "response.reasoning_summary_part.done"
    )
    assert types.index("response.reasoning_summary_part.done") < types.index(
        "response.output_item.done", types.index("response.reasoning_summary_part.done") + 1
    )
    assert "response.reasoning_summary_text.done" in types
    completed = events[-1]["response"]
    assert [item["type"] for item in completed["output"]] == ["reasoning", "message"]
    assert completed["output"][0]["summary"][0]["text"] == "Think first"


def test_internal_stream_collection_preserves_reasoning_for_tool_rounds():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"reasoning":"Plan tool call. "}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"web_search","arguments":"{\\"query\\":\\"DSv4\\"}"}}]}}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
        yield b"data: [DONE]\n\n"

    collected = run(PROXY._collect_chat_stream_response(chunks()))
    message = collected["choices"][0]["message"]
    assert message["reasoning_content"] == "Plan tool call. "
    assert message["tool_calls"][0]["function"]["name"] == "web_search"


def test_tool_round_assistant_message_preserves_ds_v4_reasoning():
    message = PROXY._assistant_message_for_tool_calls({
        "choices": [{
            "message": {
                "content": "",
                "reasoning": "Choose the search tool.",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }],
            }
        }]
    })
    assert message["reasoning_content"] == "Choose the search tool."


def test_codex_reasoning_function_call_and_output_round_is_normalized():
    chat, _ = run(
        PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Plan first."}],
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "function_call_output",
                    "id": "fco_1",
                    "call_id": "call_1",
                    "output": " /tmp",
                },
            ],
        })
    )
    assert chat["messages"][0] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "Plan first.",
    }
    assert chat["messages"][1]["role"] == "assistant"
    assert chat["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert chat["messages"][2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": " /tmp",
    }


def test_unsupported_multimodal_input():
    try:
        run(
            PROXY.responses_request_to_chat_request(
                {
                    "model": "DSv4",
                    "input": [{"role": "user", "content": [{"type": "input_image"}]}],
                }
            )
        )
    except PROXY.ResponsesRequestError as exc:
        assert exc.code == "unsupported_multimodal_content"
    else:
        raise AssertionError("Expected unsupported multimodal input to fail")


def test_native_multimodal_input_preserves_image_for_pd():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = type("Args", (), {
            "multimodal_backend_url": "http://vision/v1",
            "multimodal_backend_model": "Kimi-K2.6",
            "multimodal_max_image_bytes": 1024 * 1024,
        })()
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "gpt-5",
            "input": [{"role": "user", "content": [
                {"type": "input_text", "text": "Describe this."},
                {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto"},
            ]}],
        }))
        assert state.backend_route == "pd"
        assert state.contains_images
        assert state.backend_model is None
        assert chat["messages"][0]["content"][0] == {"type": "text", "text": "Describe this."}
        assert chat["messages"][0]["content"][1]["type"] == "image_url"
        assert chat["messages"][0]["content"][1]["image_url"]["detail"] == "auto"
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_configured_multimodal_image_in_instructions_is_preserved():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = type("Args", (), {
            "multimodal_backend_url": "http://vision/v1",
            "multimodal_backend_model": "Kimi-K2.6",
            "multimodal_max_image_bytes": 1024 * 1024,
        })()
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "gpt-5",
            "instructions": [{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}],
            "input": "Describe the image.",
        }))
        assert state.backend_route == "pd"
        assert state.contains_images
        assert chat["messages"][0]["content"][0]["type"] == "image_url"
        assert chat["messages"][1]["content"] == "Describe the image."
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_configured_multimodal_image_in_tool_output_is_preserved():
    previous_args = getattr(PROXY, "global_args", None)
    try:
        PROXY.global_args = type("Args", (), {
            "multimodal_backend_url": "http://vision/v1",
            "multimodal_backend_model": "Kimi-K2.6",
            "multimodal_max_image_bytes": 1024 * 1024,
        })()
        chat, state = run(PROXY.responses_request_to_chat_request({
            "model": "gpt-5",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_image",
                "output": [{"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="}],
            }],
        }))
        assert state.backend_route == "pd"
        assert state.contains_images
        assert chat["messages"][0]["role"] == "tool"
        assert chat["messages"][0]["content"][0]["type"] == "image_url"
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args


def test_multimodal_response_state_keeps_route_affinity():
    state = PROXY.ResponseRequestState(
        response_id="resp_vision",
        model="gpt-5",
        created_at=1,
        store=True,
        backend_route="multimodal_direct",
        backend_model="Kimi-K2.6",
    )
    payload = PROXY._state_payload(
        {"id": "resp_vision", "output": []},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]},
        state,
    )
    assert payload["_backend_route"] == "pd"
    assert payload["_contains_images"] is True
    assert payload["_backend_model"] != "Kimi-K2.6"
    assert payload["_normalized_messages"][0]["content"][0]["type"] == "image_url"


def test_stream_event_order():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        yield (
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_test", "DSv4", 1, False)
        result = []
        async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}
        ):
            result.append(chunk.decode())
        return result

    wire = run(collect())
    assert wire[-1] == "data: [DONE]\n\n"
    assert all(chunk.startswith("data: ") and "event: " not in chunk for chunk in wire[:-1])
    events = []
    for block in "\n".join(wire).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[3]["delta"] == "Hi"


def test_stream_function_call_conversion():
    async def chunks():
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"get_weather",'
            b'"arguments":"{\\"city\\":"}}]}}]}\n\n'
        )
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
            b'"function":{"arguments":" \\"Beijing\\"}"}}]},'
            b'"finish_reason":"tool_calls"}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_tool_stream", "DSv4", 1, False)
        state.tool_registry = {
            "get_weather": {
                "source_type": "function",
                "source_name": "get_weather",
                "backend_name": "get_weather",
            }
        }
        result = []
        async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}
        ):
            result.append(chunk.decode())
        return result

    events = []
    for block in "\n".join(run(collect())).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert events[-1]["response"]["output"][0]["type"] == "function_call"
    assert events[-1]["response"]["output"][0]["arguments"] == '{"city": "Beijing"}'


def test_stream_tool_search_call_uses_output_item_events_and_stable_mixed_indexes():
    async def chunks():
        reasoning = {"choices": [{"delta": {"reasoning_content": "Find the tool."}}]}
        text = {"choices": [{"delta": {"content": "Searching."}}]}
        tool = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_search_stream",
            "type": "function",
            "function": {
                "name": "__codex_tool_search",
                "arguments": '{"query":"shipping ETA"}',
            },
        }]}}]}
        finish = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        for payload in (reasoning, text, tool, finish):
            yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request({
            "model": "DSv4",
            "input": "Find a tool.",
            "tools": [_tool_search_declaration()],
        })
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}, input_tokens=1
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    added = [event for event in events if event["type"] == "response.output_item.added"]
    assert [event["output_index"] for event in added] == [0, 1, 2]
    assert [event["item"]["type"] for event in added] == [
        "reasoning", "message", "tool_search_call"
    ]
    assert "response.function_call_arguments.delta" not in event_types
    assert "response.function_call_arguments.done" not in event_types
    final = events[-1]["response"]
    assert [item["type"] for item in final["output"]] == [
        "reasoning", "message", "tool_search_call"
    ]
    assert final["output"][-1] == {
        "id": final["output"][-1]["id"],
        "type": "tool_search_call",
        "execution": "client",
        "call_id": "call_search_stream",
        "status": "completed",
        "arguments": {"query": "shipping ETA"},
    }
    done_indexes = [
        event["output_index"] for event in events
        if event["type"] == "response.output_item.done"
    ]
    assert done_indexes == [0, 1, 2]


def test_stream_bare_apply_patch_is_deferred_and_wrapped_through_exec():
    async def chunks():
        first = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_patch",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "*** Begin "},
        }]}, "finish_reason": None}]}
        second = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": "Patch\n*** End Patch"},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(first)}\n\n".encode()
        yield f"data: {json.dumps(second)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" not in event_types
    assert event_types == [
        "response.created",
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
    ]
    completed_call = events[-1]["response"]["output"][0]
    assert completed_call["type"] == "custom_tool_call"
    assert completed_call["name"] == "exec"
    assert completed_call["input"].startswith("await tools.apply_patch(")


def test_stream_bare_apply_patch_without_functions_exec_is_incomplete():
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_patch",
            "type": "function",
            "function": {"name": "apply_patch", "arguments": "*** Begin Patch"},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_patch_without_exec", "DSv4", 1, False)
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" not in event_types
    assert "response.output_item.done" not in event_types
    assert "response.completed" not in event_types
    assert next(event for event in events if event["type"] == "error")["error"]["code"] == (
        "invalid_function_arguments"
    )


def test_stream_undeclared_decoder_tool_is_not_forwarded():
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_unknown",
            "type": "function",
            "function": {"name": "__codec_unknown_functions_exec", "arguments": "{}"},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert "response.output_item.added" not in event_types
    assert "response.function_call_arguments.delta" not in event_types
    assert "response.completed" not in event_types
    assert next(event for event in events if event["type"] == "error")["error"]["code"] == "undeclared_tool"
    assert "response.incomplete" in event_types
    incomplete = next(event for event in events if event["type"] == "response.incomplete")
    assert incomplete["response"]["incomplete_details"] == {
        "reason": "undeclared_tool",
        "proxy_error_code": "undeclared_tool",
    }


def test_stream_codec_functions_exec_alias_is_deferred_and_mapped():
    async def chunks():
        first = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_alias",
            "type": "function",
            "function": {"name": "__codec_custom_functions_exec", "arguments": '{"input":"await tools.'},
        }]}, "finish_reason": None}]}
        second = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": 'update_goal({\\"status\\":\\"complete\\"});"}'},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(first)}\n\n".encode()
        yield f"data: {json.dumps(second)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" not in event_types
    assert event_types == [
        "response.created",
        "response.output_item.added",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.output_item.done",
        "response.completed",
    ]
    call = events[-1]["response"]["output"][0]
    assert call["name"] == "exec"
    assert call["namespace"] == "functions"
    assert call["input"] == 'await tools.update_goal({"status":"complete"});'


def test_stream_exact_exec_command_alias_is_deferred_and_wrapped():
    async def chunks():
        first = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_alias",
            "type": "function",
            "function": {"name": "__codec_exec_command", "arguments": '{"cmd":"p'},
        }]}, "finish_reason": None}]}
        second = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": 'wd"}'},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(first)}\n\n".encode()
        yield f"data: {json.dumps(second)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    assert events[-1]["type"] == "response.completed"
    call = events[-1]["response"]["output"][0]
    assert call["name"] == "exec"
    assert call["input"] == 'text(await tools.exec_command({"cmd": "pwd"}));'


def test_responses_drops_exact_dsml_tool_tail_and_counts_affected_request():
    previous_stats = PROXY.stats_collector
    PROXY.stats_collector = PROXY.StatsCollector()
    PROXY.stats_collector.record_responses_dsml_filter_request()

    async def chunks():
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"exec_command",'
            b'"arguments":"{\\"cmd\\":\\"true\\"}"}}]},"finish_reason":"tool_calls"}]}'
            b'\n\n'
        )
        yield (
            'data: {"choices":[{"delta":{"content":"\\n</｜DSML｜ calls>\\n"},'
            '"finish_reason":null}]}\n\n'.encode()
        )
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_dsml_filter", "DSv4", 1, False)
        state.tool_registry = {
            "exec_command": {
                "source_type": "function",
                "source_name": "exec_command",
                "backend_name": "exec_command",
            }
        }
        return [
            chunk.decode()
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
        ]

    try:
        wire = run(collect())
        assert "</｜DSML｜ calls>" not in "".join(wire)
        assert '"type": "response.function_call_arguments.delta"' in "".join(wire)
        assert PROXY.stats_collector.responses_dsml_filter_requests_total == 1
        assert PROXY.stats_collector.responses_dsml_filter_affected_requests_total == 1
    finally:
        PROXY.stats_collector = previous_stats


def test_responses_dsml_filter_keeps_unrelated_text_and_counts_request():
    previous_stats = PROXY.stats_collector
    PROXY.stats_collector = PROXY.StatsCollector()
    PROXY.stats_collector.record_responses_dsml_filter_request()

    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"ordinary text"},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_dsml_plain", "DSv4", 1, False)
        return [
            chunk.decode()
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
        ]

    try:
        wire = run(collect())
        assert '"delta": "ordinary text"' in "".join(wire)
        assert PROXY.stats_collector.responses_dsml_filter_requests_total == 1
        assert PROXY.stats_collector.responses_dsml_filter_affected_requests_total == 0
    finally:
        PROXY.stats_collector = previous_stats


def test_metrics_snapshot_includes_dsml_filter_counters():
    stats = PROXY.StatsCollector()
    stats.record_responses_dsml_filter_request()
    stats.record_responses_dsml_filter_affected()

    previous_args = getattr(PROXY, "global_args", None)
    PROXY.global_args = type(
        "Args",
        (),
        {
            "stats_interval": 10,
            "stats_enabled": True,
            "certain_kv_leak_log_path": "",
            "suspected_kv_leak_log_path": "",
        },
    )()
    try:
        snapshot = stats.snapshot([], [])
    finally:
        if previous_args is None:
            delattr(PROXY, "global_args")
        else:
            PROXY.global_args = previous_args

    assert snapshot["responses_dsml_filter"] == {
        "total_requests": 1,
        "affected_requests": 1,
    }


@pytest.mark.parametrize("alias", ["__codex_custom_exec_command", "codex_exec_command"])
def test_stream_custom_exec_command_input_alias_is_deferred_and_mapped(alias):
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_alias",
            "type": "function",
            "function": {
                "name": alias,
                "arguments": '{"input":"await tools.get_goal({});"}',
            },
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["input"] == "await tools.get_goal({});"


@pytest.mark.parametrize("declared", [True, False])
def test_stream_exec_command_input_rejection_preserves_error_code(declared):
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_invalid_exec", "type": "function",
            "function": {"name": "codex_exec_command",
                         "arguments": '{"input":"<tool_call>text(1)</tool_call>"}'
                         if declared else '{"input":"text(1);"}'},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        body = _codex_functions_exec_request() if declared else {
            "model": "DSv4", "input": "hello", "store": False,
        }
        _, state = await PROXY.responses_request_to_chat_request(body)
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    error = next(event for event in events if event["type"] == "error")
    expected_code = "invalid_function_arguments" if declared else "undeclared_tool"
    assert error["error"]["code"] == expected_code
    terminal = events[-1]
    assert terminal["type"] == "response.incomplete"
    assert terminal["response"]["incomplete_details"]["reason"] == expected_code
    assert not any(event["type"] in {
        "response.completed", "response.output_item.done",
        "response.custom_tool_call_input.delta",
    } for event in events)


def test_stream_decoder_codex_custom_containment_maps_declared_tool():
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_contained",
            "type": "function",
            "function": {
                "name": "decoder_prefix___codex_custom_functions_exec__suffix",
                "arguments": '{"input":"await tools.get_goal({});"}',
            },
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    final_call = events[-1]["response"]["output"][0]
    assert final_call["type"] == "custom_tool_call"
    assert final_call["name"] == "exec"
    assert final_call["namespace"] == "functions"
    assert final_call["input"] == "await tools.get_goal({});"
    assert "response.completed" in [event["type"] for event in events]


def test_stream_custom_functions_exec_suffix_maps_unique_declared_tool():
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_passthrough",
            "type": "function",
            "function": {
                "name": "__codecode_custom_functions_exec",
                "arguments": '{"input":"text(1);"}',
            },
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    final_call = events[-1]["response"]["output"][0]
    assert final_call["type"] == "custom_tool_call"
    assert final_call["name"] == "exec"
    assert final_call["namespace"] == "functions"
    assert final_call["input"] == "text(1);"
    assert "response.completed" in [event["type"] for event in events]


def test_stream_bare_exec_is_deferred_and_restored_through_functions_exec():
    async def chunks():
        payload = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "id": "call_exec",
            "type": "function",
            "function": {"name": "exec", "arguments": "await tools."},
        }]}, "finish_reason": None}]}
        final = {"choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": "apply_patch('*** Begin Patch');"},
        }]}, "finish_reason": "tool_calls"}]}
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield f"data: {json.dumps(final)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def collect():
        _, state = await PROXY.responses_request_to_chat_request(_codex_functions_exec_request())
        return [
            _sse_payload(chunk)
            async for chunk in PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}
            )
            if _sse_payload(chunk) is not None
        ]

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert "response.function_call_arguments.delta" not in event_types
    call = events[-1]["response"]["output"][0]
    assert call["type"] == "custom_tool_call"
    assert call["name"] == "exec"
    assert call["input"] == "await tools.apply_patch('*** Begin Patch');"


def test_stream_truncated_function_call_does_not_complete():
    async def chunks():
        yield (
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
            b'"type":"function","function":{"name":"get_weather",'
            b'"arguments":"{\\"city\\":"}}]}}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_bad_tool_stream", "DSv4", 1, False)
        result = []
        async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}
        ):
            result.append(chunk.decode())
        return result

    events = []
    for block in "\n".join(run(collect())).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    event_types = [event["type"] for event in events]
    assert "incomplete_function_call" in [
        event.get("error", {}).get("code") for event in events if event["type"] == "error"
    ]
    assert "response.function_call_arguments.done" not in event_types
    assert "response.output_item.done" not in event_types
    assert "response.completed" not in event_types
    assert "response.incomplete" in event_types


def test_stream_without_done_is_incomplete():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'

    async def collect():
        state = PROXY.ResponseRequestState("resp_no_done", "DSv4", 1, False)
        result = []
        async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}
        ):
            result.append(chunk.decode())
        return result

    wire = run(collect())
    assert "data: [DONE]\n\n" not in wire
    events = []
    for block in "\n".join(wire).split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: ") and line[6:] != "[DONE]":
                events.append(json.loads(line[6:]))
    assert "response.completed" not in [event["type"] for event in events]
    incomplete = next(event for event in events if event["type"] == "response.incomplete")
    assert incomplete["response"]["incomplete_details"]["reason"] == "decoder_stream_error"
    assert incomplete["response"]["incomplete_details"]["proxy_error_code"] == "decoder_stream_error"


def test_responses_stream_failure_has_no_done_terminator():
    async def collect():
        state = PROXY.ResponseRequestState("resp_failure", "DSv4", 1, False)
        return [
            chunk
            async for chunk in PROXY._responses_stream_failure(
                state,
                code="pd_first_output_timeout",
                message="Decoder did not produce a first SSE chunk.",
            )
        ]

    wire = run(collect())
    assert b"data: [DONE]\\n\\n" not in wire
    event_types = [
        json.loads(chunk.decode()[6:])["type"]
        for chunk in wire
        if chunk.startswith(b"data: {")
    ]
    assert event_types == ["error", "response.incomplete"]
    incomplete = _sse_payload(wire[-1])
    assert incomplete["response"]["incomplete_details"] == {
        "reason": "pd_first_output_timeout",
        "proxy_error_code": "pd_first_output_timeout",
    }


def test_responses_lifecycle_converts_internal_error_without_done_to_incomplete():
    async def chunks():
        yield b'data: {"error":{"type":"stream_error","code":"pd_first_output_timeout","message":"deadline exceeded"}}\n\n'

    async def collect():
        state = PROXY.ResponseRequestState("resp_timeout", "DSv4", 1, False)

        async def backend(*_args):
            return PROXY.StreamingResponse(chunks(), media_type="text/event-stream")

        wire = [
            chunk
            async for chunk in PROXY._responses_stream_lifecycle(
                None,
                {"messages": []},
                state,
                None,
                b"{}",
                backend_runner=backend,
            )
        ]
        return wire

    wire = run(collect())
    assert b"data: [DONE]\\n\\n" not in wire
    events = [
        json.loads(chunk.decode()[6:])
        for chunk in wire
        if chunk.startswith(b"data: {")
    ]
    event_types = [event["type"] for event in events]
    assert event_types[:2] == ["response.created", "response.in_progress"]
    assert event_types[-2:] == ["error", "response.incomplete"]
    assert events[-1]["response"]["incomplete_details"] == {
        "reason": "pd_first_output_timeout",
        "proxy_error_code": "pd_first_output_timeout",
    }
    assert "response.completed" not in event_types


def test_tool_search_with_web_search_uses_full_stream_lifecycle(monkeypatch):
    """Match the real client's simultaneous tool_search/web_search declarations."""
    async def check():
        chat, state = await PROXY.responses_request_to_chat_request({
            "model": "gpt-5.6-luna", "store": False, "stream": True,
            "input": "Find the JavaScript tool.",
            "tools": [_tool_search_declaration(), {"type": "web_search"}],
        })
        arguments = {"query": "node_repl javascript execution @oai/sky"}
        calls = 0

        async def chunks():
            for delta in (
                {"reasoning_content": "Find a tool."},
                {"content": "Searching."},
                {"tool_calls": [{
                    "index": 0, "id": "call_search", "type": "function",
                    "function": {
                        "name": "__codex_tool_search", "arguments": json.dumps(arguments),
                    },
                }]},
            ):
                yield f'data: {json.dumps({"choices": [{"delta": delta}]})}\n\n'.encode()
            yield b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            yield b"data: [DONE]\n\n"

        async def backend(*_args):
            nonlocal calls
            calls += 1
            assert calls == 1, "The proxy must not execute client-owned tool_search"
            return PROXY.StreamingResponse(chunks(), media_type="text/event-stream")

        async def unexpected_web_search(*_args):
            raise AssertionError("Declaring web_search must not execute it")

        monkeypatch.setattr(PROXY, "execute_web_search", unexpected_web_search)
        wire = [chunk async for chunk in PROXY._responses_stream_lifecycle(
            None, chat, state, None, b"{}", backend_runner=backend,
        )]
        assert calls == 1
        assert wire[-1] == b"data: [DONE]\n\n"
        events = [_sse_payload(chunk) for chunk in wire[:-1]]
        types = [event["type"] for event in events]
        assert types[:2] == ["response.created", "response.in_progress"]
        assert types[-1] == "response.completed"
        assert types.count("response.completed") == 1
        assert "error" not in types and "response.incomplete" not in types
        assert not any(t.startswith("response.function_call_arguments.") for t in types)
        added = [e for e in events if e["type"] == "response.output_item.added"]
        done = [e for e in events if e["type"] == "response.output_item.done"]
        assert [e["output_index"] for e in added] == [0, 1, 2]
        assert [e["output_index"] for e in done] == [0, 1, 2]
        assert [e["item"]["type"] for e in done] == ["reasoning", "tool_search_call", "message"]
        search = done[1]["item"]
        assert search == {
            "id": search["id"], "type": "tool_search_call", "execution": "client",
            "call_id": "call_search", "status": "completed", "arguments": arguments,
        }
        assert events[-1]["response"]["output"] == [e["item"] for e in done]
        assert "__codex_tool_search" not in b"".join(wire).decode()

        # Replay exactly the emitted item with a client-supplied namespace/custom result.
        next_chat, next_state = await PROXY.responses_request_to_chat_request({
            "model": "gpt-5.6-luna", "store": False,
            "tools": [_tool_search_declaration(), {"type": "web_search"}],
            "input": [search, _tool_search_output(tools=[{
                "type": "namespace", "name": "node_repl", "tools": [{
                    "type": "custom", "name": "js", "description": "Run JavaScript.",
                    "defer_loading": True, "format": {"type": "text"},
                }],
            }])],
        })
        assert next_chat["messages"][-1]["content"] == "Loaded tools: node_repl.js"
        assert next_state.tool_registry["__codex_custom_node_repl_js"]["source_type"] == "custom"

    run(check())


def test_web_tool_success_stream_has_text_part_done_and_done_terminator():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Search result"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def complete_tool_rounds(chat_response, *_args):
        return chat_response

    async def collect():
        previous = PROXY._run_proxy_web_tool_rounds
        PROXY._run_proxy_web_tool_rounds = complete_tool_rounds
        try:
            state = PROXY.ResponseRequestState("resp_web_terminal", "DSv4", 1, False)
            return [
                chunk
                async for chunk in PROXY._convert_web_chat_stream_to_responses_stream(
                    chunks(), {"messages": []}, None, state, None, emit_created=False
                )
            ]
        finally:
            PROXY._run_proxy_web_tool_rounds = previous

    wire = run(collect())
    assert wire[-1] == b"data: [DONE]\n\n"
    events = [_sse_payload(chunk) for chunk in wire[:-1]]
    assert [event["type"] for event in events] == [
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]


def _sse_payload(chunk: bytes) -> dict | None:
    for line in chunk.decode().splitlines():
        if line.startswith("data: "):
            if line[6:] == "[DONE]":
                return None
            return json.loads(line[6:])
    raise AssertionError("Missing SSE data payload")


async def _sse_json_payloads(chunks):
    async for chunk in chunks:
        payload = _sse_payload(chunk)
        if payload is not None:
            yield payload


def test_stream_converter_can_skip_outer_response_created():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        state = PROXY.ResponseRequestState("resp_no_duplicate_created", "DSv4", 1, False)
        return [
            payload
            async for payload in _sse_json_payloads(PROXY.convert_chat_stream_to_responses_stream(
                chunks(), state, {"messages": []}, emit_created=False
            ))
        ]

    events = run(collect())
    assert events[0]["type"] == "response.output_item.added"
    assert "response.created" not in [event["type"] for event in events]


def test_responses_stream_reuses_request_token_count_before_first_output():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    previous_count = PROXY.count_prefill_and_decode_tokens
    count_calls = 0

    def slow_count(*_args, **_kwargs):
        nonlocal count_calls
        count_calls += 1
        time.sleep(0.2)
        return 999, 999

    async def collect():
        state = PROXY.ResponseRequestState("resp_reused_input_tokens", "DSv4", 1, False)
        stream = PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}, input_tokens=7, emit_created=False
        )
        started = time.monotonic()
        events = [_sse_payload(await anext(stream))]
        first_event_elapsed = time.monotonic() - started
        async for chunk in stream:
            payload = _sse_payload(chunk)
            if payload is not None:
                events.append(payload)
        return first_event_elapsed, events

    PROXY.count_prefill_and_decode_tokens = slow_count
    try:
        first_event_elapsed, events = run(collect())
    finally:
        PROXY.count_prefill_and_decode_tokens = previous_count

    assert count_calls == 0
    assert first_event_elapsed < 0.1
    completed = next(event for event in events if event["type"] == "response.completed")
    assert completed["response"]["usage"]["input_tokens"] == 7


def test_responses_stream_aclose_does_not_emit_terminal_events():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        await asyncio.Event().wait()

    async def close_stream():
        state = PROXY.ResponseRequestState("resp_converter_aclose", "DSv4", 1, False)
        stream = PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, {"messages": []}, input_tokens=1, emit_created=False
        )
        first = _sse_payload(await anext(stream))
        assert first["type"] == "response.output_item.added"
        await stream.aclose()

    run(close_stream())


def test_responses_lifecycle_emits_created_before_backend_is_ready():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        started = asyncio.Event()
        release = asyncio.Event()
        previous = PROXY._run_responses_backend

        async def delayed_backend(*_args):
            started.set()
            await release.wait()
            return PROXY.StreamingResponse(chunks(), media_type="text/event-stream")

        PROXY._run_responses_backend = delayed_backend
        try:
            state = PROXY.ResponseRequestState("resp_early_created", "DSv4", 1, False)
            stream = PROXY._responses_stream_lifecycle(None, {"messages": []}, state, None, b"{}")
            first = _sse_payload(await anext(stream))
            in_progress = _sse_payload(await anext(stream))
            next_event = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 1)
            assert not next_event.done()
            release.set()
            events = [first, in_progress, _sse_payload(await next_event)]
            async for chunk in stream:
                payload = _sse_payload(chunk)
                if payload is not None:
                    events.append(payload)
            return events
        finally:
            PROXY._run_responses_backend = previous

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert event_types[0] == "response.created"
    assert event_types.count("response.created") == 1
    assert event_types[1] == "response.in_progress"
    assert "response.completed" in event_types


def test_responses_lifecycle_heartbeats_while_backend_is_pending():
    async def chunks():
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        started = asyncio.Event()
        release = asyncio.Event()
        previous_backend = PROXY._run_responses_backend
        previous_args = getattr(PROXY, "global_args", None)

        async def delayed_backend(*_args):
            started.set()
            await release.wait()
            return PROXY.StreamingResponse(chunks(), media_type="text/event-stream")

        PROXY._run_responses_backend = delayed_backend
        PROXY.global_args = type("Args", (), {"responses_sse_heartbeat_seconds": 0.01})()
        try:
            state = PROXY.ResponseRequestState("resp_heartbeat_prefill", "DSv4", 1, False)
            stream = PROXY._responses_stream_lifecycle(None, {"messages": []}, state, None, b"{}")
            events = [_sse_payload(await anext(stream)), _sse_payload(await anext(stream))]
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 1)
            events.append(_sse_payload(await asyncio.wait_for(pending, 1)))
            release.set()
            async for chunk in stream:
                payload = _sse_payload(chunk)
                if payload is not None:
                    events.append(payload)
            return events
        finally:
            PROXY._run_responses_backend = previous_backend
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

    events = run(collect())
    event_types = [event["type"] for event in events]
    assert event_types[0] == "response.created"
    assert event_types.count("response.created") == 1
    assert "response.in_progress" in event_types
    assert "response.completed" in event_types


def test_responses_lifecycle_heartbeats_until_first_decoder_event():
    async def delayed_chunks():
        await asyncio.sleep(0.03)
        yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
        yield b"data: [DONE]\n\n"

    async def collect():
        previous_backend = PROXY._run_responses_backend
        previous_args = getattr(PROXY, "global_args", None)

        async def ready_backend(*_args):
            return PROXY.StreamingResponse(delayed_chunks(), media_type="text/event-stream")

        PROXY._run_responses_backend = ready_backend
        PROXY.global_args = type("Args", (), {"responses_sse_heartbeat_seconds": 0.01})()
        try:
            state = PROXY.ResponseRequestState("resp_heartbeat_decoder", "DSv4", 1, False)
            return [
                payload
                async for payload in _sse_json_payloads(PROXY._responses_stream_lifecycle(
                    None, {"messages": []}, state, None, b"{}"
                ))
            ]
        finally:
            PROXY._run_responses_backend = previous_backend
            if previous_args is None:
                delattr(PROXY, "global_args")
            else:
                PROXY.global_args = previous_args

    events = run(collect())
    event_types = [event["type"] for event in events]
    first_real_event = next(
        index for index, event_type in enumerate(event_types)
        if event_type not in {"response.created", "response.in_progress"}
    )
    assert "response.in_progress" in event_types[1:first_real_event]
    assert "response.in_progress" not in event_types[first_real_event:]
    assert "response.completed" in event_types


def test_responses_lifecycle_cancels_pending_first_decoder_event():
    async def check_cancellation():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        previous_backend = PROXY._run_responses_backend

        async def delayed_chunks():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield b""

        async def ready_backend(*_args):
            return PROXY.StreamingResponse(delayed_chunks(), media_type="text/event-stream")

        PROXY._run_responses_backend = ready_backend
        try:
            state = PROXY.ResponseRequestState("resp_cancel_first_event", "DSv4", 1, False)
            stream = PROXY._responses_stream_lifecycle(None, {"messages": []}, state, None, b"{}")
            await anext(stream)
            await anext(stream)
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 1)
            pending.cancel()
            with suppress(asyncio.CancelledError):
                await pending
            await asyncio.wait_for(cancelled.wait(), 1)
        finally:
            PROXY._run_responses_backend = previous_backend

    run(check_cancellation())


def test_responses_lifecycle_closes_inner_chat_stream_after_first_event():
    async def check_close_propagation():
        inner_closed = asyncio.Event()

        async def decoder_chunks():
            try:
                yield b'data: {"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                await asyncio.Event().wait()
            finally:
                inner_closed.set()

        async def backend(*_args):
            return PROXY.StreamingResponse(
                decoder_chunks(), media_type="text/event-stream"
            )

        state = PROXY.ResponseRequestState("resp_close_inner", "DSv4", 1, False)
        stream = PROXY._responses_stream_lifecycle(
            None,
            {"messages": []},
            state,
            None,
            b"{}",
            backend_runner=backend,
        )
        assert _sse_payload(await anext(stream))["type"] == "response.created"
        assert _sse_payload(await anext(stream))["type"] == "response.in_progress"
        assert _sse_payload(await anext(stream))["type"] == "response.output_item.added"
        await stream.aclose()
        await asyncio.wait_for(inner_closed.wait(), 1)

    run(check_close_propagation())


def test_handle_responses_returns_created_before_backend_task_starts():
    async def check_handler():
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": b'{"model":"DSv4","input":"hello","stream":true,"store":false}',
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        request = PROXY.Request({
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/responses",
            "raw_path": b"/v1/responses",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8077),
        }, receive)
        started = asyncio.Event()
        previous = PROXY._run_responses_backend

        async def blocked_backend(*_args):
            started.set()
            await asyncio.Event().wait()

        PROXY._run_responses_backend = blocked_backend
        try:
            response = await PROXY.handle_responses(request)
            assert isinstance(response, PROXY.StreamingResponse)
            first = _sse_payload(await anext(response.body_iterator))
            assert first["type"] == "response.created"
            assert not started.is_set()
            await response.body_iterator.aclose()
        finally:
            PROXY._run_responses_backend = previous

    run(check_handler())


def test_responses_lifecycle_converts_early_backend_error_to_sse_terminal_error():
    async def collect():
        previous = PROXY._run_responses_backend

        async def failed_backend(*_args):
            return PROXY.JSONResponse(
                status_code=502,
                content={"error": {"code": "prefill_error", "message": "Prefill failed"}},
            )

        PROXY._run_responses_backend = failed_backend
        try:
            state = PROXY.ResponseRequestState("resp_prefill_failure", "DSv4", 1, False)
            return [
                payload
                async for payload in _sse_json_payloads(PROXY._responses_stream_lifecycle(
                    None, {"messages": []}, state, None, b"{}"
                ))
            ]
        finally:
            PROXY._run_responses_backend = previous

    events = run(collect())
    assert [event["type"] for event in events] == [
        "response.created", "response.in_progress", "error", "response.incomplete"
    ]
    assert events[2]["error"]["code"] == "prefill_error"


def test_responses_lifecycle_marks_prefill_disconnect_before_backend_handoff():
    async def check_cancellation():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        previous = PROXY._run_responses_backend

        async def blocked_backend(*_args):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        PROXY._run_responses_backend = blocked_backend
        try:
            state = PROXY.ResponseRequestState("resp_cancel_prefill", "DSv4", 1, False)
            state.pipeline_leak_state = PROXY.RequestLeakState(
                request_id="response-prefill-disconnect",
                stream=True,
            )
            stream = PROXY._responses_stream_lifecycle(None, {"messages": []}, state, None, b"{}")
            await anext(stream)
            await anext(stream)
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(started.wait(), 1)
            pending.cancel()
            try:
                await pending
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("Expected lifecycle cancellation")
            await asyncio.wait_for(cancelled.wait(), 1)
            assert state.pipeline_leak_state.client_disconnected is True
            assert state.pipeline_leak_state.disconnect_phase == "prefill_pending"
            assert state.pipeline_leak_state.drain_after_prefill is True
        finally:
            PROXY._run_responses_backend = previous

    run(check_cancellation())


class _CountingTokenizer:
    def encode(self, value, add_special_tokens=False):
        return list(value)


def test_stream_output_accounting_includes_reasoning_and_tool_arguments():
    previous_tokenizer = PROXY.tokenizer
    try:
        PROXY.tokenizer = _CountingTokenizer()
        accounting = PROXY.StreamOutputAccounting()
        attempt = accounting.start_attempt()
        accounting.observe(
            attempt,
            PROXY.StreamChoiceInfo(
                text="Hi",
                reasoning="Think",
                tool_arguments={"call_1": '{"city":"Beijing"}'},
            ),
            10.0,
        )
        accounting.observe(attempt, PROXY.StreamChoiceInfo(text="!"), 12.0)
        assert accounting.total_tokens == len('Hi!') + len("Think") + len('{"city":"Beijing"}')
        assert accounting.category_tokens() == (3, 5, len('{"city":"Beijing"}'))
        assert accounting.token_source == "tokenizer_estimate"
        assert accounting.output_window() == (10.0, 12.0)
    finally:
        PROXY.tokenizer = previous_tokenizer


def test_stream_output_accounting_prefers_final_backend_usage_and_sums_retries():
    accounting = PROXY.StreamOutputAccounting()
    first = accounting.start_attempt()
    accounting.observe(first, PROXY.StreamChoiceInfo(text="discarded", backend_completion_tokens=3), 1.0)
    second = accounting.start_attempt()
    accounting.observe(second, PROXY.StreamChoiceInfo(reasoning="answer", backend_completion_tokens=5), 2.0)
    assert accounting.total_tokens == 8
    assert accounting.token_source == "backend_usage"


def test_stream_choice_info_reads_final_usage_without_choices_and_tool_deltas():
    usage_only = PROXY.extract_stream_choice_info({"usage": {"completion_tokens": 7}})
    assert usage_only.backend_completion_tokens == 7
    tool = PROXY.extract_stream_choice_info({
        "choices": [{"delta": {"tool_calls": [{
            "index": 0,
            "function": {"arguments": '{"path":"/tmp"}'},
        }]}}],
    })
    assert tool.tool_arguments == {"0": '{"path":"/tmp"}'}
