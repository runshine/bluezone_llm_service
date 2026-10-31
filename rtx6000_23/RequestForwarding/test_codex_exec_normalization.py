"""Exec normalization and Responses protocol tests. No generated code is run."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import codex_exec_normalization as NORMALIZER
from test_responses_adapter import PROXY, run, _sse_payload, _codex_functions_wait_request
from test_responses_codex_aliases import request_body, COMMAND, PATCH, declared_command
from test_responses_web_rounds import (
    MemoryStore, NullTrace, call, chat_response, chat_stream,
)


ALIASES = sorted(PROXY.CODEX_FUNCTIONS_EXEC_DECODER_ALIASES | PROXY.CODEX_EXEC_COMMAND_DECODER_ALIASES)
MODES = ["normal", "stream", "web", "web_stream"]
JAVASCRIPT = 'const result = await tools.exec_command({cmd: "pwd"}); text(result.output);'
SHELL = "printf '%s\\n' '中文 `quoted` $(literal)' | cat"


async def make_state(*, direct=False, alias_names=True):
    body = request_body()
    if direct:
        body["tools"] = [declared_command()]
        body["input"] = "Run a command."
    chat, state = await PROXY.responses_request_to_chat_request(body)
    if alias_names:
        # Exercise even __codex_custom_functions_exec as a Decoder alias rather
        # than accidentally taking the exact-declaration path for that name.
        for key, spec in list(state.tool_registry.items()):
            if PROXY._is_functions_exec_spec(spec):
                state.tool_registry.pop(key)
                spec["backend_name"] = "__declared_exec"
                state.tool_registry["__declared_exec"] = spec
                for tool in chat["tools"]:
                    if tool["function"]["name"] == key:
                        tool["function"]["name"] = "__declared_exec"
    return chat, state


async def convert(mode, upstream, chat, state):
    if mode == "normal":
        return PROXY.chat_result_to_response_object(upstream, chat, state), []
    if mode == "web":
        canonical = await PROXY._run_proxy_web_tool_rounds(upstream, chat, None, state, NullTrace())
        return PROXY.chat_result_to_response_object(canonical, chat, state), []
    iterator = (PROXY.convert_chat_stream_to_responses_stream(chat_stream(upstream), state, chat)
                if mode == "stream" else PROXY._convert_web_chat_stream_to_responses_stream(
                    chat_stream(upstream), chat, None, state, NullTrace()))
    events = [_sse_payload(chunk) async for chunk in iterator if _sse_payload(chunk) is not None]
    completed = [event["response"] for event in events if event["type"] == "response.completed"]
    return completed[0] if completed else None, events


def expected_command(command):
    return "text(await tools.exec_command(" + json.dumps(command, ensure_ascii=False) + "));"


@pytest.mark.parametrize("alias", ALIASES + ["exec", "exec_command",
    "decoder_prefix___declared_custom_functions_exec__suffix", "__other_custom_functions_exec"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("envelope", ["javascript", "raw_javascript", "command", "inner_command", "shell"])
def test_all_names_and_formats_across_four_paths(alias, mode, envelope):
    async def exercise():
        chat, state = await make_state()
        if alias.startswith("decoder_prefix") or alias == "__other_custom_functions_exec":
            spec = state.tool_registry.pop("__declared_exec")
            spec["backend_name"] = "__declared_custom_functions_exec"
            state.tool_registry[spec["backend_name"]] = spec
        arguments = {
            "javascript": json.dumps({"input": JAVASCRIPT}),
            "raw_javascript": JAVASCRIPT,
            "command": json.dumps(COMMAND),
            "inner_command": json.dumps({"input": json.dumps(COMMAND)}),
            "shell": json.dumps({"input": SHELL}),
        }[envelope]
        upstream = chat_response([call(alias, "stable_id", arguments)])
        original = copy.deepcopy(upstream)
        response, events = await convert(mode, upstream, chat, state)
        assert upstream == original
        assert response is not None, events
        item = response["output"][0]
        assert item["type"] == "custom_tool_call"
        assert (item["namespace"], item["name"], item["call_id"]) == ("functions", "exec", "stable_id")
        assert item["input"] == (JAVASCRIPT if "javascript" in envelope else
                                 expected_command({"cmd": SHELL} if envelope == "shell" else COMMAND))
        assert not any(key in json.dumps(response) for key in ["resolution_kind", "payload_type", "normalization_ms"])
        if events:
            assert events[-1]["type"] == "response.completed"
            assert not any(e["type"].startswith("response.function_call_arguments.") for e in events)
    run(exercise())


@pytest.mark.parametrize("alias", ALIASES)
@pytest.mark.parametrize("source", ["pwd", "ls", "ls -l", "const r = await tools."])
def test_all_aliases_reject_ambiguous_or_damaged_source(alias, source):
    async def exercise():
        chat, state = await make_state()
        response, events = await convert("stream", chat_response([
            call(alias, "bad", json.dumps({"input": source}))]), chat, state)
        assert response is None
        assert events[-1]["response"]["incomplete_details"]["reason"] == "invalid_function_arguments"
        assert not any(e["type"] in {"response.output_item.added", "response.output_item.done"} for e in events)
    run(exercise())


@pytest.mark.parametrize("source,language", [
    (JAVASCRIPT, "javascript"), ("text(1);", "javascript"),
    ("await Promise.all([tools.get_goal({}), tools.get_goal({})]);", "javascript"),
    ('const s = `中文 ${1 + 2}`; text(s);', "javascript"),
    ('text("ps aux | grep python");', "javascript"),
    ("// const in comment\ntext(1);", "javascript"),
    (SHELL, "shell"), ("echo 'const r = await tools.exec_command(...)'", "shell"),
    ("echo hello > /tmp/example.txt", "shell"),
    ('for f in *.log; do printf "%s\\n" "$f"; done', "shell"),
    ("cat <<'EOF'\nconst r = await tools.exec_command(...)\nEOF\n", "shell"),
    ("A=value printf '%s' hello", "shell"),
])
def test_static_language_examples(source, language):
    assert NORMALIZER.classify_source(source)[0] == language


@pytest.mark.parametrize("source", ["pwd", "ls -l", "// comment\nconst r = await tools.", "tools.exec_command("])
def test_source_ambiguity_and_malformed_javascript(source):
    with pytest.raises(NORMALIZER.ExecNormalizationError):
        NORMALIZER.classify_source(source)


@pytest.mark.parametrize("arguments", [
    '{"input":"text(1);","input":"text(2);"}',
    '{"cmd":"pwd","yield_time_ms":NaN}',
    '{"input":"{\\"cmd\\":\\"pwd\\",\\"cmd\\":\\"ls\\"}"}',
    '{"input":"{\\"cmd\\":\\"pwd\\",\\"yield_time_ms\\":Infinity}"}',
    '{"cmd":"pwd","input":"text(1);"}',
    '{"input":"{\\"input\\":\\"text(1);\\"}"}',
    '{"input":"[]"}', '{"input":"null"}', '{"input":"123"}',
    '{"input":"{\\"foo\\":\\"bar\\"}"}', '{"input":',
    '{"input":"text(1);","extra":1}', '{"cmd":"pwd","extra":1}',
    '{"cmd":" "}', '{"cmd":"pwd","yield_time_ms":true}',
    '{"cmd":"pwd","workdir":[]}', '[]', '123', 'true', 'null', '"pwd"',
    '{"input":null}', '{"input":{}}', '{"input":""}',
    '{"input":"```javascript\\ntext(1);\\n```"}',
    '{"input":"<tool_call>text(1);</tool_call>"}',
])
def test_bad_envelopes_do_not_fall_back_to_execution(arguments):
    async def exercise():
        _, state = await make_state()
        with pytest.raises(PROXY.ResponsesRequestError) as error:
            PROXY._normalize_decoder_tool_call("__codex_exec_command", arguments, "bad", state)
        assert error.value.code == "invalid_function_arguments"
    run(exercise())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("namespace", [False, True])
def test_direct_command_receives_full_original_parameters(mode, namespace):
    async def exercise():
        body = request_body()
        tool = declared_command()
        body["tools"] = ([{"type": "namespace", "name": "functions", "tools": [tool]}]
                         if namespace else [tool])
        chat, state = await PROXY.responses_request_to_chat_request(body)
        command = dict(COMMAND, cmd="printf '%s' '`x` $(whoami) \"中文\"'\npwd", shell="/bin/bash", login=False)
        response, events = await convert(mode, chat_response([
            call("__codex_exec_command", "direct", json.dumps(command))]), chat, state)
        item = response["output"][0]
        assert item["type"] == "function_call"
        assert item["name"] == "exec_command"
        assert json.loads(item["arguments"]) == command
        assert item.get("namespace") == ("functions" if namespace else None)
        if mode == "stream":
            assert [e["item"]["type"] for e in events if e["type"] == "response.output_item.added"] == ["function_call"]
            assert not any("custom_tool_call" in e["type"] for e in events)
    run(exercise())


def test_exact_declared_exec_contract_and_arbitrary_declared_tools_are_preserved():
    async def exercise():
        _, state = await make_state(alias_names=False)
        name = next(n for n, s in state.tool_registry.items() if PROXY._is_functions_exec_spec(s))
        for source in ["pwd", "ls -l", "const incomplete =", '{"cmd":"pwd"}']:
            args = json.dumps({"input": source})
            spec, normalized, deferred = PROXY._normalize_decoder_tool_call(name, args, "exact", state)
            assert normalized == args and not deferred
        state.tool_registry["__codex_exec_command"] = {
            "backend_name": "__codex_exec_command", "source_name": "unrelated", "source_type": "function"}
        assert PROXY._normalize_decoder_tool_call("__codex_exec_command", '{"x":1}', "direct", state)[1] == '{"x":1}'
    run(exercise())


def test_javascript_is_not_sent_to_shell_and_unknown_names_are_not_enabled_by_content():
    async def exercise():
        _, state = await make_state(direct=True)
        for name in ["__codex_exec_command", "__codex_bash", "exec_query", "CODEX_EXEC"]:
            with pytest.raises(PROXY.ResponsesRequestError) as error:
                PROXY._normalize_decoder_tool_call(name, json.dumps({"input": JAVASCRIPT}), "bad", state)
            assert error.value.code == "undeclared_tool"
    run(exercise())


@pytest.mark.parametrize("mode", ["stream", "web_stream"])
def test_no_compatible_tool_is_published_before_all_calls_validate(mode):
    async def exercise():
        chat, state = await make_state()
        response, events = await convert(mode, chat_response([
            call("__exec", "good", json.dumps({"input": JAVASCRIPT})),
            call("__codex_exec_command", "bad", '{"input":"pwd"}'),
        ]), chat, state)
        assert response is None
        assert events[-1]["type"] == "response.incomplete"
        assert not any(e["type"].startswith("response.custom_tool_call_input") or
                       e["type"] in {"response.output_item.added", "response.output_item.done"} for e in events)
    run(exercise())


def test_fragmentation_preserves_order_and_never_emits_unvalidated_arguments():
    async def exercise():
        chat, state = await make_state()
        arguments = json.dumps({"input": JAVASCRIPT})
        async def chunks():
            deltas = [{"reasoning_content": "considering"}, {"content": "before"}]
            for i, char in enumerate(arguments):
                fn = {"arguments": char}
                tool = {"index": 0, "function": fn}
                if i == 0:
                    fn["name"] = "__exec"
                    tool["id"] = "split"
                deltas.append({"tool_calls": [tool]})
            deltas.append({"content": "after"})
            for delta in deltas:
                yield PROXY._sse_event("unused", {"choices": [{"delta": delta}]})
            yield PROXY._sse_event("unused", {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
            yield b"data: [DONE]\n\n"
        events = [_sse_payload(chunk) async for chunk in PROXY.convert_chat_stream_to_responses_stream(
            chunks(), state, chat) if _sse_payload(chunk) is not None]
        final = events[-1]["response"]["output"]
        assert [item["type"] for item in final] == ["reasoning", "message", "custom_tool_call"]
        assert final[1]["content"][0]["text"] == "beforeafter"
        assert final[2]["input"] == JAVASCRIPT
        first_argument_event = next(i for i, e in enumerate(events) if e["type"] == "response.custom_tool_call_input.delta")
        assert any(e["type"] == "response.output_text.delta" and e["delta"] == "after"
                   for e in events[:first_argument_event])
    run(exercise())


def test_same_chunk_text_and_tool_are_not_lost():
    async def exercise():
        chat, state = await make_state()
        response, _ = await convert("stream", chat_response([
            call("__exec", "t", json.dumps({"input": JAVASCRIPT}))], content="visible", reasoning="reason"), chat, state)
        assert {i["type"] for i in response["output"]} == {"message", "reasoning", "custom_tool_call"}
        assert next(i for i in response["output"] if i["type"] == "message")["content"][0]["text"] == "visible"
    run(exercise())


@pytest.mark.parametrize("mode", MODES)
def test_conversion_and_logging_happen_once(mode, caplog, monkeypatch):
    count = 0
    original = PROXY.normalize_exec
    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(PROXY, "normalize_exec", counted)
    async def exercise():
        chat, state = await make_state()
        upstream = chat_response([call("__exec", "once", json.dumps(COMMAND))])
        with caplog.at_level("INFO"):
            response, _ = await convert(mode, upstream, chat, state)
        assert response["output"][0]["input"] == expected_command(COMMAND)
    run(exercise())
    assert count == 1
    assert caplog.text.count("codex_decoder_tool_alias_mapped") == 1
    assert "envelope_type=cmd_object" in caplog.text and "payload_type=command_json" in caplog.text
    assert COMMAND["workdir"] not in caplog.text


def test_previous_response_restores_normalized_call_without_classifying_again(monkeypatch):
    async def exercise():
        chat, state = await make_state()
        upstream = chat_response([call("__exec", "history", json.dumps(COMMAND))])
        final, _ = await convert("normal", upstream, chat, state)
        store = MemoryStore()
        await store.put(PROXY._state_payload(final, chat, state))
        monkeypatch.setattr(PROXY, "response_state_store", store)
        def forbidden(*args, **kwargs):
            raise AssertionError("History must not be reclassified")
        monkeypatch.setattr(PROXY, "normalize_exec", forbidden)
        followup, _ = await PROXY.responses_request_to_chat_request({
            "model": "DSv4", "previous_response_id": state.response_id, "store": False,
            "input": [{"type": "custom_tool_call_output", "call_id": "history", "output": "done"}],
        })
        assistant = next(m for m in followup["messages"] if m.get("tool_calls"))
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["input"] == expected_command(COMMAND)
        assert followup["messages"][-1]["content"] == "done"
        public = PROXY._public_response_state(await store.get(state.response_id))
        assert not any(key.startswith("_") for key in public)
    run(exercise())


def test_timeout_and_missing_grammars_fail_closed(monkeypatch):
    import tree_sitter
    class TimeoutParser:
        def __init__(self, language):
            pass
        def parse(self, source):
            raise ValueError("Parsing failed")
    with monkeypatch.context() as patch:
        patch.setattr(tree_sitter, "Parser", TimeoutParser)
        with pytest.raises(NORMALIZER.ExecNormalizationError, match="parser_timeout"):
            NORMALIZER.classify_source(JAVASCRIPT)
    def unavailable():
        raise ImportError("missing grammar")
    monkeypatch.setattr(NORMALIZER, "_languages", unavailable)
    with pytest.raises(RuntimeError, match="startup self-test"):
        NORMALIZER.validate_parsers()


def test_startup_self_test_precedes_any_backend_client_creation(monkeypatch):
    def fail():
        raise RuntimeError("parser startup failure")
    def forbidden(*args, **kwargs):
        raise AssertionError("Backend clients must not be created")
    monkeypatch.setattr(PROXY, "validate_parsers", fail)
    monkeypatch.setattr(PROXY, "ProxyState", forbidden)
    async def exercise():
        with pytest.raises(RuntimeError, match="parser startup failure"):
            async with PROXY.lifespan(PROXY.app):
                raise AssertionError("Startup should fail closed")
    run(exercise())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,arguments", [
    ("apply_patch", PATCH),
    *[(alias, json.dumps({"input": PATCH})) for alias in sorted(PROXY.CODEX_PATCH_DECODER_ALIASES)],
    ("write_stdin", '{"session_id":1,"chars":"ok"}'),
    ("update_plan", '{"plan":[]}'), ("update_goal", '{"status":"complete"}'),
    ("__codex_namespace_functions_update_goal", '{"status":"complete"}'),
    ("create_goal", '{"objective":"review"}'), ("get_goal", '{}'),
    ("view_image", '{"path":"/tmp/image.png"}'),
])
def test_existing_non_exec_conversions_across_all_paths(mode, name, arguments):
    async def exercise():
        chat, state = await make_state()
        response, events = await convert(mode, chat_response([call(name, "kept", arguments)]), chat, state)
        assert response is not None, events
        item = response["output"][0]
        target = "update_goal" if name == "__codex_namespace_functions_update_goal" else name
        if name in PROXY.CODEX_PATCH_DECODER_ALIASES:
            target, payload = "apply_patch", PATCH
        else:
            payload = arguments if target == "apply_patch" else json.loads(arguments)
        assert item["input"] == "await tools." + target + "(" + json.dumps(
            payload, ensure_ascii=False, **({} if target == "apply_patch" else {"separators": (",", ":")})
        ) + ");"
        assert item["call_id"] == "kept"
    run(exercise())


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("alias", sorted(PROXY.CODEX_WAIT_DECODER_ALIASES))
def test_wait_alias_uses_function_protocol_in_all_paths(mode, alias):
    async def exercise():
        body = _codex_functions_wait_request()
        body["store"] = False
        chat, state = await PROXY.responses_request_to_chat_request(body)
        arguments = {"cell_id": "test", "yield_time_ms": 1000, "max_tokens": 200}
        response, events = await convert(mode, chat_response([
            call(alias, "wait_id", json.dumps(arguments))]), chat, state)
        assert response is not None, events
        item = response["output"][0]
        assert (item["type"], item["name"], item["namespace"]) == ("function_call", "wait", "functions")
        assert json.loads(item["arguments"]) == arguments
        assert item["call_id"] == "wait_id"
    run(exercise())


@pytest.mark.parametrize("kind", ["direct_ambiguous", "exec_ambiguous", "missing_schema", "invalid_local_ref", "missing_nested_contract", "remote_base_uri"])
def test_target_and_schema_failure_modes(kind):
    async def exercise():
        _, state = await make_state()
        exec_spec = next(iter(state.tool_registry.values()))
        if kind == "exec_ambiguous":
            state.tool_registry["other"] = dict(exec_spec, backend_name="other")
            args = {"input": JAVASCRIPT}
        elif kind in {"missing_schema", "invalid_local_ref", "direct_ambiguous", "remote_base_uri"}:
            spec = {"source_name": "exec_command", "source_type": "function", "backend_name": "command"}
            if kind == "invalid_local_ref":
                spec["parameters"] = {"type": "object", "properties": {"cmd": {"$ref": "#/missing"}}}
            if kind == "remote_base_uri":
                spec["parameters"] = {"type": "object", "properties": {
                    "cmd": {"id": "https://invalid.example/schema", "$ref": "#/missing"}}}
            if kind == "direct_ambiguous":
                spec["parameters"] = copy.deepcopy(PROXY.CODEX_EXEC_COMMAND_PARAMETERS)
                state.tool_registry["second_command"] = dict(spec, backend_name="second_command")
            state.tool_registry["command"] = spec
            args = COMMAND
        else:
            exec_spec["description"] = None
            args = COMMAND
        with pytest.raises(PROXY.ResponsesRequestError) as error:
            PROXY._normalize_decoder_tool_call("__exec", json.dumps(args), "bad", state)
        assert error.value.code == ("invalid_function_arguments" if kind in {"invalid_local_ref", "remote_base_uri"} else "undeclared_tool")
    run(exercise())


@pytest.mark.parametrize("arguments", ['{"cmd":"pwd","yield_time_ms":1e999}', '{"cmd":"pwd","max_output_tokens":-1}'])
def test_nonfinite_and_negative_command_budgets_are_rejected(arguments):
    async def exercise():
        _, state = await make_state()
        with pytest.raises(PROXY.ResponsesRequestError) as error:
            PROXY._normalize_decoder_tool_call("__exec", arguments, "bad", state)
        assert error.value.code == "invalid_function_arguments"
    run(exercise())


def test_no_shell_or_javascript_subprocess_is_used_for_classification(monkeypatch):
    import subprocess
    def forbidden(*args, **kwargs):
        raise AssertionError("Generated content must not be executed")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    NORMALIZER.validate_parsers()
    assert NORMALIZER.classify_source(JAVASCRIPT)[0] == "javascript"
    assert NORMALIZER.classify_source(SHELL)[0] == "shell"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("call_id", [None, "", ["wrong_type"]])
def test_compatible_call_id_is_never_synthesized(mode, call_id):
    async def exercise():
        chat, state = await make_state()
        upstream = chat_response([call("__exec", call_id, json.dumps({"input": JAVASCRIPT}))])
        if mode in {"normal", "web"}:
            with pytest.raises((PROXY.ResponsesRequestError, PROXY.ResponsesBackendError)) as error:
                await convert(mode, upstream, chat, state)
            assert error.value.code == "incomplete_function_call"
        else:
            response, events = await convert(mode, upstream, chat, state)
            assert response is None
            assert events[-1]["response"]["incomplete_details"]["reason"] == "incomplete_function_call"
    run(exercise())


def test_cancel_during_compatible_call_does_not_classify_or_emit_terminals(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Incomplete canceled arguments must not be classified")
    monkeypatch.setattr(PROXY, "normalize_exec", forbidden)
    async def exercise():
        chat, state = await make_state()
        async def chunks():
            yield PROXY._sse_event("unused", {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "pending", "function": {"name": "__exec", "arguments": '{"input":"const'}
            }]}}]})
            yield PROXY._sse_event("unused", {"choices": [{"delta": {"content": "waiting"}}]})
        stream = PROXY.convert_chat_stream_to_responses_stream(chunks(), state, chat)
        events = []
        async for chunk in stream:
            event = _sse_payload(chunk)
            events.append(event)
            if event and event["type"] == "response.output_text.delta":
                break
        await stream.aclose()
        assert not any(e and e["type"] in {"response.completed", "response.incomplete", "error"} for e in events)
    run(exercise())


def test_concurrent_classification_has_no_shared_parser_state():
    sources = [JAVASCRIPT, SHELL] * 32
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(NORMALIZER.classify_source, sources))
    assert [r[0] for r in results] == ["javascript", "shell"] * 32


@pytest.mark.parametrize("mode", MODES)
def test_reported_trace_regression_from_recovered_fixture(mode):
    fixture = json.loads((Path(__file__).parent / "testdata" / "exec_trace_regression.json").read_text())
    async def exercise():
        chat, state = await make_state()
        upstream = chat_response([call(fixture["tool_name"], "trace_call", json.dumps(fixture["arguments"]))])
        response, events = await convert(mode, upstream, chat, state)
        assert response is not None, events
        assert response["output"][0]["input"] == fixture["arguments"]["input"]
        assert response["output"][0]["call_id"] == "trace_call"
    run(exercise())


LATEST_TRACE_CALLS = json.loads((Path(__file__).parent / "testdata" / "exec_alias_trace_20261023.json").read_text())
LATEST_TRACE_CALLS += json.loads((Path(__file__).parent / "testdata" / "exec_alias_trace_20261026.json").read_text())


@pytest.mark.parametrize("fixture", LATEST_TRACE_CALLS, ids=lambda item: item["call"]["function"]["name"])
@pytest.mark.parametrize("mode", MODES)
def test_latest_trace_alias_calls(mode, fixture):
    async def exercise():
        traced_call = fixture["call"]
        is_wait = traced_call["function"]["name"] == "codex_namespace_functions_wait"
        if is_wait:
            body = _codex_functions_wait_request()
            body["store"] = False
            chat, state = await PROXY.responses_request_to_chat_request(body)
        else:
            chat, state = await make_state()
        upstream = chat_response([copy.deepcopy(traced_call)])
        original = copy.deepcopy(upstream)
        response, events = await convert(mode, upstream, chat, state)
        assert upstream == original
        assert response is not None, events
        item = response["output"][0]
        assert item["call_id"] == traced_call["id"]
        assert item["namespace"] == "functions"
        arguments = json.loads(traced_call["function"]["arguments"])
        if is_wait:
            assert (item["type"], item["name"]) == ("function_call", "wait")
            assert json.loads(item["arguments"]) == arguments
        else:
            assert (item["type"], item["name"]) == ("custom_tool_call", "exec")
            assert item["input"] == arguments["input"]
    run(exercise())


@pytest.mark.parametrize("name", ["codex_debug_ignore", "codex_judge",
    "codex_internal_context", 'codex_internal_context source="goal',
    'codex_internal_context" source="goal',
    'codex</｜DSML｜ parameter>\n<｜DSML｜ parameter name="content" string="true'])
@pytest.mark.parametrize("mode", MODES)
def test_unrelated_or_corrupted_names_do_not_gain_exec_eligibility(name, mode):
    async def exercise():
        chat, state = await make_state()
        upstream = chat_response([call(name, "rejected_id", json.dumps({"input": JAVASCRIPT}))])
        if mode in {"normal", "web"}:
            with pytest.raises(PROXY.ResponsesRequestError) as error:
                await convert(mode, upstream, chat, state)
            assert error.value.code == "undeclared_tool"
        else:
            response, events = await convert(mode, upstream, chat, state)
            assert response is None
            assert events[-1]["response"]["incomplete_details"]["reason"] == "undeclared_tool"
    run(exercise())


@pytest.mark.parametrize("declared,arguments,expected", [
    (False, {"cell_id": "195"}, "undeclared_tool"),
    (True, {"cell_id": 195}, "invalid_function_arguments"),
    (True, {"cell_id": "195", "unknown": True}, "invalid_function_arguments"),
    (True, {"cell_id": "195", "terminate": "false"}, "invalid_function_arguments"),
])
def test_new_wait_alias_retains_declaration_and_parameter_guards(declared, arguments, expected):
    async def exercise():
        if declared:
            chat, state = await PROXY.responses_request_to_chat_request(_codex_functions_wait_request())
        else:
            chat, state = await make_state()
        with pytest.raises(PROXY.ResponsesRequestError) as error:
            await convert("normal", chat_response([call("codex_namespace_functions_wait", "wait_guard",
                json.dumps(arguments))]), chat, state)
        assert error.value.code == expected
    run(exercise())
