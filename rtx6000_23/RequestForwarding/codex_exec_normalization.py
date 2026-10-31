"""Static, non-executing normalization of Decoder exec compatibility artifacts.

Exact client declarations are handled by the adapter before entering this module.
JSON is an envelope, not an execution language. Ambiguous source is rejected.
"""

import copy
import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any

import jsonschema

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    # Covers jsonschema 3.x and the wrapped reference errors in 4.x without
    # depending on the private exception classes of either implementation.
    RefResolutionError = jsonschema.exceptions.RefResolutionError
    RefResolver = jsonschema.RefResolver


class ExecNormalizationError(ValueError):
    def __init__(self, reason: str, code: str = "invalid_function_arguments"):
        self.reason = reason
        self.code = code
        super().__init__(
            f"Codex exec normalization failed ({reason}). "
            'Use an explicit {"cmd":"..."} command or valid JavaScript for functions.exec.'
        )


class _LocalSchemaResolver(RefResolver):
    def resolve_remote(self, uri):
        # A nested schema id can change a fragment-only $ref's base URI. Block
        # all remote resolution, not just visibly external $ref strings.
        raise ExecNormalizationError("external_schema_reference")


@dataclass(frozen=True)
class NormalizedExec:
    original_tool_name: str
    resolution_kind: str
    envelope_type: str
    payload_type: str
    classification_reason: str
    target_tool: dict[str, Any]
    normalized_arguments: str
    conversion_type: str
    requires_deferred_output: bool = True


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ExecNormalizationError("duplicate_json_key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ExecNormalizationError("nonstandard_json_number")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ExecNormalizationError("nonstandard_json_number")
    return number


def strict_json_loads(value: str):
    try:
        return json.loads(value, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except (json.JSONDecodeError, ExecNormalizationError):
        raise
    except RecursionError:
        raise ExecNormalizationError("json_nesting_limit") from None
    except ValueError:
        raise ExecNormalizationError("invalid_json_number") from None


def _walk(node):
    pending = [node]
    while pending:
        current = pending.pop()
        yield current
        pending.extend(reversed(current.named_children))


def _parse(language, source: bytes):
    from tree_sitter import Parser

    # Each invocation owns its parser, including the parser's timeout state.
    parser = Parser(language)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        parser.timeout_micros = 50_000
    try:
        tree = parser.parse(source)
    except ValueError as exc:
        raise ExecNormalizationError("parser_timeout") from exc
    if tree is None:
        raise ExecNormalizationError("parser_timeout")
    return tree.root_node


def _languages():
    from tree_sitter import Language
    import tree_sitter_javascript
    import tree_sitter_bash

    return Language(tree_sitter_javascript.language()), Language(tree_sitter_bash.language())


def _root_identifier(node, source):
    while node is not None and node.type in {"member_expression", "subscript_expression"}:
        node = node.child_by_field_name("object")
    if node is not None and node.type == "identifier":
        return source[node.start_byte:node.end_byte].decode("utf-8")
    return None


def _strong_javascript(root, source):
    for node in _walk(root):
        if node.type in {"lexical_declaration", "variable_declaration", "await_expression"}:
            return True
        if node.type == "call_expression" and _root_identifier(
            node.child_by_field_name("function"), source
        ) in {"tools", "text", "image", "audio"}:
            return True
    return False


def _shell_evidence(root):
    for node in _walk(root):
        if node.type in {
            "pipeline", "redirected_statement", "command_substitution",
            "if_statement", "for_statement", "while_statement", "case_statement",
            "c_style_for_statement",
        }:
            return True
        if node.type == "command":
            if node.children_by_field_name("argument"):
                return True
            if any(child.type == "variable_assignment" for child in node.named_children):
                return True
    return False


def classify_source(value: str):
    """Return language and evidence without executing or rewriting source."""
    if value.lstrip().startswith("```"):
        raise ExecNormalizationError("unsupported_code_fence")
    source = value.encode("utf-8")
    js_language, bash_language = _languages()
    javascript = _parse(js_language, source)
    bash = _parse(bash_language, source)
    js_ok = not javascript.has_error
    bash_ok = not bash.has_error
    if js_ok and (not bash_ok or _strong_javascript(javascript, source)):
        return "javascript", "javascript_only" if not bash_ok else "javascript_ast"
    if js_ok and bash_ok:
        raise ExecNormalizationError("ambiguous_language")
    if not js_ok:
        # Use the first real Bash word, not text appearing in quotes/comments.
        first_command = next((n for n in _walk(bash) if n.type == "command_name"), None)
        first = (source[first_command.start_byte:first_command.end_byte].decode("utf-8")
                 if first_command is not None else "")
        first_js_node = next((n for n in javascript.named_children if n.type != "comment"), None)
        first_js_text = (source[first_js_node.start_byte:].decode("utf-8")
                         if first_js_node is not None else "")
        if (first in {"const", "let", "var", "await", "import", "export"} or re.match(
            r"^(?:tools|text|image|audio)[.\[(]", first
        ) or re.match(r"^(?:(?:const|let|var|await|import|export)\b|(?:tools|text|image|audio)[.\[(])", first_js_text)):
            raise ExecNormalizationError("invalid_javascript")
        if bash_ok and _shell_evidence(bash):
            return "shell", "shell_ast"
    raise ExecNormalizationError("invalid_source" if not bash_ok else "ambiguous_language")


def validate_parsers():
    """Fail startup before creating any backend clients if grammars are unavailable."""
    try:
        if (classify_source("const r = await tools.exec_command({cmd: `pwd`}); text(r);")[0] != "javascript"
                or classify_source("printf '%s\\n' hello | cat")[0] != "shell"):
            raise RuntimeError("Static grammar sample mismatch")
    except Exception as exc:
        raise RuntimeError("Codex exec static parsers are unavailable or failed their startup self-test.") from exc


def _extract(arguments):
    if isinstance(arguments, str):
        try:
            parsed = strict_json_loads(arguments)
        except json.JSONDecodeError:
            # A broken object/array/quoted envelope must not become a shell command.
            if arguments.lstrip().startswith(("{", "[", '"')):
                raise ExecNormalizationError("invalid_json_envelope") from None
            return "raw_string", arguments
    else:
        parsed = arguments
        try:
            json.dumps(parsed, allow_nan=False)
        except (ValueError, TypeError, RecursionError):
            raise ExecNormalizationError("unsupported_json_shape") from None
    if not isinstance(parsed, dict):
        raise ExecNormalizationError("unsupported_json_shape")
    if "cmd" in parsed:
        if "input" in parsed:
            raise ExecNormalizationError("conflicting_envelope")
        return "cmd_object", parsed
    if set(parsed) != {"input"} or not isinstance(parsed["input"], str):
        raise ExecNormalizationError("unsupported_json_shape")
    value = parsed["input"]
    try:
        inner = strict_json_loads(value)
    except json.JSONDecodeError:
        return "input", value
    if not isinstance(inner, dict) or "cmd" not in inner or "input" in inner:
        raise ExecNormalizationError("unsupported_json_shape")
    return "input_command_json", inner


def _check_schema(command, schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict):
        raise ExecNormalizationError("target_schema_missing", "undeclared_tool")
    schema = copy.deepcopy(schema)
    schema["additionalProperties"] = False
    pending = [schema]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            if "$ref" in node and not str(node["$ref"]).startswith("#"):
                raise ExecNormalizationError("external_schema_reference")
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)
    try:
        jsonschema.Draft4Validator.check_schema(schema)
        jsonschema.Draft4Validator(schema, resolver=_LocalSchemaResolver.from_schema(schema)).validate(command)
    except (jsonschema.SchemaError, jsonschema.ValidationError, RefResolutionError, RecursionError):
        raise ExecNormalizationError("command_schema_mismatch") from None


def normalize_exec(tool_name, arguments, call_id, registry, resolution_kind,
                   command_schema, protocol_markers):
    context = {"envelope_type": "unknown", "payload_type": "unknown"}
    try:
        return _normalize_exec(tool_name, arguments, call_id, registry, resolution_kind,
                               command_schema, protocol_markers, context)
    except ExecNormalizationError as exc:
        exc.envelope_type = context["envelope_type"]
        exc.payload_type = context["payload_type"]
        raise


def _normalize_exec(tool_name, arguments, call_id, registry, resolution_kind,
                    command_schema, protocol_markers, context):
    if not isinstance(call_id, str) or not call_id:
        raise ExecNormalizationError("missing_call_id", "incomplete_function_call")
    if resolution_kind == "unresolved_exec":
        raise ExecNormalizationError("target_ambiguous_or_missing", "undeclared_tool")
    envelope, payload = _extract(arguments)
    context["envelope_type"] = envelope
    serialized = json.dumps(payload, ensure_ascii=False) if isinstance(payload, dict) else payload
    if not isinstance(serialized, str) or not serialized.strip():
        raise ExecNormalizationError("empty_input")
    if any(marker.lower() in serialized.lower() for marker in protocol_markers):
        raise ExecNormalizationError("protocol_markup")
    if isinstance(payload, dict):
        payload_type, evidence = "command_json", "explicit_cmd"
        command = payload
    else:
        payload_type, evidence = classify_source(payload)
        command = {"cmd": payload} if payload_type == "shell" else None
    context["payload_type"] = payload_type

    exec_targets = [s for s in registry.values() if isinstance(s, dict)
                    and s.get("namespace") == "functions" and s.get("source_type") == "custom"
                    and s.get("source_name") == "exec"]
    direct_targets = [s for s in registry.values() if isinstance(s, dict)
                      and s.get("source_name") == "exec_command"
                      and s.get("namespace") in {None, "functions"}
                      and s.get("source_type") in {"function", "namespace"}]
    if payload_type == "javascript":
        if len(exec_targets) != 1:
            raise ExecNormalizationError("target_ambiguous_or_missing", "undeclared_tool")
        target, output = exec_targets[0], {"input": payload}
        conversion = "javascript"
    else:
        if not isinstance(command.get("cmd"), str) or not command["cmd"].strip():
            raise ExecNormalizationError("empty_command")
        if len(direct_targets) > 1:
            raise ExecNormalizationError("target_ambiguous", "undeclared_tool")
        if direct_targets:
            target = direct_targets[0]
            _check_schema(command, target.get("parameters"))
            output = command
        else:
            if len(exec_targets) != 1:
                raise ExecNormalizationError("target_ambiguous_or_missing", "undeclared_tool")
            target = exec_targets[0]
            if "exec_command(args: {" not in (target.get("description") or ""):
                raise ExecNormalizationError("nested_command_not_declared", "undeclared_tool")
            _check_schema(command, command_schema)
            output = {"input": "text(await tools.exec_command(" + json.dumps(command, ensure_ascii=False) + "));"}
        conversion = payload_type
    return NormalizedExec(tool_name, resolution_kind, envelope, payload_type, evidence,
                          copy.deepcopy(target), json.dumps(output, ensure_ascii=False), conversion)
