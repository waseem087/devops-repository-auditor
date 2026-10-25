import json

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

from auditor import AgentLimits, AuditAgent
from llm_client import LLMError, MalformedResponseError, tool_result_block
from openai_client import (
    OpenAICompatClient,
    OpenAICompatSettings,
    to_openai_messages,
    to_openai_schema,
    to_openai_tools,
)
from schemas import SUBMIT_TOOL_SPEC
from tools import TOOL_SPECS

REQ = httpx2.Request("POST", "https://example.com/v1/chat/completions")


def settings(**over):
    base = {"provider": "test", "base_url": "https://example.com/v1", "model": "m", "api_key": "k-123456"}
    base.update(over)
    return OpenAICompatSettings(**base)


def completion(message, finish="tool_calls", usage=(7, 3)):
    return ChatCompletion.model_validate(
        {
            "id": "c1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)},
        }
    )  # fmt: skip


def tool_msg(name="list_files", args="{}", call_id="call_1"):
    return {"content": None, "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}]}


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeSDK:
    def __init__(self, *outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = type("Chat", (), {"completions": self.completions})()


def client_for(*outcomes, **over):
    sdk = FakeSDK(*outcomes)
    return OpenAICompatClient(settings(**over), client=sdk, sleep=lambda s: None), sdk


# ---------------------------------------------------------------- schemas
def test_nullable_types_become_single_types_and_optional():
    out = to_openai_schema(SUBMIT_TOOL_SPEC["input_schema"])
    item = out["properties"]["findings"]["items"]
    assert item["properties"]["file"]["type"] == "string" and item["properties"]["line"]["type"] == "integer"
    assert "file" not in item["required"] and "line" not in item["required"] and "rule_id" in item["required"]
    assert "Omit if not applicable" in item["properties"]["file"]["description"]
    assert "additionalProperties" not in json.dumps(out)
    assert item["properties"]["severity"]["enum"] == ["critical", "high", "medium", "low"]


def test_conversion_does_not_mutate_the_original_schema():
    before = json.dumps(SUBMIT_TOOL_SPEC, sort_keys=True)
    to_openai_tools([*TOOL_SPECS, SUBMIT_TOOL_SPEC])
    assert json.dumps(SUBMIT_TOOL_SPEC, sort_keys=True) == before


def test_tool_conversion_shape():
    tools = to_openai_tools(TOOL_SPECS)
    assert [t["function"]["name"] for t in tools] == [s["name"] for s in TOOL_SPECS]
    assert all(t["type"] == "function" and t["function"]["parameters"]["type"] == "object" for t in tools)
    assert tools[1]["function"]["parameters"]["required"] == ["path"]


# --------------------------------------------------------------- messages
def test_message_conversion_round_trip_shape():
    history = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.py"}},
        ]},
        {"role": "user", "content": [
            tool_result_block("t1", '{"ok":true}'),
            {"type": "text", "text": "Please continue."},
        ]},
    ]  # fmt: skip
    out = to_openai_messages("SYS", history)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "start"}
    assert out[2]["content"] == "Looking." and out[2]["tool_calls"][0]["id"] == "t1"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"path": "a.py"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"ok":true}'}
    assert out[4] == {"role": "user", "content": "Please continue."}


def test_assistant_without_text_has_null_content():
    out = to_openai_messages("S", [{"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "list_files", "input": {}}]}])
    assert out[1]["content"] is None and out[1]["tool_calls"][0]["function"]["arguments"] == "{}"


# ---------------------------------------------------------------- parsing
def test_parses_tool_calls_and_usage():
    client, sdk = client_for(completion(tool_msg("read_file", '{"path": "a.py"}')))
    r = client.complete("S", [{"role": "user", "content": "go"}], TOOL_SPECS)
    assert [(c.id, c.name, c.input) for c in r.tool_calls] == [("call_1", "read_file", {"path": "a.py"})]
    assert r.stop_reason == "tool_use" and (r.input_tokens, r.output_tokens) == (7, 3)
    assert r.assistant_content == [{"type": "tool_use", "id": "call_1", "name": "read_file", "input": {"path": "a.py"}}]
    sent = sdk.completions.calls[0]
    assert sent["tool_choice"] == "auto" and sent["model"] == "m" and sent["messages"][0]["role"] == "system"
    assert (client.calls, client.total_input_tokens, client.total_output_tokens) == (1, 7, 3)


def test_text_response_and_finish_reasons():
    client, _ = client_for(completion({"content": "done"}, finish="stop"), completion({"content": "cut"}, finish="length"))
    assert client.complete("S", [], TOOL_SPECS).stop_reason == "end_turn"
    r = client.complete("S", [], TOOL_SPECS)
    assert r.stop_reason == "max_tokens" and r.text == "cut" and r.tool_calls == []


def test_stop_finish_with_tool_calls_is_tool_use():
    client, _ = client_for(completion(tool_msg(), finish="stop"))
    assert client.complete("S", [], TOOL_SPECS).stop_reason == "tool_use"


def test_empty_arguments_and_missing_or_duplicate_ids():
    msg = {"content": None, "tool_calls": [
        {"id": "", "type": "function", "function": {"name": "list_files", "arguments": ""}},
        {"id": "same", "type": "function", "function": {"name": "git_log", "arguments": "{}"}},
        {"id": "same", "type": "function", "function": {"name": "list_files", "arguments": "{}"}},
    ]}  # fmt: skip
    client, _ = client_for(completion(msg))
    r = client.complete("S", [], TOOL_SPECS)
    ids = [c.id for c in r.tool_calls]
    assert len(set(ids)) == 3 and r.tool_calls[0].input == {}


@pytest.mark.parametrize("args", ["{not json", "[1, 2]", '"text"'])
def test_bad_tool_arguments_are_malformed(args):
    client, _ = client_for(completion(tool_msg(args=args)))
    with pytest.raises(MalformedResponseError):
        client.complete("S", [], TOOL_SPECS)


def test_no_choices_is_malformed():
    class Empty:
        def __init__(self):
            self.choices = []

    client, _ = client_for(Empty())
    with pytest.raises(MalformedResponseError):
        client.complete("S", [], TOOL_SPECS)


def test_forced_tool_sends_only_that_tool_as_required():
    client, sdk = client_for(completion(tool_msg("submit_findings", '{"findings": []}')))
    client.complete("S", [], [*TOOL_SPECS, SUBMIT_TOOL_SPEC], force_tool="submit_findings")
    sent = sdk.completions.calls[0]
    assert sent["tool_choice"] == "required"
    assert [t["function"]["name"] for t in sent["tools"]] == ["submit_findings"]


def test_requests_are_paced():
    sleeps, now = [], [100.0]
    sdk = FakeSDK(completion({"content": "a"}, "stop"), completion({"content": "b"}, "stop"))
    client = OpenAICompatClient(settings(min_interval_s=5.0), client=sdk, sleep=sleeps.append, clock=lambda: now[0])
    client.complete("S", [], TOOL_SPECS)
    now[0] = 102.0
    client.complete("S", [], TOOL_SPECS)
    assert sleeps == [pytest.approx(3.0)]


# ----------------------------------------------------------------- errors
def status_error(cls, code, text="boom"):
    return cls(text, response=httpx2.Response(code, request=REQ), body=None)


@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (status_error(openai.AuthenticationError, 401), "Authentication failed"),
        (status_error(openai.PermissionDeniedError, 403), "not permitted"),
        (status_error(openai.NotFoundError, 404), "AUDITOR_MODEL"),
        (status_error(openai.RateLimitError, 429, "quota exceeded"), "free quota exhausted"),
        (status_error(openai.BadRequestError, 400, "invalid schema"), "rejected the request: invalid schema"),
        (status_error(openai.InternalServerError, 500), "HTTP 500"),
        (openai.APITimeoutError(request=REQ), "timed out"),
        (openai.APIConnectionError(request=REQ), "Could not connect"),
    ],
)
def test_errors_become_llm_errors(exc, fragment):
    client, _ = client_for(exc)
    with pytest.raises(LLMError, match=fragment):
        client.complete("S", [], TOOL_SPECS)


def test_secrets_in_provider_errors_are_redacted():
    exc = status_error(openai.BadRequestError, 400, "bad key sk-abcdefghijklmnopqrstuvwxyz123456")
    client, _ = client_for(exc)
    with pytest.raises(LLMError) as info:
        client.complete("S", [], TOOL_SPECS)
    assert "abcdefghijklmnop" not in str(info.value)


def test_api_key_is_not_in_settings_repr():
    assert "k-123456" not in repr(settings())


# -------------------------------------------- full loop through the adapter
def test_agent_runs_end_to_end_through_the_openai_format(repo):
    from config import Limits
    from tools import RepoTools

    submit_args = json.dumps({"findings": [{
        "rule_id": "SEC-001", "category": "security", "severity": "critical", "file": "config/settings.py",
        "line": "1", "description": "Hardcoded API token in source",
        "evidence": "API_TOKEN assigned a literal value", "recommendation": "Use an environment variable",
    }]})  # fmt: skip
    client, sdk = client_for(
        completion(tool_msg("read_file", '{"path": "config/settings.py"}', "call_a")),
        completion(tool_msg("submit_findings", submit_args, "call_b")),
    )
    result = AuditAgent(client, RepoTools(repo, Limits()), "https://github.com/o/r", AgentLimits()).run()
    assert [v.file for v in result.violations] == ["config/settings.py"] and result.violations[0].line == 1
    second_call = sdk.completions.calls[1]["messages"]
    roles = [m["role"] for m in second_call]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second_call[2]["tool_calls"][0]["id"] == "call_a" and second_call[3]["tool_call_id"] == "call_a"
    assert "API_TOKEN" in second_call[3]["content"]