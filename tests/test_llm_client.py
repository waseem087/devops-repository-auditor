import anthropic
import httpx2
import pytest
from anthropic.types import Message

import llm_client
from llm_client import (
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMSettings,
    MalformedResponseError,
    load_env_file,
    tool_result_block,
)
from tools import TOOL_SPECS

REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_message(content, stop_reason="tool_use"):
    return Message.model_validate(
        {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "test-model",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 11, "output_tokens": 7},
        }
    )  # fmt: skip


class FakeMessages:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeSDK:
    def __init__(self, outcome):
        self.messages = FakeMessages(outcome)


def client_for(outcome):
    sdk = FakeSDK(outcome)
    return LLMClient(LLMSettings(api_key="sk-test-key-123456"), client=sdk), sdk


def status_error(cls, code):
    return cls("boom sk-ant-api03-supersecretvalue1234567890", response=httpx2.Response(code, request=REQ), body=None)


# ------------------------------------------------------------------ settings
def test_settings_requires_api_key():
    with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY is not set"):
        LLMSettings.from_env({})


def test_settings_rejects_placeholder_key():
    with pytest.raises(LLMConfigError, match="placeholder"):
        LLMSettings.from_env({"ANTHROPIC_API_KEY": "your-api-key-here"})


def test_settings_defaults_and_overrides():
    s = LLMSettings.from_env({"ANTHROPIC_API_KEY": "sk-abc123456789"})
    assert s.model == llm_client.DEFAULT_MODEL and s.max_tokens == 4096
    s2 = LLMSettings.from_env(
        {"ANTHROPIC_API_KEY": "sk-abc123456789", "AUDITOR_MODEL": "other-model", "AUDITOR_LLM_TIMEOUT_S": "5", "AUDITOR_LLM_MAX_RETRIES": "0"}
    )
    assert s2.model == "other-model" and s2.timeout_s == 5.0 and s2.max_retries == 0


def test_settings_bad_numbers_and_key_not_in_repr():
    with pytest.raises(LLMConfigError):
        LLMSettings.from_env({"ANTHROPIC_API_KEY": "sk-abc123456789", "AUDITOR_LLM_MAX_TOKENS": "lots"})
    with pytest.raises(LLMConfigError):
        LLMSettings.from_env({"ANTHROPIC_API_KEY": "sk-abc123456789", "AUDITOR_LLM_TIMEOUT_S": "0"})
    assert "sk-abc123456789" not in repr(LLMSettings.from_env({"ANTHROPIC_API_KEY": "sk-abc123456789"}))


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nFOO_TEST_A=1\nexport FOO_TEST_B='two'\nFOO_TEST_C=\"3\"\nbad line\n")
    monkeypatch.setenv("FOO_TEST_A", "keep")
    monkeypatch.delenv("FOO_TEST_B", raising=False)
    monkeypatch.delenv("FOO_TEST_C", raising=False)
    load_env_file(f)
    import os

    assert os.environ["FOO_TEST_A"] == "keep"
    assert os.environ["FOO_TEST_B"] == "two" and os.environ["FOO_TEST_C"] == "3"
    load_env_file(tmp_path / "missing.env")  # no error


# ------------------------------------------------------------------- parsing
def test_parses_text_and_tool_calls():
    msg = make_message([
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {}},
    ])  # fmt: skip
    client, sdk = client_for(msg)
    r = client.complete("sys", [{"role": "user", "content": "go"}], TOOL_SPECS)
    assert r.text == "Let me look." and r.stop_reason == "tool_use"
    assert [(c.id, c.name, c.input) for c in r.tool_calls] == [("toolu_1", "list_files", {})]
    assert r.assistant_content == [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "list_files", "input": {}},
    ]
    assert (r.input_tokens, r.output_tokens) == (11, 7)
    assert (client.calls, client.total_input_tokens, client.total_output_tokens) == (1, 11, 7)
    sent = sdk.messages.calls[0]
    assert sent["system"] == "sys" and sent["tools"] == TOOL_SPECS and "tool_choice" not in sent


def test_text_only_response_has_no_tool_calls():
    client, _ = client_for(make_message([{"type": "text", "text": "done"}], "end_turn"))
    r = client.complete("s", [], TOOL_SPECS)
    assert r.tool_calls == [] and r.stop_reason == "end_turn"


def test_empty_content_is_returned_not_crashed():
    client, _ = client_for(make_message([], "end_turn"))
    r = client.complete("s", [], TOOL_SPECS)
    assert r.text == "" and r.assistant_content == []


def test_force_tool_sets_tool_choice():
    client, sdk = client_for(make_message([{"type": "text", "text": "x"}]))
    client.complete("s", [], TOOL_SPECS, force_tool="submit_findings")
    assert sdk.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_findings"}


def test_malformed_responses_raise_malformed_error():
    class Bad:
        content = None

    client, _ = client_for(Bad())
    with pytest.raises(MalformedResponseError):
        client.complete("s", [], TOOL_SPECS)

    class Block:
        type = "tool_use"
        id = "t1"
        name = "read_file"
        input = "not a dict"

    class Resp:
        def __init__(self):
            self.content = [Block()]
            self.stop_reason = "tool_use"
            self.usage = None

    client2, _ = client_for(Resp())
    with pytest.raises(MalformedResponseError):
        client2.complete("s", [], TOOL_SPECS)


def test_tool_result_block():
    assert tool_result_block("t1", "ok") == {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
    assert tool_result_block("t1", "bad", True)["is_error"] is True


# ------------------------------------------------------------------- errors
@pytest.mark.parametrize(
    ("exc", "fragment"),
    [
        (status_error(anthropic.AuthenticationError, 401), "Authentication failed"),
        (status_error(anthropic.PermissionDeniedError, 403), "not permitted"),
        (status_error(anthropic.NotFoundError, 404), "AUDITOR_MODEL"),
        (status_error(anthropic.RateLimitError, 429), "Rate limited"),
        (status_error(anthropic.InternalServerError, 500), "HTTP 500"),
        (anthropic.APITimeoutError(request=REQ), "timed out"),
        (anthropic.APIConnectionError(request=REQ), "Could not connect"),
    ],
)
def test_sdk_errors_become_sanitized_llm_errors(exc, fragment):
    client, _ = client_for(exc)
    with pytest.raises(LLMError, match=fragment) as info:
        client.complete("s", [], TOOL_SPECS)
    assert "supersecretvalue" not in str(info.value)


def test_bad_request_message_is_redacted():
    client, _ = client_for(status_error(anthropic.BadRequestError, 400))
    with pytest.raises(LLMError) as info:
        client.complete("s", [], TOOL_SPECS)
    assert "supersecretvalue" not in str(info.value) and "rejected" in str(info.value)