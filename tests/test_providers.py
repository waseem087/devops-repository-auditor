import pytest

import check_llm
from llm_client import LLMClient, LLMConfigError, LLMResponse, LLMSettings, ToolCall
from openai_client import OpenAICompatClient, OpenAICompatSettings
from providers import PRESETS, load_provider, usable_key

from .test_agent import ScriptedLLM, reply, tc


def test_no_key_gives_a_helpful_message():
    with pytest.raises(LLMConfigError, match="No LLM API key found") as info:
        load_provider({})
    assert "GEMINI_API_KEY" in str(info.value) and "aistudio.google.com" in str(info.value)


def test_placeholder_keys_do_not_count():
    assert not usable_key("your-gemini-key-here") and not usable_key("") and not usable_key(None)
    assert usable_key("AIza-real-looking-key")
    with pytest.raises(LLMConfigError):
        load_provider({"GEMINI_API_KEY": "your-gemini-key-here"})


def test_auto_detection_order():
    both = {"GEMINI_API_KEY": "g-123456", "GROQ_API_KEY": "q-123456"}
    assert load_provider(both).provider == "gemini"
    assert load_provider({"GROQ_API_KEY": "q-123456"}).provider == "groq"
    assert load_provider({**both, "ANTHROPIC_API_KEY": "sk-ant-123456"}).provider == "anthropic"


def test_explicit_provider_wins():
    env = {"AUDITOR_PROVIDER": "groq", "GEMINI_API_KEY": "g-123456", "GROQ_API_KEY": "q-123456"}
    assert load_provider(env).provider == "groq"
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY is not set"):
        load_provider({"AUDITOR_PROVIDER": "groq", "GEMINI_API_KEY": "g-123456"})


def test_unknown_provider_rejected():
    with pytest.raises(LLMConfigError, match="AUDITOR_PROVIDER must be one of"):
        load_provider({"AUDITOR_PROVIDER": "skynet", "GEMINI_API_KEY": "g-123456"})


def test_gemini_preset_values_and_overrides():
    cfg = load_provider({"GEMINI_API_KEY": "g-123456"})
    s = cfg.settings
    assert isinstance(s, OpenAICompatSettings) and s.base_url == PRESETS["gemini"].base_url
    assert cfg.model == PRESETS["gemini"].model and s.min_interval_s == PRESETS["gemini"].min_interval_s
    assert s.base_url.startswith("https://generativelanguage.googleapis.com/")
    cfg2 = load_provider({"GEMINI_API_KEY": "g-123456", "AUDITOR_MODEL": "gemma-4-31b-it", "AUDITOR_MIN_INTERVAL_S": "0", "AUDITOR_LLM_MAX_RETRIES": "2"})
    assert cfg2.model == "gemma-4-31b-it" and cfg2.settings.min_interval_s == 0.0 and cfg2.settings.max_retries == 2


def test_api_key_never_in_repr():
    assert "g-123456" not in repr(load_provider({"GEMINI_API_KEY": "g-123456"}))


@pytest.mark.parametrize("bad", [{"AUDITOR_LLM_MAX_TOKENS": "lots"}, {"AUDITOR_LLM_TIMEOUT_S": "0"}, {"AUDITOR_MIN_INTERVAL_S": "-1"}])
def test_bad_numbers_rejected(bad):
    with pytest.raises(LLMConfigError):
        load_provider({"GEMINI_API_KEY": "g-123456", **bad})


@pytest.mark.parametrize("url", ["http://evil.example.com/v1", "ftp://x", "example.com"])
def test_insecure_base_urls_rejected(url):
    with pytest.raises(LLMConfigError, match="https://"):
        load_provider({"GEMINI_API_KEY": "g-123456", "AUDITOR_BASE_URL": url})


def test_local_server_allowed_and_custom_provider():
    ok = load_provider({"AUDITOR_PROVIDER": "openai_compatible", "AUDITOR_API_KEY": "local-key-1", "AUDITOR_BASE_URL": "http://localhost:11434/v1", "AUDITOR_MODEL": "qwen"})
    assert ok.provider == "openai_compatible" and ok.settings.base_url.startswith("http://localhost")
    with pytest.raises(LLMConfigError, match="needs AUDITOR_API_KEY"):
        load_provider({"AUDITOR_PROVIDER": "openai_compatible"})


def test_anthropic_path_and_client_types():
    cfg = load_provider({"ANTHROPIC_API_KEY": "sk-ant-123456"})
    assert isinstance(cfg.settings, LLMSettings) and isinstance(cfg.build_client(), LLMClient)
    assert isinstance(load_provider({"GEMINI_API_KEY": "g-123456"}).build_client(), OpenAICompatClient)


# ----------------------------------------------------------- check_llm.py
class Good:
    def complete(self, system, messages, tools, force_tool=None):
        if force_tool:
            return LLMResponse("", [ToolCall("c", force_tool, {"findings": []})], "tool_use", [])
        names = [t["name"] for t in tools]
        name = "ping" if names == ["ping"] else "list_files"
        call = ToolCall("c1", name, {"message": "hello"} if name == "ping" else {})
        return LLMResponse("", [call], "tool_use", [{"type": "tool_use", "id": "c1", "name": name, "input": call.input}])


def test_check_llm_all_pass():
    results = check_llm.run_checks(Good())
    assert [ok for _, ok, _ in results] == [True, True, True, True]


def test_check_llm_reports_failures_without_crashing():
    from llm_client import LLMError

    class Broken:
        def complete(self, *a, **k):
            raise LLMError("The provider rejected the request: bad schema")

    results = check_llm.run_checks(Broken())
    assert [ok for _, ok, _ in results] == [False, False, False, False]
    assert "bad schema" in results[2][2]


def test_check_llm_main_without_key(monkeypatch, tmp_path, capsys):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "AUDITOR_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert check_llm.main() == 2
    assert "No LLM API key found" in capsys.readouterr().err


def test_check_llm_main_runs_checks(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    class Cfg:
        provider, model = "fake", "m1"

        def build_client(self):
            return Good()

    monkeypatch.setattr(check_llm, "load_provider", lambda: Cfg())
    assert check_llm.main() == 0
    out = capsys.readouterr().out
    assert "[PASS] tool call" in out and "All checks passed" in out


def test_scripted_helpers_are_importable():
    assert ScriptedLLM([reply(tc("list_files"))]).steps