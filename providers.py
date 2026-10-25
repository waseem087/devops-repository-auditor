"""Choose and build the LLM provider from environment variables.

AUDITOR_PROVIDER = anthropic | gemini | groq | openrouter | openai_compatible
If it is not set, the first provider whose API key is present is used (in that order).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from llm_client import LLMClient, LLMConfigError, LLMSettings
from openai_client import OpenAICompatClient, OpenAICompatSettings


@dataclass(frozen=True)
class Preset:
    key_env: str
    base_url: str
    model: str
    min_interval_s: float


PRESETS = {
    "gemini": Preset("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash-lite", 5.0),
    "groq": Preset("GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b", 3.0),
    "openrouter": Preset("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openai/gpt-oss-120b:free", 4.0),
}
DETECTION_ORDER = ("anthropic", "gemini", "groq", "openrouter")
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", **{name: p.key_env for name, p in PRESETS.items()}}
VALID = (*DETECTION_ORDER, "openai_compatible")


def usable_key(value: str | None) -> bool:
    """True for a real-looking key (not empty, not a placeholder copied from .env.example)."""
    v = (value or "").strip()
    return bool(v) and not v.lower().startswith(("your", "<", "changeme"))


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    settings: LLMSettings | OpenAICompatSettings

    def build_client(self) -> Any:
        if isinstance(self.settings, LLMSettings):
            return LLMClient(self.settings)
        return OpenAICompatClient(self.settings)


def _number(env: Mapping[str, str], var: str, kind: type, minimum: float = 0.0) -> Any:
    raw = env.get(var)
    if not raw:
        return None
    try:
        value = kind(raw)
    except ValueError as exc:
        raise LLMConfigError(f"{var} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise LLMConfigError(f"{var} must be at least {minimum}, got {raw!r}")
    return value


def load_provider(env: Mapping[str, str] | None = None) -> ProviderConfig:
    env = os.environ if env is None else env
    explicit = (env.get("AUDITOR_PROVIDER") or "").strip().lower()
    if explicit and explicit not in VALID:
        raise LLMConfigError(f"AUDITOR_PROVIDER must be one of {', '.join(VALID)}; got {explicit!r}")
    provider = explicit or next((p for p in DETECTION_ORDER if usable_key(env.get(KEY_ENV[p]))), "")
    if not provider:
        names = ", ".join(KEY_ENV.values())
        raise LLMConfigError(
            f"No LLM API key found. Set one of: {names} (see .env.example). "
            "A free option is GEMINI_API_KEY from https://aistudio.google.com/apikey."
        )
    if provider == "anthropic":
        settings = LLMSettings.from_env(env)
        return ProviderConfig("anthropic", settings.model, settings)

    if provider == "openai_compatible":
        key, base_url = env.get("AUDITOR_API_KEY", ""), (env.get("AUDITOR_BASE_URL") or "").strip()
        model = (env.get("AUDITOR_MODEL") or "").strip()
        if not (usable_key(key) and base_url and model):
            raise LLMConfigError("openai_compatible needs AUDITOR_API_KEY, AUDITOR_BASE_URL and AUDITOR_MODEL.")
        min_interval = 0.0
    else:
        preset = PRESETS[provider]
        key = env.get(preset.key_env, "")
        if not usable_key(key):
            raise LLMConfigError(f"{preset.key_env} is not set. Put your {provider} key in .env (see .env.example).")
        base_url = (env.get("AUDITOR_BASE_URL") or "").strip() or preset.base_url
        model = (env.get("AUDITOR_MODEL") or "").strip() or preset.model
        min_interval = preset.min_interval_s
    if not base_url.startswith(("https://", "http://localhost", "http://127.0.0.1")):
        raise LLMConfigError("AUDITOR_BASE_URL must start with https:// (or http://localhost for a local server).")

    values: dict[str, Any] = {"min_interval_s": min_interval}
    for var, name, kind, minimum in (
        ("AUDITOR_LLM_MAX_TOKENS", "max_tokens", int, 1),
        ("AUDITOR_LLM_TIMEOUT_S", "timeout_s", float, 1),
        ("AUDITOR_LLM_MAX_RETRIES", "max_retries", int, 0),
        ("AUDITOR_MIN_INTERVAL_S", "min_interval_s", float, 0),
    ):
        number = _number(env, var, kind, minimum)
        if number is not None:
            values[name] = number
    settings = OpenAICompatSettings(provider=provider, base_url=base_url, model=model, api_key=key.strip(), **values)
    return ProviderConfig(provider, model, settings)