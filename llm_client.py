"""Thin wrapper around the Anthropic Messages API (native tool use).

This module only talks to the model. It never runs tools and never decides what to
investigate; that is the agent loop's job. It converts SDK objects into plain
dataclasses, maps every SDK error to a sanitized LLMError, and never puts the API
key (or request bodies) into an error message.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

from redact import redact_text

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(Exception):
    """A model/API problem. The message is safe to print."""


class LLMConfigError(LLMError):
    """Missing or invalid configuration (for example no API key)."""


class MalformedResponseError(LLMError):
    """The API answered, but the response is not something the agent can use."""


def load_env_file(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from a .env file without overriding real environment variables."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class LLMSettings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    max_tokens: int = 4096
    timeout_s: float = 90.0
    max_retries: int = 3

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMSettings:
        env = os.environ if env is None else env
        key = (env.get("ANTHROPIC_API_KEY") or "").strip()
        if not key:
            raise LLMConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and put your key there, "
                "or set the environment variable."
            )
        if key.lower().startswith(("your", "<", "changeme")) or any(c.isspace() for c in key) or not key.isascii():
            raise LLMConfigError("ANTHROPIC_API_KEY looks like a placeholder; set your real API key.")
        values: dict[str, Any] = {}
        for var, name, kind in (
            ("AUDITOR_LLM_MAX_TOKENS", "max_tokens", int),
            ("AUDITOR_LLM_TIMEOUT_S", "timeout_s", float),
            ("AUDITOR_LLM_MAX_RETRIES", "max_retries", int),
        ):
            raw = env.get(var)
            if raw:
                try:
                    number = kind(raw)
                except ValueError as exc:
                    raise LLMConfigError(f"{var} must be a number, got {raw!r}") from exc
                if number < 0 or (number == 0 and name != "max_retries"):
                    raise LLMConfigError(f"{var} must be positive, got {raw!r}")
                values[name] = number
        model = (env.get("AUDITOR_MODEL") or "").strip() or DEFAULT_MODEL
        return cls(api_key=key, model=model, **values)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str | None
    assistant_content: list[dict[str, Any]]  # append this to the message history as-is
    input_tokens: int = 0
    output_tokens: int = 0


def tool_result_block(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    """Build one tool_result block (goes into a *user* message after the model's tool_use)."""
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


class LLMClient:
    def __init__(self, settings: LLMSettings, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client or anthropic.Anthropic(
            api_key=settings.api_key,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> LLMResponse:
        """One model turn. `force_tool` makes the model call that specific tool."""
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": messages,
            "tools": tools,
        }
        if force_tool:
            kwargs["tool_choice"] = {"type": "tool", "name": force_tool}
        self.calls += 1
        try:
            raw = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as exc:
            raise LLMError("Authentication failed. Check that ANTHROPIC_API_KEY is valid.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("The API key is not permitted to use this model or endpoint.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model or endpoint not found. Check AUDITOR_MODEL (currently {self.settings.model!r}).") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the API even after retries. Wait a few minutes and retry.") from exc
        except anthropic.APITimeoutError as exc:
            raise LLMError(f"The API request timed out after {self.settings.timeout_s:.0f}s (with retries).") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not connect to the Anthropic API. Check your network connection.") from exc
        except anthropic.BadRequestError as exc:
            raise LLMError("The API rejected the request: " + redact_text(str(exc))[:200]) from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The API returned an error (HTTP {exc.status_code}).") from exc
        except anthropic.APIError as exc:
            raise LLMError(f"Unexpected API error ({type(exc).__name__}).") from exc
        return self._parse(raw)

    def _parse(self, raw: Any) -> LLMResponse:
        content = getattr(raw, "content", None)
        if not isinstance(content, list):
            raise MalformedResponseError("The model response had no content list.")
        texts: list[str] = []
        calls: list[ToolCall] = []
        blocks: list[dict[str, Any]] = []
        for block in content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text = getattr(block, "text", "")
                if isinstance(text, str) and text:
                    texts.append(text)
                    blocks.append({"type": "text", "text": text})
            elif kind == "tool_use":
                call_id, name, args = getattr(block, "id", None), getattr(block, "name", None), getattr(block, "input", None)
                if not isinstance(call_id, str) or not isinstance(name, str) or not isinstance(args, dict):
                    raise MalformedResponseError("The model produced a malformed tool call.")
                calls.append(ToolCall(call_id, name, args))
                blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": args})
            # other block types (thinking, etc.) are ignored
        usage = getattr(raw, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        stop = getattr(raw, "stop_reason", None)
        return LLMResponse(
            text="\n".join(texts),
            tool_calls=calls,
            stop_reason=stop if isinstance(stop, str) else None,
            assistant_content=blocks,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )