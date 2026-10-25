"""Client for any OpenAI-compatible Chat Completions API (Gemini, Groq, OpenRouter, local servers).

It offers exactly the same `complete(system, messages, tools, force_tool)` interface as
llm_client.LLMClient, so the agent loop does not know or care which provider is behind it. The agent
keeps its history in one neutral format; this module converts it for the provider.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import openai

from llm_client import LLMError, LLMResponse, MalformedResponseError, ToolCall
from redact import redact_text

_DROP_KEYS = {"additionalProperties", "$schema"}


def to_openai_schema(schema: Any) -> Any:
    """Simplify a JSON schema for stricter providers: single types, no additionalProperties.

    A property typed ["string", "null"] becomes "string" and is no longer required; the agent treats a
    missing value as null.
    """
    if isinstance(schema, list):
        return [to_openai_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    optional: set[str] = set()
    for key, value in schema.items():
        if key in _DROP_KEYS:
            continue
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            out["type"] = non_null[0] if non_null else "string"
        elif key == "properties" and isinstance(value, dict):
            props: dict[str, Any] = {}
            for name, sub in value.items():
                cleaned = to_openai_schema(sub)
                if isinstance(sub, dict) and isinstance(sub.get("type"), list) and "null" in sub["type"]:
                    optional.add(name)
                    cleaned["description"] = (cleaned.get("description", "") + " Omit if not applicable.").strip()
                props[name] = cleaned
            out["properties"] = props
        elif isinstance(value, (dict, list)):
            out[key] = to_openai_schema(value)
        else:
            out[key] = value
    if optional and isinstance(out.get("required"), list):
        out["required"] = [name for name in out["required"] if name not in optional]
        if not out["required"]:
            del out["required"]
    return out


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": to_openai_schema(t["input_schema"]),
            },
        }
        for t in tools
    ]


def to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the agent's history (text / tool_use / tool_result blocks) to chat-completions messages."""
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for message in messages:
        role, content = message["role"], message["content"]
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif role == "assistant":
            texts = [b["text"] for b in content if b.get("type") == "text"]
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        else:
            extra: list[str] = []
            for block in content:
                if block.get("type") == "tool_result":
                    out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": str(block.get("content", ""))})
                elif block.get("type") == "text":
                    extra.append(block["text"])
            if extra:
                out.append({"role": "user", "content": "\n".join(extra)})
    return out


@dataclass(frozen=True)
class OpenAICompatSettings:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    max_tokens: int = 4096
    timeout_s: float = 120.0
    max_retries: int = 5
    min_interval_s: float = 0.0  # pause between requests, to stay under free-tier rate limits


_FINISH = {"tool_calls": "tool_use", "stop": "end_turn", "length": "max_tokens"}


class OpenAICompatClient:
    def __init__(
        self,
        settings: OpenAICompatSettings,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._client = client or openai.OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=settings.max_retries,
        )
        self._sleep, self._clock = sleep, clock
        self._last_request: float | None = None
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _pace(self) -> None:
        if self._last_request is not None and self.settings.min_interval_s > 0:
            wait = self._last_request + self.settings.min_interval_s - self._clock()
            if wait > 0:
                self._sleep(wait)

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        force_tool: str | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if force_tool:
            kwargs["tools"] = to_openai_tools([t for t in tools if t["name"] == force_tool])
            kwargs["tool_choice"] = "required"
        else:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        self._pace()
        self.calls += 1
        try:
            raw = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as exc:
            raise LLMError("Authentication failed. Check your API key. Detail: " + redact_text(str(exc))[:300]) from exc
        except openai.PermissionDeniedError as exc:
            raise LLMError("The API key is not permitted to use this model or endpoint. Detail: " + redact_text(str(exc))[:300]) from exc
        except openai.NotFoundError as exc:
            raise LLMError(
                f"Model or endpoint not found. Check AUDITOR_MODEL (currently {self.settings.model!r}). "
                "Detail: " + redact_text(str(exc))[:300]
            ) from exc
        except openai.RateLimitError as exc:
            raise LLMError(
                "Rate limited or free quota exhausted, even after retries. Wait and retry later, "
                "or choose another model/provider. Detail: " + redact_text(str(exc))[:200]
            ) from exc
        except openai.APITimeoutError as exc:
            raise LLMError(f"The API request timed out after {self.settings.timeout_s:.0f}s (with retries).") from exc
        except openai.APIConnectionError as exc:
            raise LLMError("Could not connect to the provider. Check your network and AUDITOR_BASE_URL.") from exc
        except openai.BadRequestError as exc:
            raise LLMError("The provider rejected the request: " + redact_text(str(exc))[:300]) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"The provider returned an error (HTTP {exc.status_code}).") from exc
        except openai.APIError as exc:
            raise LLMError(f"Unexpected provider error ({type(exc).__name__}).") from exc
        finally:
            self._last_request = self._clock()
        return self._parse(raw)

    def _parse(self, raw: Any) -> LLMResponse:
        choices = getattr(raw, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise MalformedResponseError("The provider response contained no choices.")
        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise MalformedResponseError("The provider response contained no message.")
        text = message.content if isinstance(getattr(message, "content", None), str) else ""
        calls: list[ToolCall] = []
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
        for index, item in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(item, "function", None)
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
            if not isinstance(name, str) or not name:
                raise MalformedResponseError("The model produced a malformed tool call.")
            try:
                parsed = json.loads(arguments) if isinstance(arguments, str) and arguments.strip() else {}
            except json.JSONDecodeError as exc:
                raise MalformedResponseError("The model produced tool arguments that are not valid JSON.") from exc
            if not isinstance(parsed, dict):
                raise MalformedResponseError("The model produced tool arguments that are not a JSON object.")
            call_id = getattr(item, "id", None)
            if not isinstance(call_id, str) or not call_id or any(c.id == call_id for c in calls):
                call_id = f"call_{self.calls}_{index}"
            calls.append(ToolCall(call_id, name, parsed))
            blocks.append({"type": "tool_use", "id": call_id, "name": name, "input": parsed})
        usage = getattr(raw, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        finish = getattr(choice, "finish_reason", None)
        stop = _FINISH.get(finish, finish) if isinstance(finish, str) else None
        if calls and stop == "end_turn":
            stop = "tool_use"
        return LLMResponse(text, calls, stop, blocks, in_tok, out_tok)