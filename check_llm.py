"""Quick diagnostic for your LLM provider: python check_llm.py

Makes four tiny requests to confirm that the key works, the model supports tool calling, a tool
result can be sent back, and the agent's real tool schemas are accepted. Run it before a full audit.
"""

from __future__ import annotations

import sys
import time
from typing import Any

from llm_client import LLMError, load_env_file, tool_result_block
from prompts import SYSTEM_PROMPT
from providers import load_provider
from schemas import SUBMIT_TOOL_NAME, SUBMIT_TOOL_SPEC
from tools import TOOL_SPECS

PING = {
    "name": "ping",
    "description": "Echo a short message back.",
    "input_schema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
        "additionalProperties": False,
    },
}


def run_checks(client: Any) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))

    ask = [{"role": "user", "content": "Call the ping tool with the message 'hello'. Write nothing else."}]
    first = None
    try:
        first = client.complete("You are a test assistant. Use tools when asked.", ask, [PING])
        called = [c for c in first.tool_calls if c.name == "ping"]
        record("tool call", bool(called), "the model called ping" if called else "the model did not call the tool")
    except LLMError as exc:
        record("tool call", False, str(exc))

    try:
        if first is None or not first.tool_calls:
            raise LLMError("skipped: the first check produced no tool call")
        history = [
            *ask,
            {"role": "assistant", "content": first.assistant_content},
            {"role": "user", "content": [tool_result_block(first.tool_calls[0].id, '{"echo":"hello"}')]},
        ]
        client.complete("You are a test assistant. Use tools when asked.", history, [PING])
        record("tool result round trip", True, "the provider accepted the tool result")
    except LLMError as exc:
        record("tool result round trip", False, str(exc))

    specs = [*TOOL_SPECS, SUBMIT_TOOL_SPEC]
    try:
        reply = client.complete(SYSTEM_PROMPT, [{"role": "user", "content": "Call list_files now."}], specs)
        ok = any(c.name == "list_files" for c in reply.tool_calls)
        record("real tool schemas", ok, "accepted; list_files was called" if ok else "accepted, but list_files was not called")
    except LLMError as exc:
        record("real tool schemas", False, str(exc))

    try:
        reply = client.complete(
            SYSTEM_PROMPT, [{"role": "user", "content": "Submit an empty list of findings."}], specs, force_tool=SUBMIT_TOOL_NAME
        )
        call = next((c for c in reply.tool_calls if c.name == SUBMIT_TOOL_NAME), None)
        ok = call is not None and isinstance(call.input.get("findings", []), list)
        record("forced submit_findings", ok, "forced tool choice works" if ok else "forced tool choice was ignored")
    except LLMError as exc:
        record("forced submit_findings", False, str(exc))
    return results


def main() -> int:
    load_env_file(".env")
    try:
        provider = load_provider()
    except (LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Checking provider '{provider.provider}' with model '{provider.model}' (4 small requests)...")
    started = time.monotonic()
    results = run_checks(provider.build_client())
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    passed = all(ok for _, ok, _ in results)
    print(("All checks passed." if passed else "Some checks failed; see above.") + f" ({time.monotonic() - started:.0f}s)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())