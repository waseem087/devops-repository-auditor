"""The audit agent: an explicit, LLM-driven tool loop written in plain Python.

Observe -> Reason -> Select Tool -> Execute Tool -> Interpret Result -> Continue -> Report

The LLM chooses every tool call. This code only (1) validates and executes the call through the
registered, read-only tools, (2) bounds and sanitizes the result, (3) enforces limits, and
(4) validates the final findings against the evidence the agent actually gathered.
There is no fixed list of checks anywhere in this module.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import PurePosixPath
from typing import Any

from llm_client import (
    LLMClient,
    LLMError,
    LLMResponse,
    MalformedResponseError,
    ToolCall,
    tool_result_block,
)
from prompts import SYSTEM_PROMPT, build_initial_message
from redact import redact_text
from schemas import (
    SUBMIT_TOOL_NAME,
    SUBMIT_TOOL_SPEC,
    Violation,
    dedupe_violations,
    parse_finding,
)
from tools import TOOL_SPECS, RepoTools, _categorize

ELIDED = "[older tool result removed to save context; call the tool again if you need it]"
MAX_FINDINGS_PER_SUBMISSION = 50
MAX_TRACE_ENTRIES = 300

# Rules for which "nothing exists" is a valid claim (file and line may be null). Every other rule
# describes a specific place and must name its file.
ABSENCE_RULES = frozenset({"TST-001", "HYG-001", "HYG-002", "DEP-002", "QUA-001", "COL-001", "BLD-001", "CI-001"})
LOCKFILES = frozenset(
    {
        "poetry.lock", "uv.lock", "pipfile.lock", "package-lock.json", "yarn.lock",
        "pnpm-lock.yaml", "cargo.lock", "gemfile.lock", "composer.lock", "go.sum",
    }
)  # fmt: skip


class AuditError(Exception):
    """The audit could not be completed. The message is safe to show to the user."""


@dataclass(frozen=True)
class AgentLimits:
    max_iterations: int = 30
    max_tool_calls: int = 40
    max_submissions: int = 3
    wall_clock_s: float = 600.0
    max_result_chars: int = 12_000
    history_budget_chars: int = 300_000
    max_malformed: int = 3
    max_nudges: int = 2
    max_unproductive: int = 6
    unread_nudges: int = 1

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AgentLimits:
        env = os.environ if env is None else env
        values: dict[str, Any] = {}
        for f in fields(cls):
            var = f"AUDITOR_{f.name.upper()}"
            raw = env.get(var)
            if not raw:
                continue
            try:
                value = type(f.default)(raw)
            except ValueError as exc:
                raise ValueError(f"{var} must be a number, got {raw!r}") from exc
            if value < 0 or (value == 0 and f.name != "unread_nudges"):
                raise ValueError(f"{var} must be greater than zero, got {raw!r}")
            values[f.name] = value
        return cls(**values)


@dataclass(frozen=True)
class TraceEntry:
    step: int
    tool: str
    args: str
    ok: bool
    summary: str


@dataclass
class AuditResult:
    violations: list[Violation]
    trace: list[TraceEntry] = field(default_factory=list)
    stop_reason: str = "submitted"  # submitted | iteration_limit | tool_limit | time_limit | unproductive | no_tool_use
    iterations: int = 0
    tool_calls: int = 0
    rejected_findings: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def limit_reached(self) -> bool:
        return self.stop_reason != "submitted"


def _summarize(name: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return redact_text(str(result.get("error", "error")))[:120]
    if name == "list_files":
        return f"{result.get('file_count')} files under {result.get('path')}"
    if name == "read_file":
        return f"{result.get('path')} lines {result.get('start_line')}-{result.get('end_line')} of {result.get('total_lines')}"
    if name == "search_code":
        return f"{result.get('match_count')} matches ({result.get('files_scanned')} files scanned)"
    if name == "git_log":
        return f"{len(result.get('commits', []))} commits"
    return "ok"


def _top_level(root: Any) -> list[str]:
    entries: list[str] = []
    try:
        with os.scandir(root) as it:
            for e in sorted(it, key=lambda x: x.name):
                if e.name == ".git":
                    continue
                suffix = "/" if e.is_dir(follow_symlinks=False) else ""
                entries.append(redact_text(e.name[:80]) + suffix)
    except OSError:
        return []
    if len(entries) > 60:
        entries = [*entries[:60], f"... (+{len(entries) - 60} more)"]
    return entries


def _coerce_finding(raw: Any) -> Any:
    """Be liberal in what a model may send (case, "null", numeric strings, omitted null fields)."""
    if not isinstance(raw, dict):
        return raw
    finding = dict(raw)
    for key in ("severity", "category"):
        if isinstance(finding.get(key), str):
            finding[key] = finding[key].strip().lower()
    if isinstance(finding.get("rule_id"), str):
        finding["rule_id"] = finding["rule_id"].strip().upper()
    if isinstance(finding.get("file"), str) and finding["file"].strip().lower() in {"", "null", "none", "n/a"}:
        finding["file"] = None
    finding.setdefault("file", None)
    line = finding.get("line")
    if isinstance(line, str):
        text = line.strip()
        if text.isdigit():
            finding["line"] = int(text)
        elif text.lower() in {"", "null", "none", "n/a"}:
            finding["line"] = None
    elif isinstance(line, float) and line.is_integer():
        finding["line"] = int(line)
    finding.setdefault("line", None)
    return finding


def _append_user_text(messages: list[dict[str, Any]], text: str) -> None:
    """Add user text without breaking the required message structure."""
    if messages and messages[-1]["role"] == "user":
        content = messages[-1]["content"]
        if isinstance(content, list):
            content.append({"type": "text", "text": text})
        else:
            messages[-1]["content"] = [{"type": "text", "text": str(content)}, {"type": "text", "text": text}]
    else:
        messages.append({"role": "user", "content": text})


class AuditAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: RepoTools,
        repository_url: str,
        limits: AgentLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.repository_url = repository_url
        self.limits = limits or AgentLimits()
        self.clock = clock
        self.tool_specs = [*TOOL_SPECS, SUBMIT_TOOL_SPEC]
        self.trace: list[TraceEntry] = []
        self.accepted: list[Violation] = []
        self.rejected_total = 0
        self.tool_calls = 0
        self.successful = 0
        self.submissions = 0
        self.unproductive = 0
        self._seen: dict[tuple[str, str], str] = {}
        self._key_by_id: dict[str, tuple[str, str]] = {}
        self._forced = False
        self._unread_nudges_used = 0

    # ------------------------------------------------------------------ main loop
    def run(self) -> AuditResult:
        lim = self.limits
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": build_initial_message(
                    self.repository_url, _top_level(self.tools.root), lim.max_tool_calls, lim.max_iterations
                ),
            }
        ]
        start = self.clock()
        iterations = nudges = 0
        stop = "submitted"
        finished = False
        while not finished:
            if iterations >= lim.max_iterations:
                stop = "iteration_limit"
            elif self.clock() - start > lim.wall_clock_s:
                stop = "time_limit"
            elif self.tool_calls >= lim.max_tool_calls:
                stop = "tool_limit"
            elif self.unproductive >= lim.max_unproductive:
                stop = "unproductive"
            if stop != "submitted":
                break
            iterations += 1
            self._compact_history(messages)
            response = self._ask(messages)
            if response.assistant_content:
                messages.append({"role": "assistant", "content": response.assistant_content})
            if not response.tool_calls:
                nudges += 1
                if nudges > lim.max_nudges:
                    stop = "no_tool_use"
                    break
                _append_user_text(
                    messages,
                    "Continue the investigation by calling a tool, or call submit_findings if you are done. "
                    "Do not answer in prose.",
                )
                continue
            results = []
            for call in response.tool_calls:
                block, done = self._handle_call(call)
                results.append(block)
                finished = finished or done
            messages.append({"role": "user", "content": results})
        if not finished:
            self._force_submission(messages, stop)
        return AuditResult(
            violations=dedupe_violations(self.accepted),
            trace=self.trace,
            stop_reason=stop,
            iterations=iterations,
            tool_calls=self.tool_calls,
            rejected_findings=self.rejected_total,
            input_tokens=self.llm.total_input_tokens,
            output_tokens=self.llm.total_output_tokens,
        )

    # ------------------------------------------------------------- model access
    def _ask(self, messages: list[dict[str, Any]]) -> LLMResponse:
        for _ in range(self.limits.max_malformed):
            try:
                response = self.llm.complete(SYSTEM_PROMPT, messages, self.tool_specs)
            except MalformedResponseError:
                continue
            except LLMError as exc:
                raise AuditError(str(exc)) from exc
            if response.stop_reason == "max_tokens" and response.tool_calls:
                continue  # the tool call may be cut off; ask again
            return response
        raise AuditError("The model repeatedly produced unusable responses.")

    def _force_submission(self, messages: list[dict[str, Any]], stop: str) -> None:
        if self.successful == 0:
            raise AuditError(f"The audit stopped ({stop}) before any successful inspection, so no report was produced.")
        self._forced = True
        _append_user_text(
            messages,
            "RESOURCE LIMIT REACHED. Call submit_findings now with only the evidence-backed findings you "
            "have already confirmed. Do not call any other tool.",
        )
        try:
            response = self.llm.complete(SYSTEM_PROMPT, messages, self.tool_specs, force_tool=SUBMIT_TOOL_NAME)
        except LLMError as exc:
            raise AuditError(str(exc)) from exc
        call = next((c for c in response.tool_calls if c.name == SUBMIT_TOOL_NAME), None)
        if call is None:
            raise AuditError("The model did not submit findings after the resource limit was reached.")
        self.submissions += 1
        self._process_submission(call.input)

    # --------------------------------------------------------------- tool calls
    def _handle_call(self, call: ToolCall) -> tuple[dict[str, Any], bool]:
        if call.name == SUBMIT_TOOL_NAME:
            return self._handle_submit(call)
        args_text = redact_text(json.dumps(call.input, sort_keys=True, default=str))[:200]
        key = (call.name, json.dumps(call.input, sort_keys=True, default=str))
        if self.tool_calls >= self.limits.max_tool_calls:
            result: dict[str, Any] = {"ok": False, "error": "tool-call budget exhausted; call submit_findings now"}
        elif key in self._seen:
            result = {
                "ok": False,
                "error": "duplicate call: these exact arguments were already used and the result is unchanged. "
                "Use different arguments or a different tool, or call submit_findings.",
            }
            self.unproductive += 1
        else:
            self.tool_calls += 1
            self._seen[key] = call.id
            self._key_by_id[call.id] = key
            result = self.tools.dispatch(call.name, call.input)
            if result.get("ok"):
                self.successful += 1
                self.unproductive = 0
            else:
                self.unproductive += 1
        self._record(call.name, args_text, bool(result.get("ok")), _summarize(call.name, result))
        return tool_result_block(call.id, self._format(result), is_error=not result.get("ok")), False

    def _format(self, result: dict[str, Any]) -> str:
        text = json.dumps(result, ensure_ascii=True, separators=(",", ":"), default=str)
        limit = self.limits.max_result_chars
        if len(text) > limit:
            text = text[:limit] + f" ...[truncated: output exceeded {limit} characters; narrow your request]"
        return text

    def _record(self, tool: str, args: str, ok: bool, summary: str) -> None:
        if len(self.trace) < MAX_TRACE_ENTRIES:
            self.trace.append(TraceEntry(len(self.trace) + 1, tool, args, ok, summary))

    # --------------------------------------------------------------- submission
    def _handle_submit(self, call: ToolCall) -> tuple[dict[str, Any], bool]:
        if self.successful == 0:
            msg = "You have not successfully inspected the repository yet. Use the tools first, then submit."
            self._record(SUBMIT_TOOL_NAME, "", False, "rejected: no inspection yet")
            return tool_result_block(call.id, msg, is_error=True), False
        unread = self._unread_to_nudge()
        if unread:
            self._unread_nudges_used += 1
            self._record(SUBMIT_TOOL_NAME, "", False, f"nudged: {len(unread)} unread notable file(s)")
            names = ", ".join(unread)
            note = (
                f"Before you finish: these notable files exist but you have not opened them: {names}. "
                "CI workflows and dependency files often hold defects. Read the ones that could matter, then "
                "call submit_findings again with ALL your findings (nothing has been recorded yet)."
            )
            return tool_result_block(call.id, note, is_error=True), False
        self.submissions += 1
        text, done, ok = self._process_submission(call.input)
        return tool_result_block(call.id, text, is_error=not ok), done

    def _process_submission(self, raw_input: Any) -> tuple[str, bool, bool]:
        findings = raw_input.get("findings") if isinstance(raw_input, dict) else None
        if not isinstance(findings, list):
            self._record(SUBMIT_TOOL_NAME, "", False, "malformed submission")
            return "submit_findings needs a 'findings' array (it may be empty).", self._forced, False
        accepted: list[Violation] = []
        rejected: list[str] = []
        for index, raw in enumerate(findings[:MAX_FINDINGS_PER_SUBMISSION]):
            violation, reasons = parse_finding(_coerce_finding(raw))
            if violation is not None:
                reasons = self._evidence_problems(violation)
            if reasons:
                rejected.append(f"finding #{index + 1}: " + "; ".join(reasons))
            elif violation is not None:
                accepted.append(violation)
        self.accepted.extend(accepted)
        self.rejected_total += len(rejected)
        self._record(SUBMIT_TOOL_NAME, "", not rejected, f"accepted {len(accepted)}, rejected {len(rejected)}")
        retry_possible = (
            bool(rejected)
            and not self._forced
            and self.submissions < self.limits.max_submissions
            and self.tool_calls < self.limits.max_tool_calls
        )
        if retry_possible:
            details = " | ".join(r[:300] for r in rejected[:8])
            feedback = (
                f"Accepted {len(accepted)} finding(s); do NOT resend them. Rejected {len(rejected)}: {details}. "
                "Fix or drop the rejected ones (read the relevant lines first if needed) and call submit_findings "
                "again with only those, or with an empty list to finish."
            )
            return feedback, False, False
        return f"Recorded {len(accepted)} finding(s); {len(rejected)} rejected. The audit is complete.", True, not rejected

    def _evidence_problems(self, v: Violation) -> list[str]:
        """Evidence gate: a finding must rest on files and lines the agent actually saw."""
        problems: list[str] = []
        if v.file is None:
            if v.rule_id not in ABSENCE_RULES:
                allowed = ", ".join(sorted(ABSENCE_RULES))
                problems.append(
                    f"{v.rule_id} describes a specific place: give its file and line (read the file first). "
                    f"file=null is only for absences, with rule ids {allowed}"
                )
            elif not self.tools.listed_paths:
                problems.append("an absence finding requires list_files first, to show what exists")
            else:
                contradiction = self._contradiction(v.rule_id)
                if contradiction:
                    problems.append(contradiction)
            return problems
        if not self.tools.has_observed(v.file):
            problems.append(f"{v.file} was never read or matched in this audit; read it with read_file first")
        elif v.line is not None and not self.tools.has_observed(v.file, v.line):
            problems.append(f"line {v.line} of {v.file} was not shown to you; read that region first")
        return problems

    # ------------------------------------------------------------ context control
    def _contradiction(self, rule_id: str) -> str | None:
        """Reject an absence claim that the agent's own file listing contradicts."""
        listed = sorted(self.tools.listed_paths)
        hint = ""
        if rule_id == "TST-001":
            found = [p for p in listed if "tests" in _categorize(p)]
            hint = "if tests exist but CI never runs them, report CI-002 on the workflow file instead"
        elif rule_id == "HYG-001":
            found = [p for p in listed if p.lower().startswith("readme") and "/" not in p]
            hint = "judge its content with read_file instead"
        elif rule_id == "HYG-002":
            found = [p for p in listed if p.lower() == ".gitignore"]
            hint = "read it and report a specific gap instead"
        elif rule_id == "CI-001":
            found = [p for p in listed if "workflows" in _categorize(p)]
            hint = "read the workflow and report the specific trigger problem with its file and line"
        elif rule_id == "DEP-002":
            found = [p for p in listed if PurePosixPath(p).name.lower() in LOCKFILES]
            hint = "a lockfile exists"
        else:
            return None
        if not found:
            return None
        return f"{rule_id} claims something is missing, but list_files showed {', '.join(found[:3])}; {hint}"

    def _unread_to_nudge(self) -> list[str]:
        """Notable files the agent listed but never opened (only while a nudge is still available)."""
        if self._forced or self._unread_nudges_used >= self.limits.unread_nudges:
            return []
        unread = []
        for path in sorted(self.tools.listed_paths):
            name = PurePosixPath(path).name.lower()
            cats = _categorize(path)
            manifest = "dependency_manifests" in cats and name not in LOCKFILES
            notable = "workflows" in cats or manifest or name.startswith(("dockerfile", "docker-compose"))
            if notable and path not in self.tools.read_spans:
                unread.append(path)
        return unread[:8]

    def _compact_history(self, messages: list[dict[str, Any]]) -> None:
        """Replace the oldest bulky tool results once the history grows too large."""
        blocks = [
            b
            for m in messages
            if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"]
            if b.get("type") == "tool_result" and isinstance(b.get("content"), str)
        ]
        total = sum(len(b["content"]) for b in blocks)
        for b in blocks:
            if total <= self.limits.history_budget_chars:
                break
            if b["content"] == ELIDED or len(b["content"]) < 300:
                continue
            total -= len(b["content"]) - len(ELIDED)
            b["content"] = ELIDED
            key = self._key_by_id.get(b["tool_use_id"])
            if key is not None:
                self._seen.pop(key, None)  # allowed to run again, since the model no longer sees it