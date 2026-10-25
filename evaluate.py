"""Measure detection quality on the local synthetic fixtures (precision, recall, F1).

    python evaluate.py                 # run every fixture with the real model (needs an LLM API key)
    python evaluate.py --only healthy  # run selected fixtures
    python evaluate.py --list          # show fixtures and their expected defects

IMPORTANT: these are the project author's own labeled fixtures. They measure local behaviour only
and say nothing certain about the instructor's hidden repositories.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from auditor import AgentLimits, AuditAgent, AuditError
from config import Limits
from llm_client import LLMError, load_env_file
from providers import load_provider
from repo_access import safe_rmtree
from schemas import Violation
from tests.fixture_repos import FIXTURES, Fixture
from tools import RepoTools

LINE_TOLERANCE = 5
FIXTURE_URL = "https://github.com/fixture/{name}"


@dataclass
class FixtureResult:
    name: str
    expected: list[dict[str, Any]]
    predicted: list[Violation] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    error: str | None = None
    stop_reason: str = ""
    tool_calls: int = 0
    tokens: int = 0
    false_positives: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    trace: list[Any] = field(default_factory=list)


def _matches(exp: dict[str, Any], pred: Violation, tolerance: int) -> bool:
    if pred.file != exp["file"] or pred.category not in exp["categories"]:
        return False
    if exp["line"] is None or pred.line is None:
        return True
    return abs(pred.line - exp["line"]) <= tolerance


def match_findings(
    expected: list[dict[str, Any]], predicted: list[Violation], tolerance: int = LINE_TOLERANCE
) -> tuple[int, list[int], list[int]]:
    """One-to-one greedy matching. Returns (true_positives, unmatched_predicted, unmatched_expected)."""
    used: set[int] = set()
    missed: list[int] = []
    tp = 0
    for ei, exp in enumerate(expected):
        hit = next((pi for pi, pred in enumerate(predicted) if pi not in used and _matches(exp, pred, tolerance)), None)
        if hit is None:
            missed.append(ei)
        else:
            used.add(hit)
            tp += 1
    extra = [pi for pi in range(len(predicted)) if pi not in used]
    return tp, extra, missed


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1. With nothing expected and nothing predicted, all three are 1.0."""
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def summarize(results: list[FixtureResult]) -> dict[str, Any]:
    """Micro-averaged scores over every completed fixture, plus a count of failed runs."""
    done = [r for r in results if r.error is None]
    tp, fp = sum(r.tp for r in done), sum(r.fp for r in done)
    # a fixture that crashed contributes its expected defects as misses
    fn = sum(r.fn for r in done) + sum(len(r.expected) for r in results if r.error is not None)
    return {"tp": tp, "fp": fp, "fn": fn, "failed_runs": len(results) - len(done), **prf(tp, fp, fn)}


def prepare_checkout(fixture: Fixture, base_dir: Path) -> Path:
    """Write the fixture to disk (as a git repository when git is available)."""
    root = base_dir / fixture.name
    for rel, text in fixture.files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    try:
        git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.com", "-c", "commit.gpgsign=false"]
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "Initial fixture commit"]):
            subprocess.run([*git, *args], cwd=root, check=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        pass  # git_log will simply report that no history is available
    return root


def run_fixture(
    fixture: Fixture,
    llm_factory: Callable[[], Any],
    work_dir: Path,
    tolerance: int = LINE_TOLERANCE,
    limits: Limits | None = None,
    agent_limits: AgentLimits | None = None,
) -> FixtureResult:
    expected = fixture.resolved_expected()
    result = FixtureResult(fixture.name, expected)
    root = prepare_checkout(fixture, work_dir)
    try:
        llm = llm_factory()
        agent = AuditAgent(llm, RepoTools(root, limits or Limits()), FIXTURE_URL.format(name=fixture.name), agent_limits)
        audit = agent.run()
    except (AuditError, LLMError) as exc:
        result.error = str(exc)
        result.fn = len(expected)
        return result
    result.predicted = audit.violations
    result.stop_reason, result.tool_calls = audit.stop_reason, audit.tool_calls
    result.trace = audit.trace
    result.tokens = audit.input_tokens + audit.output_tokens
    tp, extra, missed = match_findings(expected, audit.violations, tolerance)
    result.tp, result.fp, result.fn = tp, len(extra), len(missed)
    result.false_positives = [f"{audit.violations[i].rule_id} {audit.violations[i].file}:{audit.violations[i].line}" for i in extra]
    result.missed = [f"{expected[i]['categories']} {expected[i]['file']}:{expected[i]['line']}" for i in missed]
    return result


def evaluate(
    fixtures: list[Fixture], llm_factory: Callable[[], Any], tolerance: int = LINE_TOLERANCE, **kwargs: Any
) -> list[FixtureResult]:
    work = Path(tempfile.mkdtemp(prefix="auditor-eval-"))
    try:
        return [run_fixture(f, llm_factory, work, tolerance, **kwargs) for f in fixtures]
    finally:
        safe_rmtree(work)


def format_table(results: list[FixtureResult]) -> str:
    rows = [f"{'fixture':<26}{'expected':>9}{'found':>7}{'TP':>4}{'FP':>4}{'FN':>4}  notes"]
    for r in results:
        notes = r.error or "; ".join([*(f"FP {x}" for x in r.false_positives), *(f"missed {x}" for x in r.missed)])
        rows.append(f"{r.name:<26}{len(r.expected):>9}{len(r.predicted):>7}{r.tp:>4}{r.fp:>4}{r.fn:>4}  {notes}")
    s = summarize(results)
    rows.append("-" * 70)
    rows.append(
        f"TOTAL  TP={s['tp']} FP={s['fp']} FN={s['fn']}  precision={s['precision']:.2f} "
        f"recall={s['recall']:.2f} F1={s['f1']:.2f}  (failed runs: {s['failed_runs']})"
    )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the auditor on local synthetic fixtures.")
    parser.add_argument("--only", action="append", metavar="NAME", help="run only this fixture (repeatable)")
    parser.add_argument("--list", action="store_true", help="list fixtures and exit")
    parser.add_argument("--verbose", action="store_true", help="print the investigation trace of each fixture")
    parser.add_argument("--output", default="eval_results.json", help="where to save results (default: eval_results.json)")
    parser.add_argument("--tolerance", type=int, default=LINE_TOLERANCE, help="allowed line difference when matching")
    args = parser.parse_args(argv)

    if args.list:
        for f in FIXTURES.values():
            print(f"{f.name}: {f.description} -> {len(f.expected)} expected defect(s)")
        return 0
    unknown = [n for n in args.only or [] if n not in FIXTURES]
    if unknown:
        print(f"error: unknown fixture(s): {unknown}; use --list", file=sys.stderr)
        return 2
    selected = [FIXTURES[n] for n in args.only] if args.only else list(FIXTURES.values())

    load_env_file(".env")
    try:
        provider = load_provider()
        agent_limits, limits = AgentLimits.from_env(), Limits.from_env()
    except (ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Evaluating {len(selected)} fixture(s) with {provider.provider}/{provider.model} (this makes real API calls)...")
    results = evaluate(selected, provider.build_client, args.tolerance, limits=limits, agent_limits=agent_limits)
    print(format_table(results))
    if args.verbose:
        for r in results:
            print(f"--- trace: {r.name}")
            for e in r.trace:
                print(f"  {e.step:>3}. {e.tool} {e.args} -> {'ok' if e.ok else 'ERROR'}: {e.summary}")
    print("NOTE: local synthetic fixtures labelled by the project author; not the instructor's hidden evaluation.")
    payload = {
        "kind": "local-synthetic-fixtures",
        "model": f"{provider.provider}/{provider.model}",
        "summary": summarize(results),
        "fixtures": [
            {
                "name": r.name, "expected": r.expected, "tp": r.tp, "fp": r.fp, "fn": r.fn, "error": r.error,
                "stop_reason": r.stop_reason, "tool_calls": r.tool_calls, "tokens": r.tokens,
                "false_positives": r.false_positives, "missed": r.missed,
            }
            for r in results
        ],
    }  # fmt: skip
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())