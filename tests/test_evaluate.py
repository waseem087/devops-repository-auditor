"""Offline tests for the evaluation machinery. They check the scoring logic and the fixture labels.

They do NOT measure how well a real model detects defects: that needs `python evaluate.py` and an API key.
"""

import json

import pytest

import evaluate
from auditor import AgentLimits
from prompts import SYSTEM_PROMPT
from redact import _TOKEN_PATTERNS
from schemas import Violation
from tests.fixture_repos import FIXTURES

from .test_agent import ScriptedLLM, finding, reply, submit, tc

NO_NUDGE = AgentLimits(unread_nudges=0)


def viol(**over):
    base = {
        "rule_id": "SEC-001", "category": "security", "severity": "high", "file": "a.py", "line": 10,
        "description": "Hardcoded secret", "evidence": "SECRET = literal", "recommendation": "Use environment variables",
    }  # fmt: skip
    base.update(over)
    return Violation.model_validate(base)


def exp(file="a.py", line=10, categories=("security",)):
    return {"categories": list(categories), "file": file, "line": line}


# ------------------------------------------------------------------ scoring
def test_perfect_match():
    tp, extra, missed = evaluate.match_findings([exp()], [viol()])
    assert (tp, extra, missed) == (1, [], [])


def test_false_positive_and_false_negative():
    tp, extra, missed = evaluate.match_findings([exp(file="a.py")], [viol(file="b.py")])
    assert (tp, extra, missed) == (0, [0], [0])


def test_line_tolerance_and_missing_lines():
    assert evaluate.match_findings([exp(line=10)], [viol(line=14)])[0] == 1
    assert evaluate.match_findings([exp(line=10)], [viol(line=30)])[0] == 0
    assert evaluate.match_findings([exp(line=None)], [viol(line=99)])[0] == 1  # file-level label
    assert evaluate.match_findings([exp(line=10)], [viol(line=None)])[0] == 1  # prediction without a line
    assert evaluate.match_findings([exp(line=10)], [viol(line=30)], tolerance=50)[0] == 1


def test_category_alternatives_and_wrong_category():
    ci = {"categories": ["ci", "testing"], "file": "a.py", "line": None}
    assert evaluate.match_findings([ci], [viol(category="testing", rule_id="TST-002")])[0] == 1
    assert evaluate.match_findings([ci], [viol(category="security")])[0] == 0


def test_matching_is_one_to_one():
    tp, extra, _ = evaluate.match_findings([exp()], [viol(), viol(rule_id="SEC-006")])
    assert tp == 1 and extra == [1]
    tp, extra, missed = evaluate.match_findings([exp(), exp()], [viol()])
    assert (tp, extra, missed) == (1, [], [1])


def test_absence_labels_match_null_file_only():
    absence = {"categories": ["testing"], "file": None, "line": None}
    pred = viol(file=None, line=None, category="testing", rule_id="TST-001")
    assert evaluate.match_findings([absence], [pred])[0] == 1
    assert evaluate.match_findings([absence], [viol(category="testing", rule_id="TST-001")])[0] == 0


def test_precision_recall_f1_arithmetic():
    assert evaluate.prf(3, 1, 1) == {"precision": 0.75, "recall": 0.75, "f1": 0.75}
    assert evaluate.prf(2, 0, 2) == {"precision": 1.0, "recall": 0.5, "f1": 0.6667}
    assert evaluate.prf(0, 3, 0) == {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    assert evaluate.prf(0, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert evaluate.prf(0, 2, 2)["f1"] == 0.0


def test_summary_counts_crashed_fixtures_as_misses():
    ok = evaluate.FixtureResult("a", [exp()], tp=1)
    crashed = evaluate.FixtureResult("b", [exp(), exp()], error="API down", fn=2)
    s = evaluate.summarize([ok, crashed])
    assert (s["tp"], s["fp"], s["fn"], s["failed_runs"]) == (1, 0, 2, 1)
    assert s["recall"] == pytest.approx(0.3333, abs=1e-4)


# ------------------------------------------------------------------ fixtures
def test_every_fixture_label_resolves_to_a_real_line():
    for fixture in FIXTURES.values():
        for item in fixture.resolved_expected():
            if item["file"] is not None:
                assert item["file"] in fixture.files
            if item["line"] is not None:
                assert 1 <= item["line"] <= len(fixture.files[item["file"]].split("\n"))


def test_fixture_set_covers_the_required_scenarios():
    names = set(FIXTURES)
    assert {"hardcoded_secret", "workflow_write_all", "ci_tests_swallowed", "ci_no_test_step", "machine_specific_path",
            "acceptable_alternatives", "healthy", "suspicious_but_fine"} <= names  # fmt: skip
    clean = [n for n, f in FIXTURES.items() if not f.expected]
    assert {"acceptable_alternatives", "healthy", "suspicious_but_fine"} <= set(clean)
    assert len(FIXTURES["mixed_defects"].expected) == 4


def test_fixtures_contain_no_real_looking_provider_tokens():
    for fixture in FIXTURES.values():
        for text in fixture.files.values():
            assert not any(pattern.search(text) for _, pattern in _TOKEN_PATTERNS)


def test_marked_lines_really_contain_the_defect():
    f = FIXTURES["hardcoded_secret"]
    line = f.resolved_expected()[0]["line"]
    assert "SECRET_KEY" in f.files["app/config.py"].split("\n")[line - 1]
    assert "write-all" in FIXTURES["workflow_write_all"].files[".github/workflows/ci.yml"]
    assert "|| true" in FIXTURES["ci_tests_swallowed"].files[".github/workflows/ci.yml"]
    assert "pytest" not in FIXTURES["ci_no_test_step"].files[".github/workflows/ci.yml"]


def test_prompt_explains_masked_values():
    assert "[REDACTED]" in SYSTEM_PROMPT and "Never try to guess a masked value" in SYSTEM_PROMPT


# ------------------------------------------------------------- the pipeline
def oracle_steps(fixture):
    """A scripted 'model' that reads each labelled file and reports the labelled defects.

    This only proves the evaluation pipeline works end to end. It says NOTHING about real detection quality.
    """
    steps = [reply(tc("list_files"))]
    files = sorted({e["file"] for e in fixture.resolved_expected() if e["file"]})
    steps += [reply(tc("read_file", path=f)) for f in files]
    findings = []
    for e in fixture.resolved_expected():
        category = e["categories"][0]
        prefix = {"security": "SEC", "configuration": "CFG", "ci": "CI", "testing": "TST"}[category]
        findings.append(
            finding(rule_id=f"{prefix}-099", category=category, file=e["file"], line=e["line"], severity="medium")
        )
    return [*steps, reply(submit(*findings))]


def test_pipeline_scores_perfectly_with_an_oracle_model(tmp_path):
    scored = []
    for fixture in FIXTURES.values():
        llm = ScriptedLLM(oracle_steps(fixture))
        scored.append(evaluate.run_fixture(fixture, lambda llm=llm: llm, tmp_path, agent_limits=NO_NUDGE))
    summary = evaluate.summarize(scored)
    assert summary["fp"] == 0 and summary["fn"] == 0 and summary["failed_runs"] == 0
    assert summary["f1"] == 1.0 and summary["tp"] == sum(len(f.expected) for f in FIXTURES.values())


def test_pipeline_penalises_a_model_that_hallucinates_and_one_that_misses(tmp_path):
    healthy = FIXTURES["healthy"]
    llm = ScriptedLLM([reply(tc("read_file", path="README.md")), reply(submit(finding(file="README.md", line=1, rule_id="HYG-099", category="repository_hygiene", severity="low")))])
    r = evaluate.run_fixture(healthy, lambda: llm, tmp_path, agent_limits=NO_NUDGE)
    assert (r.tp, r.fp, r.fn) == (0, 1, 0) and r.false_positives

    secret = FIXTURES["hardcoded_secret"]
    llm2 = ScriptedLLM([reply(tc("list_files")), reply(submit())])
    r2 = evaluate.run_fixture(secret, lambda: llm2, tmp_path, agent_limits=NO_NUDGE)
    assert (r2.tp, r2.fp, r2.fn) == (0, 0, 1) and r2.missed


def test_failed_run_is_recorded_not_raised(tmp_path):
    from llm_client import LLMError

    llm = ScriptedLLM([LLMError("Rate limited by the API")])
    r = evaluate.run_fixture(FIXTURES["healthy"], lambda: llm, tmp_path)
    assert r.error and "Rate limited" in r.error


def test_fixture_checkout_is_a_git_repo_when_git_exists(tmp_path):
    import shutil

    root = evaluate.prepare_checkout(FIXTURES["healthy"], tmp_path)
    assert (root / "README.md").is_file()
    if shutil.which("git"):
        assert (root / ".git").is_dir()


def test_table_and_cli_listing(capsys):
    r = evaluate.FixtureResult("x", [exp()], tp=1)
    text = evaluate.format_table([r])
    assert "TOTAL" in text and "precision=1.00" in text
    assert evaluate.main(["--list"]) == 0
    assert "hardcoded_secret" in capsys.readouterr().out
    assert evaluate.main(["--only", "nope"]) == 2


def test_cli_requires_api_key(monkeypatch, tmp_path, capsys):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "AUDITOR_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # so no real .env is picked up
    assert evaluate.main(["--only", "healthy", "--output", str(tmp_path / "o.json")]) == 2
    assert "No LLM API key found" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_results_json_is_labelled_as_local(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDITOR_UNREAD_NUDGES", "0")

    class FakeProvider:
        provider, model = "fake", "model-1"

        def build_client(self):
            return ScriptedLLM([reply(tc("list_files")), reply(submit())])

    monkeypatch.setattr(evaluate, "load_provider", lambda: FakeProvider())
    out = tmp_path / "o.json"
    assert evaluate.main(["--only", "healthy", "--output", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["kind"] == "local-synthetic-fixtures" and data["summary"]["f1"] == 1.0
    assert data["model"] == "fake/model-1"
