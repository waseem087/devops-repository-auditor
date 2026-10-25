"""Tests for the stricter evidence gates and the one-time 'unread files' nudge."""

import json
import shutil

import pytest

import evaluate
from auditor import AgentLimits
from config import Limits
from tools import RepoTools

from .conftest import write
from .test_agent import (
    ScriptedLLM,
    finding,
    last_results,
    make_agent,
    names,
    reply,
    submit,
    tc,
)


def absence(rule_id, category, description="Something is missing", **over):
    return finding(file=None, line=None, rule_id=rule_id, category=category, severity="medium",
                   description=description, evidence="list_files output shows it", **over)  # fmt: skip


# ----------------------------------------------------- file=null discipline
def test_specific_rules_cannot_use_a_null_file(tools):
    steps = [
        reply(tc("list_files")),
        reply(submit(finding(file=None, line=None))),
        reply(tc("read_file", path="config/settings.py")),
        reply(submit(finding())),
    ]
    agent, llm = make_agent(tools, steps)
    result = agent.run()
    feedback = last_results(llm.calls[2]["messages"])[0]["content"]
    assert "give its file and line" in feedback and "SEC-001" in feedback
    assert [v.file for v in result.violations] == ["config/settings.py"] and result.rejected_findings == 1


def test_catch_all_rules_cannot_be_absences(tools):
    steps = [reply(tc("list_files")), reply(submit(absence("SEC-099", "security"))), reply(submit())]
    result = make_agent(tools, steps)[0].run()
    assert result.violations == [] and result.rejected_findings == 1


# ---------------------------------------------------- absence contradictions
def test_no_tests_claim_is_rejected_when_tests_were_listed(tools):
    steps = [reply(tc("list_files")), reply(submit(absence("TST-001", "testing"))), reply(submit())]
    agent, llm = make_agent(tools, steps)
    result = agent.run()
    feedback = last_results(llm.calls[2]["messages"])[0]["content"]
    assert "tests/test_app.py" in feedback and "CI-002" in feedback
    assert result.violations == [] and result.rejected_findings == 1


def test_no_tests_claim_is_accepted_when_there_really_are_none(repo):
    shutil.rmtree(repo / "tests")
    tools = RepoTools(repo, Limits())
    steps = [reply(tc("list_files")), reply(submit(absence("TST-001", "testing", "No automated tests exist")))]
    result = make_agent(tools, steps)[0].run()
    assert [v.rule_id for v in result.violations] == ["TST-001"] and result.violations[0].file is None


def test_missing_readme_claim(repo):
    claim = absence("HYG-001", "repository_hygiene", "README is missing")
    first = make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim)), reply(submit())])[0].run()
    assert first.violations == [] and first.rejected_findings == 1
    (repo / "README.md").unlink()
    second = make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim))])[0].run()
    assert [v.rule_id for v in second.violations] == ["HYG-001"]


def test_missing_gitignore_claim(repo):
    claim = absence("HYG-002", "repository_hygiene", "No .gitignore present")
    assert make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim)), reply(submit())])[0].run().violations == []
    (repo / ".gitignore").unlink()
    result = make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim))])[0].run()
    assert len(result.violations) == 1


def test_no_ci_claim_is_rejected_when_a_workflow_exists(tools):
    claim = absence("CI-001", "ci", "There is no CI workflow")
    result = make_agent(tools, [reply(tc("list_files")), reply(submit(claim)), reply(submit())])[0].run()
    assert result.violations == [] and result.rejected_findings == 1


def test_no_lockfile_claim_checks_for_lockfiles(repo):
    write(repo, "poetry.lock", "# lock\n")
    claim = absence("DEP-002", "dependencies", "No lockfile is committed")
    blocked = make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim)), reply(submit())])[0].run()
    assert blocked.violations == []
    (repo / "poetry.lock").unlink()
    allowed = make_agent(RepoTools(repo, Limits()), [reply(tc("list_files")), reply(submit(claim))])[0].run()
    assert len(allowed.violations) == 1


def test_other_absence_rules_are_not_second_guessed(tools):
    claim = absence("COL-001", "collaboration", "No contribution guidance")
    result = make_agent(tools, [reply(tc("list_files")), reply(submit(claim))])[0].run()
    assert len(result.violations) == 1


# ------------------------------------------------------------------- nudge
def test_unread_workflows_trigger_one_reminder(tools):
    steps = [
        reply(tc("list_files")),
        reply(submit(finding())),  # nudged: nothing recorded
        reply(tc("read_file", path="config/settings.py")),
        reply(submit(finding())),
    ]
    agent, llm = make_agent(tools, steps, unread_nudges=1)
    result = agent.run()
    note = last_results(llm.calls[2]["messages"])[0]
    assert note["is_error"] is True and ".github/workflows/ci.yml" in note["content"] and "requirements.txt" in note["content"]
    assert "nothing has been recorded yet" in note["content"]
    assert "nudged" in result.trace[1].summary and agent.submissions == 1
    assert len(result.violations) == 1


def test_nudge_happens_only_once(tools):
    steps = [reply(tc("list_files")), reply(submit()), reply(submit())]
    agent, _ = make_agent(tools, steps, unread_nudges=1)
    result = agent.run()
    assert names(result) == ["list_files", "submit_findings", "submit_findings"] and result.stop_reason == "submitted"


def test_no_nudge_when_notable_files_were_read(tools):
    steps = [
        reply(tc("list_files")),
        reply(tc("read_file", path=".github/workflows/ci.yml")),
        reply(tc("read_file", path="requirements.txt")),
        reply(submit()),
    ]
    result = make_agent(tools, steps, unread_nudges=1)[0].run()
    assert not any("nudged" in t.summary for t in result.trace)


def test_lockfiles_do_not_trigger_the_nudge(tmp_path):
    write(tmp_path, "README.md", "# x\n")
    write(tmp_path, "poetry.lock", "# lock\n")
    result = make_agent(RepoTools(tmp_path, Limits()), [reply(tc("list_files")), reply(submit())], unread_nudges=1)[0].run()
    assert names(result) == ["list_files", "submit_findings"]


def test_forced_submission_is_never_nudged(tools):
    steps = [reply(tc("list_files")), reply(submit())]
    result = make_agent(tools, steps, max_iterations=1, unread_nudges=1)[0].run()
    assert result.stop_reason == "iteration_limit" and names(result) == ["list_files", "submit_findings"]


def test_nudge_can_be_disabled(tools):
    result = make_agent(tools, [reply(tc("list_files")), reply(submit())], unread_nudges=0)[0].run()
    assert names(result) == ["list_files", "submit_findings"]


def test_unread_nudges_setting_from_environment():
    assert AgentLimits.from_env({"AUDITOR_UNREAD_NUDGES": "0"}).unread_nudges == 0
    assert AgentLimits.from_env({"AUDITOR_UNREAD_NUDGES": "2"}).unread_nudges == 2
    assert AgentLimits().unread_nudges == 1
    with pytest.raises(ValueError):
        AgentLimits.from_env({"AUDITOR_MAX_ITERATIONS": "0"})
    with pytest.raises(ValueError):
        AgentLimits.from_env({"AUDITOR_UNREAD_NUDGES": "-1"})


# ------------------------------------------------------- evaluate --verbose
def test_evaluate_verbose_prints_the_trace(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUDITOR_UNREAD_NUDGES", "0")

    class FakeProvider:
        provider, model = "fake", "m"

        def build_client(self):
            return ScriptedLLM([reply(tc("list_files")), reply(submit())])

    monkeypatch.setattr(evaluate, "load_provider", lambda: FakeProvider())
    out = tmp_path / "o.json"
    assert evaluate.main(["--only", "healthy", "--verbose", "--output", str(out)]) == 0
    text = capsys.readouterr().out
    assert "--- trace: healthy" in text and "list_files" in text and "submit_findings" in text
    assert json.loads(out.read_text())["kind"] == "local-synthetic-fixtures"