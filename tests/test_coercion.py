import pytest

from auditor import _coerce_finding

from .test_agent import ScriptedLLM, finding, make_agent, reply, submit, tc


def test_normalises_case_and_numeric_strings():
    f = _coerce_finding({**finding(), "severity": " HIGH ", "category": "Security", "rule_id": "sec-001", "line": "7"})
    assert (f["severity"], f["category"], f["rule_id"], f["line"]) == ("high", "security", "SEC-001", 7)


@pytest.mark.parametrize("value", ["", "null", "None", "N/A"])
def test_null_like_strings_become_none(value):
    f = _coerce_finding({**finding(), "file": value, "line": value})
    assert f["file"] is None and f["line"] is None


def test_missing_file_and_line_default_to_none():
    raw = finding()
    del raw["file"], raw["line"]
    f = _coerce_finding(raw)
    assert f["file"] is None and f["line"] is None


def test_float_lines_and_non_dicts():
    assert _coerce_finding({**finding(), "line": 3.0})["line"] == 3
    assert _coerce_finding({**finding(), "line": 3.5})["line"] == 3.5  # stays invalid; validation rejects it
    assert _coerce_finding("text") == "text"


def test_garbage_is_left_for_validation_to_reject():
    f = _coerce_finding({**finding(), "line": "seven", "severity": 5})
    assert f["line"] == "seven" and f["severity"] == 5


def test_original_is_not_mutated():
    raw = {**finding(), "line": "7"}
    _coerce_finding(raw)
    assert raw["line"] == "7"


def test_agent_accepts_a_sloppy_but_sensible_submission(tools):
    sloppy = {**finding(), "severity": "CRITICAL", "line": "1"}
    steps = [reply(tc("read_file", path="config/settings.py")), reply(submit(sloppy))]
    result = make_agent(tools, steps)[0].run()
    assert [(v.severity, v.line) for v in result.violations] == [("critical", 1)]


def test_agent_accepts_an_absence_without_file_or_line_keys(tools):
    absence = finding(rule_id="COL-001", category="collaboration", severity="low",
                      description="No contribution guidance found", evidence="list_files shows no CONTRIBUTING file")  # fmt: skip
    del absence["file"], absence["line"]
    result = make_agent(tools, [reply(tc("list_files")), reply(submit(absence))])[0].run()
    assert len(result.violations) == 1 and result.violations[0].file is None


def test_invalid_values_are_still_rejected(tools):
    steps = [reply(tc("read_file", path="config/settings.py")), reply(submit({**finding(), "severity": "urgent"})), reply(submit())]
    agent, llm = make_agent(tools, steps)
    assert agent.run().violations == [] and isinstance(llm, ScriptedLLM)