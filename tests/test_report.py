import pytest

from schemas import (
    CATEGORY_PREFIX,
    RULE_CATALOG,
    SUBMIT_TOOL_SPEC,
    Report,
    Violation,
    dedupe_violations,
    parse_finding,
    sort_violations,
)

from .conftest import fake_github_token


def good(**over):
    base = {
        "rule_id": "SEC-001", "category": "security", "severity": "critical",
        "file": "config.py", "line": 17, "description": "Hardcoded API credential",
        "evidence": "Credential assigned directly in source code",
        "recommendation": "Use a secrets manager or environment variable",
    }  # fmt: skip
    base.update(over)
    return base


def test_valid_finding_round_trips_with_exact_fields():
    v, errs = parse_finding(good())
    assert errs == [] and v is not None
    assert list(v.model_dump()) == [
        "rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation",
    ]  # fmt: skip


@pytest.mark.parametrize("sev", ["critical", "high", "medium", "low"])
def test_all_severities_allowed(sev):
    assert parse_finding(good(severity=sev))[1] == []


@pytest.mark.parametrize("sev", ["Critical", "severe", "info", "", None, 3])
def test_invalid_severity_rejected(sev):
    v, errs = parse_finding(good(severity=sev))
    assert v is None and any("severity" in e for e in errs)


def test_missing_and_extra_fields_rejected():
    raw = good()
    del raw["evidence"]
    assert parse_finding(raw)[0] is None
    assert parse_finding(good(extra_field="x"))[0] is None
    for field in ("file", "line"):
        raw = good()
        del raw[field]
        assert parse_finding(raw)[0] is None  # null must be explicit


def test_null_file_and_line_allowed_together_only():
    v, errs = parse_finding(good(file=None, line=None, rule_id="TST-001", category="testing"))
    assert errs == [] and v.file is None and v.line is None
    assert parse_finding(good(file=None, line=3))[0] is None
    assert parse_finding(good(line=None))[1] == []  # file known, line unknown is fine


@pytest.mark.parametrize("line", [0, -4, True, "7", 2.5])
def test_bad_line_rejected(line):
    assert parse_finding(good(line=line))[0] is None


@pytest.mark.parametrize("rule", ["sec-001", "SEC001", "SEC-1", "SECURITY-001", "", "SEC-0001"])
def test_bad_rule_id_rejected(rule):
    assert parse_finding(good(rule_id=rule))[0] is None


def test_rule_prefix_must_match_category():
    v, errs = parse_finding(good(rule_id="CI-002"))
    assert v is None and any("prefix" in e for e in errs)
    assert parse_finding(good(rule_id="CI-002", category="ci"))[1] == []


def test_unknown_category_rejected():
    assert parse_finding(good(category="misc"))[0] is None


@pytest.mark.parametrize("path", ["/etc/passwd", "../x.py", "a/../../b", "C:\\x.py", ""])
def test_unsafe_file_paths_rejected(path):
    assert parse_finding(good(file=path))[0] is None


def test_file_path_normalized():
    assert parse_finding(good(file="./src\\app.py"))[0].file == "src/app.py"


def test_empty_text_fields_rejected():
    for field in ("description", "evidence", "recommendation"):
        assert parse_finding(good(**{field: "  "}))[0] is None


def test_not_an_object_rejected():
    assert parse_finding("text")[0] is None and parse_finding(None)[0] is None


def test_secrets_redacted_from_text_fields():
    tok = fake_github_token()
    v, _ = parse_finding(good(description=f"Token {tok} committed", evidence=f"API_TOKEN = '{tok}'", recommendation=f"rotate {tok}"))
    assert tok not in v.description + v.evidence + v.recommendation


def test_error_messages_do_not_echo_input_values():
    tok = fake_github_token()
    _, errs = parse_finding(good(severity=tok))
    assert tok not in " ".join(errs)


def test_dedupe_exact_and_same_location():
    a = Violation.model_validate(good())
    b = Violation.model_validate(good())
    c = Violation.model_validate(good(rule_id="SEC-006", description="Token handled insecurely"))
    other = Violation.model_validate(good(line=40))
    out = dedupe_violations([a, b, c, other])
    assert [(v.rule_id, v.line) for v in out] == [("SEC-001", 17), ("SEC-001", 40)]


def test_dedupe_keeps_most_severe_for_same_location():
    low = Violation.model_validate(good(severity="low", rule_id="SEC-006"))
    high = Violation.model_validate(good(severity="high"))
    assert [v.severity for v in dedupe_violations([low, high])] == ["high"]


def test_dedupe_absence_findings_by_description():
    x = Violation.model_validate(good(file=None, line=None, rule_id="TST-001", category="testing", description="No tests exist"))
    y = Violation.model_validate(good(file=None, line=None, rule_id="TST-001", category="testing", description="No  tests exist!"))
    assert len(dedupe_violations([x, y])) == 1


def test_sort_is_by_severity_then_location():
    items = [
        Violation.model_validate(good(severity="low", file="b.py", rule_id="CFG-001", category="configuration")),
        Violation.model_validate(good(severity="critical", file="z.py")),
        Violation.model_validate(good(severity="critical", file="a.py")),
    ]
    assert [(v.severity, v.file) for v in sort_violations(items)] == [("critical", "a.py"), ("critical", "z.py"), ("low", "b.py")]


def test_report_model_is_strict():
    r = Report(repository="https://github.com/o/r", violations=[Violation.model_validate(good())])
    assert list(r.model_dump()) == ["repository", "violations"]
    with pytest.raises(ValueError):
        Report.model_validate({"repository": "x", "violations": [], "extra": 1})


def test_submit_spec_matches_violation_fields():
    item = SUBMIT_TOOL_SPEC["input_schema"]["properties"]["findings"]["items"]
    assert set(item["properties"]) == set(Violation.model_fields) == set(item["required"])
    assert item["additionalProperties"] is False
    assert item["properties"]["category"]["enum"] == sorted(CATEGORY_PREFIX)


def test_rule_catalog_is_consistent_with_prefixes():
    prefixes = set(CATEGORY_PREFIX.values())
    assert all(rule.split("-")[0] in prefixes for rule in RULE_CATALOG)
    assert all(any(r.startswith(p + "-") for r in RULE_CATALOG) for p in prefixes)