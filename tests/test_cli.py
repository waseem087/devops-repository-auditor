"""CLI tests: fully offline. A scripted fake model and a local fixture repo replace the network."""

import json
import os
from contextlib import contextmanager

import pytest

import agent
from repo_access import RepoAccessError
from schemas import Violation

from .conftest import fake_github_token
from .test_agent import ScriptedLLM, finding, reply, submit, tc

URL = "https://github.com/octo/demo"
KEY_FIELDS = ["rule_id", "category", "severity", "file", "line", "description", "evidence", "recommendation"]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-1234567890")
    for name in list(os.environ):
        if name.startswith("AUDITOR_") or name in {"GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"}:
            monkeypatch.delenv(name)
    monkeypatch.setenv("AUDITOR_UNREAD_NUDGES", "0")


class Provider:
    """Stands in for acquire_repo: yields a local fixture and records the lifecycle."""

    def __init__(self, root=None, error=None):
        self.root, self.error = root, error
        self.entered = self.closed = False

    @contextmanager
    def __call__(self, ref, limits):
        if self.error:
            raise self.error
        self.entered = True
        try:
            yield self.root
        finally:
            self.closed = True


def run(argv, repo, steps, tmp_path, **kw):
    llm = ScriptedLLM(steps)
    provider = kw.pop("provider", None) or Provider(repo)
    code = agent.main(argv, repo_provider=provider, llm_factory=lambda settings: llm, env_file=None, **kw)
    return code, provider, llm


def failing_factory(message):
    def factory(settings):
        raise RuntimeError(message)

    return factory


def read_steps():
    return [reply(tc("list_files")), reply(tc("read_file", path="config/settings.py"))]


# ------------------------------------------------------------------- success
def test_success_writes_exact_contract(repo, tmp_path, capsys):
    out = tmp_path / "report.json"
    steps = [*read_steps(), reply(submit(finding(evidence=f"API_TOKEN = '{fake_github_token()}'")))]
    code, provider, _ = run(["--repo", URL, "--output", str(out)], repo, steps, tmp_path)
    assert code == 0 and provider.entered and provider.closed
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    assert list(data) == ["repository", "violations"]
    assert data["repository"] == URL
    assert len(data["violations"]) == 1 and list(data["violations"][0]) == KEY_FIELDS
    v = data["violations"][0]
    assert (v["file"], v["line"], v["severity"]) == ("config/settings.py", 1, "critical")
    assert fake_github_token() not in text
    assert "1 violation(s)" in capsys.readouterr().out
    assert text.endswith("\n") and not text.startswith("```")


def test_healthy_repo_gives_empty_violations_after_inspection(repo, tmp_path):
    out = tmp_path / "r.json"
    code, _, _ = run(["--repo", URL, "--output", str(out)], repo, [*read_steps(), reply(submit())], tmp_path)
    assert code == 0 and json.loads(out.read_text()) == {"repository": URL, "violations": []}


def test_null_file_and_line_are_json_null(repo, tmp_path):
    out = tmp_path / "r.json"
    absence = finding(file=None, line=None, rule_id="COL-001", category="collaboration", severity="low",
                      description="No contribution guidance found", evidence="list_files shows no CONTRIBUTING file")  # fmt: skip
    code, _, _ = run(["--repo", URL, "--output", str(out)], repo, [reply(tc("list_files")), reply(submit(absence))], tmp_path)
    assert code == 0
    assert '"file": null' in out.read_text() and '"line": null' in out.read_text()


def test_repository_field_is_the_validated_url(repo, tmp_path):
    out = tmp_path / "r.json"
    run(["--repo", URL + ".git/", "--output", str(out)], repo, [*read_steps(), reply(submit())], tmp_path)
    assert json.loads(out.read_text())["repository"] == URL


def test_output_is_deterministic_and_sorted(repo, tmp_path):
    low = finding(severity="low", rule_id="QUA-099", category="code_quality", file="src/app.py", line=1)
    crit = finding()

    def steps():
        return [
            reply(tc("read_file", path="config/settings.py")),
            reply(tc("read_file", path="src/app.py")),
            reply(submit(low, crit)),
        ]

    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run(["--repo", URL, "--output", str(a)], repo, steps(), tmp_path)
    run(["--repo", URL, "--output", str(b)], repo, steps(), tmp_path)
    assert a.read_text() == b.read_text()
    assert [v["severity"] for v in json.loads(a.read_text())["violations"]] == ["critical", "low"]


def test_nested_output_directory_is_created(repo, tmp_path):
    out = tmp_path / "deep" / "er" / "report.json"
    code, _, _ = run(["--repo", URL, "--output", str(out)], repo, [*read_steps(), reply(submit())], tmp_path)
    assert code == 0 and out.is_file()


def test_resource_limit_still_writes_report_with_warning(repo, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("AUDITOR_MAX_ITERATIONS", "2")
    out = tmp_path / "r.json"
    steps = [*read_steps(), reply(submit(finding()))]
    code, _, _ = run(["--repo", URL, "--output", str(out), "--verbose"], repo, steps, tmp_path)
    err = capsys.readouterr().err
    assert code == 0 and len(json.loads(out.read_text())["violations"]) == 1
    assert "resource limit reached (iteration_limit)" in err and "list_files" in err


# ------------------------------------------------------- usage / config errors
@pytest.mark.parametrize(
    "bad", ["https://gitlab.com/a/b", "http://github.com/a/b", "/etc/passwd", "https://github.com/a", "https://u:p@github.com/a/b"]
)
def test_bad_urls_exit_2_without_cloning_or_writing(bad, repo, tmp_path, capsys):
    out = tmp_path / "r.json"
    code, provider, llm = run(["--repo", bad, "--output", str(out)], repo, [], tmp_path)
    assert code == 2 and not provider.entered and not out.exists() and llm.calls == []
    assert "Unsupported repository URL" in capsys.readouterr().err


def test_missing_api_key_exits_2_before_any_clone(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    code, provider, _ = run(["--repo", URL, "--output", str(tmp_path / "r.json")], repo, [], tmp_path)
    assert code == 2 and not provider.entered
    assert "No LLM API key found" in capsys.readouterr().err


def test_invalid_limit_variable_exits_2(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("AUDITOR_MAX_TOOL_CALLS", "lots")
    code, provider, _ = run(["--repo", URL, "--output", str(tmp_path / "r.json")], repo, [], tmp_path)
    assert code == 2 and not provider.entered and "AUDITOR_MAX_TOOL_CALLS" in capsys.readouterr().err


def test_missing_required_argument_exits_2(capsys):
    assert agent.main([], env_file=None) == 2
    assert "--repo" in capsys.readouterr().err


def test_help_exits_0(capsys):
    assert agent.main(["--help"], env_file=None) == 0
    assert "--repo" in capsys.readouterr().out


def test_output_path_that_is_a_directory_exits_2(repo, tmp_path):
    code, provider, _ = run(["--repo", URL, "--output", str(tmp_path)], repo, [], tmp_path)
    assert code == 2 and not provider.entered


# ------------------------------------------------------------ fatal failures
def test_clone_failure_exits_1_and_writes_nothing(repo, tmp_path, capsys):
    out = tmp_path / "r.json"
    provider = Provider(error=RepoAccessError("Repository not found, deleted, or not public."))
    code, _, _ = run(["--repo", URL, "--output", str(out)], repo, [], tmp_path, provider=provider)
    assert code == 1 and not out.exists()
    err = capsys.readouterr().err
    assert "not found" in err and "No report was written" in err


def test_llm_failure_exits_1_without_a_report(repo, tmp_path, capsys):
    from llm_client import LLMError

    out = tmp_path / "r.json"
    code, provider, _ = run(["--repo", URL, "--output", str(out)], repo, [LLMError("Rate limited by the API")], tmp_path)
    assert code == 1 and not out.exists() and provider.closed
    assert "Rate limited" in capsys.readouterr().err


def test_incomplete_inspection_is_not_reported_as_healthy(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDITOR_MAX_ITERATIONS", "1")
    out = tmp_path / "r.json"
    code, _, _ = run(["--repo", URL, "--output", str(out)], repo, [reply(tc("delete_repo"))], tmp_path)
    assert code == 1 and not out.exists()


def test_failure_does_not_touch_an_existing_report_or_leave_temp_files(repo, tmp_path):
    out = tmp_path / "r.json"
    out.write_text("previous run")
    run(["--repo", URL, "--output", str(out)], repo, [], tmp_path, provider=Provider(error=RepoAccessError("x")))
    assert out.read_text() == "previous run" and [p.name for p in tmp_path.iterdir() if p.name != "repo"] == ["r.json"]


def test_unexpected_error_hides_internals_and_secrets(repo, tmp_path, capsys):
    secret = fake_github_token()
    factory = failing_factory(f"boom {secret}")
    code = agent.main(["--repo", URL, "--output", str(tmp_path / "r.json")], repo_provider=Provider(repo), llm_factory=factory, env_file=None)
    err = capsys.readouterr().err
    assert code == 3 and secret not in err and "RuntimeError" in err and "Traceback" not in err


def test_debug_traceback_is_redacted(repo, tmp_path, capsys):
    secret = fake_github_token()
    factory = failing_factory(f"token = '{secret}'")
    agent.main(["--repo", URL, "--output", str(tmp_path / "r.json"), "--debug"], repo_provider=Provider(repo), llm_factory=factory, env_file=None)
    err = capsys.readouterr().err
    assert "Traceback" in err and secret not in err


def test_keyboard_interrupt_exits_130(repo, tmp_path):
    def interrupted(settings):
        raise KeyboardInterrupt

    code = agent.main(["--repo", URL, "--output", str(tmp_path / "r.json")], repo_provider=Provider(repo), llm_factory=interrupted, env_file=None)
    assert code == 130


def test_never_prompts_for_input(repo, tmp_path, monkeypatch):
    def no_input(*a, **k):
        raise AssertionError("the CLI must never ask questions")

    monkeypatch.setattr("builtins.input", no_input)
    code, _, _ = run(["--repo", URL, "--output", str(tmp_path / "r.json")], repo, [*read_steps(), reply(submit())], tmp_path)
    assert code == 0


# ------------------------------------------------------------ report writer
def test_render_report_is_valid_and_strict():
    v = Violation.model_validate(finding())
    text = agent.render_report(URL, [v])
    assert json.loads(text)["violations"][0]["rule_id"] == "SEC-001"
    assert agent.render_report(URL, []) == '{\n  "repository": "' + URL + '",\n  "violations": []\n}\n'


def test_atomic_write_replaces_whole_file(tmp_path):
    target = tmp_path / "r.json"
    target.write_text("old")
    agent.write_json_atomic(target, '{"a": 1}\n')
    assert target.read_text() == '{"a": 1}\n' and list(tmp_path.iterdir()) == [target]


def test_atomic_write_failure_keeps_old_file_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "r.json"
    target.write_text("old")
    monkeypatch.setattr(agent.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        agent.write_json_atomic(target, "new")
    assert target.read_text() == "old" and list(tmp_path.iterdir()) == [target]