"""Report data contract: the exact violation shape, validation, de-duplication.

A Violation has exactly eight fields (no extras). Anything the LLM submits goes through
`parse_finding`, which validates the shape and redacts secrets from the free-text fields.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from redact import redact_text

Severity = Literal["critical", "high", "medium", "low"]
SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_ORDER = {name: i for i, name in enumerate(SEVERITIES)}

# category -> rule-id prefix
CATEGORY_PREFIX = {
    "security": "SEC",
    "ci": "CI",
    "dependencies": "DEP",
    "testing": "TST",
    "code_quality": "QUA",
    "configuration": "CFG",
    "repository_hygiene": "HYG",
    "collaboration": "COL",
    "build_release": "BLD",
}

# Stable rule ids. NNN=099 is the catch-all for a defensible issue that fits no listed rule.
RULE_CATALOG = {
    "SEC-001": "Hardcoded credential or secret in source/config",
    "SEC-002": "Insecure workflow permissions (e.g. write-all, excessive token scope)",
    "SEC-003": "Unsafe pull_request_target / untrusted PR code running with secrets",
    "SEC-004": "Untrusted input interpolated into a shell command (injection)",
    "SEC-005": "Third-party action or dependency referenced insecurely (mutable ref, unpinned)",
    "SEC-006": "Risky token/secret handling (printed, echoed, passed insecurely)",
    "SEC-099": "Other security issue",
    "CI-001": "Workflow trigger missing or incorrect",
    "CI-002": "Tests not run in CI, or test step skipped/ineffective",
    "CI-003": "Lint/quality gate missing or ineffective",
    "CI-004": "Incorrect job dependency or condition",
    "CI-099": "Other CI defect",
    "DEP-001": "Inconsistent dependency declarations",
    "DEP-002": "Missing lockfile where reproducibility requires one",
    "DEP-003": "Unbounded or weak dependency constraints (material)",
    "DEP-004": "Risky dependency installation practice",
    "DEP-099": "Other dependency issue",
    "TST-001": "No automated tests where tests are relevant",
    "TST-002": "Incorrect test command or configuration",
    "TST-099": "Other testing issue",
    "QUA-001": "Missing or ineffective lint/format configuration",
    "QUA-099": "Other code-quality issue",
    "CFG-001": "Machine-specific absolute path",
    "CFG-002": "Committed local settings or environment files",
    "CFG-003": "Environment-specific value embedded in source",
    "CFG-099": "Other configuration issue",
    "HYG-001": "Missing or inadequate README",
    "HYG-002": "Missing or unsuitable .gitignore",
    "HYG-003": "Tracked generated, temporary or build artifacts",
    "HYG-099": "Other repository-hygiene issue",
    "COL-001": "Missing contribution or review guidance where it matters",
    "COL-099": "Other collaboration issue",
    "BLD-001": "Missing build validation where a build exists",
    "BLD-002": "Risky release or deployment automation",
    "BLD-099": "Other build/release issue",
}

_RULE_ID_RE = re.compile(r"[A-Z]{2,3}-\d{3}")


class Violation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    rule_id: str
    category: str
    severity: Severity
    file: str | None
    line: StrictInt | None
    description: str = Field(min_length=3, max_length=300)
    evidence: str = Field(min_length=3, max_length=600)
    recommendation: str = Field(min_length=3, max_length=500)

    @field_validator("rule_id")
    @classmethod
    def _check_rule_id(cls, v: str) -> str:
        if not _RULE_ID_RE.fullmatch(v):
            raise ValueError("rule_id must look like SEC-001 (PREFIX-NNN)")
        return v

    @field_validator("category")
    @classmethod
    def _check_category(cls, v: str) -> str:
        if v not in CATEGORY_PREFIX:
            raise ValueError(f"category must be one of {sorted(CATEGORY_PREFIX)}")
        return v

    @field_validator("file")
    @classmethod
    def _check_file(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().replace("\\", "/")
        while v.startswith("./"):
            v = v[2:]
        if not v or v.startswith("/") or re.match(r"^[A-Za-z]:", v) or ".." in v.split("/") or "\x00" in v:
            raise ValueError("file must be a relative path inside the repository, or null")
        return v

    @field_validator("line")
    @classmethod
    def _check_line(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("line must be >= 1, or null")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> Violation:
        if not self.rule_id.startswith(CATEGORY_PREFIX[self.category] + "-"):
            raise ValueError(f"rule_id prefix must be {CATEGORY_PREFIX[self.category]}- for category {self.category}")
        if self.file is None and self.line is not None:
            raise ValueError("line must be null when file is null")
        return self


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    repository: str
    violations: list[Violation]


def parse_finding(raw: Any) -> tuple[Violation | None, list[str]]:
    """Validate one LLM-supplied finding. Returns (violation, []) or (None, reasons)."""
    if not isinstance(raw, dict):
        return None, ["finding must be a JSON object"]
    try:
        v = Violation.model_validate(raw)
    except ValidationError as exc:
        reasons = []
        for err in exc.errors():
            where = ".".join(str(p) for p in err["loc"]) or "finding"
            reasons.append(f"{where}: {err['msg']}")
        return None, reasons
    clean = v.model_copy(
        update={
            "description": redact_text(v.description),
            "evidence": redact_text(v.evidence),
            "recommendation": redact_text(v.recommendation),
        }
    )
    return clean, []


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def sort_violations(items: list[Violation]) -> list[Violation]:
    return sorted(items, key=lambda v: (SEVERITY_ORDER[v.severity], v.file or "", v.line or 0, v.rule_id))


def dedupe_violations(items: list[Violation]) -> list[Violation]:
    """Drop duplicates (same location+category, or same rule/location/description)."""
    seen_loc: set[tuple[str, int, str]] = set()
    seen_desc: set[tuple[str, str | None, int | None, str]] = set()
    out: list[Violation] = []
    for v in sort_violations(items):  # most severe first, so it wins
        desc_key = (v.rule_id, v.file, v.line, _normalize(v.description))
        loc_key = (v.file, v.line, v.category) if v.file is not None and v.line is not None else None
        if desc_key in seen_desc or (loc_key is not None and loc_key in seen_loc):
            continue
        seen_desc.add(desc_key)
        if loc_key is not None:
            seen_loc.add(loc_key)
        out.append(v)
    return out


SUBMIT_TOOL_NAME = "submit_findings"

SUBMIT_TOOL_SPEC: dict[str, Any] = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Submit your final, evidence-backed findings and finish the audit. Only report defects you "
        "confirmed by reading the file (a search match alone is not enough). Submit an empty list if "
        "the repository has no defensible violations. Findings that fail validation are returned to you "
        "with reasons so you can fix them; do not resend findings that were accepted."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": 50,
                "items": {
                    "type": "object",
                    "properties": {
                        "rule_id": {"type": "string", "description": "e.g. SEC-001; prefix must match category"},
                        "category": {"type": "string", "enum": sorted(CATEGORY_PREFIX)},
                        "severity": {"type": "string", "enum": list(SEVERITIES)},
                        "file": {"type": ["string", "null"], "description": "Repo-relative path, or null for an absence"},
                        "line": {"type": ["integer", "null"], "minimum": 1, "description": "1-based line as shown by read_file"},
                        "description": {"type": "string", "maxLength": 300},
                        "evidence": {"type": "string", "maxLength": 600, "description": "What you saw (redacted); quote the key line"},
                        "recommendation": {"type": "string", "maxLength": 500},
                    },
                    "required": [
                        "rule_id", "category", "severity", "file", "line",
                        "description", "evidence", "recommendation",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    },
}  # fmt: skip