"""Central configuration: resource limits shared by tools, repo access and the agent.

Every limit can be overridden with an environment variable named
AUDITOR_<FIELD_NAME_IN_UPPERCASE>, e.g. AUDITOR_MAX_FILE_BYTES=100000.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Limits:
    # --- repository inspection tools ---
    max_files: int = 5000  # files counted by list_files before it stops
    max_listed_files: int = 300  # file paths echoed back to the LLM
    max_depth: int = 8  # default directory depth for list_files
    max_file_bytes: int = 256_000  # largest file read_file / search_code will open
    max_output_chars: int = 20_000  # cap on text returned by read_file
    max_search_results: int = 50  # matches returned by search_code
    max_search_files: int = 3000  # files search_code will open
    search_deadline_s: float = 10.0  # wall-clock budget for one search
    git_timeout_s: float = 15.0  # timeout for each git subprocess
    max_log_entries: int = 50  # commits returned by git_log
    # --- repository acquisition ---
    clone_timeout_s: float = 180.0
    max_repo_bytes: int = 300_000_000

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Limits:
        """Build limits from AUDITOR_* environment variables (defaults otherwise)."""
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        for f in fields(cls):
            var = f"AUDITOR_{f.name.upper()}"
            raw = env.get(var)
            if raw is None or raw == "":
                continue
            try:
                value = type(f.default)(raw)
            except ValueError as exc:
                raise ValueError(f"{var} must be a number, got {raw!r}") from exc
            if value <= 0:
                raise ValueError(f"{var} must be greater than zero, got {raw!r}")
            values[f.name] = value
        return cls(**values)  # type: ignore[arg-type]
