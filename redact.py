"""Secret redaction. Everything the agent shows to the LLM, logs or writes to the
report should pass through these helpers first.

The redactor is deliberately conservative: it may hide a harmless value, but it
should never reveal a real credential. Line counts are always preserved so that
line numbers stay accurate.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit, urlunsplit

REDACTED = "[REDACTED]"

# Well-known token formats: (label, regex)
_TOKEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("github-token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("api-key", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b")),
    ("slack-token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
]

_URL_PASSWORD = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<user>[^/\s:@]+):(?P<pw>[^@\s/]+)@")
_URL_TOKEN_ONLY = re.compile(r"(?P<scheme>https?://)(?P<tok>[^/\s:@]+)@")

_PK_INLINE = re.compile(
    r"(-----BEGIN [A-Z ]*PRIVATE KEY[A-Z ]*-----).+?(-----END [A-Z ]*PRIVATE KEY[A-Z ]*-----)"
)
_PK_BEGIN = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY[A-Z ]*-----")
_PK_END = re.compile(r"-----END [A-Z ]*PRIVATE KEY[A-Z ]*-----")

_ASSIGNMENT = re.compile(
    r"(?P<key>[\w.\-]*(?:password|passwd|pwd|secret|token|api[_-]?key|apikey|access[_-]?key"
    r"|private[_-]?key|credential)[\w.\-]*)"
    r"(?P<sep>[\"']?\s*[:=]\s*)"
    r"(?P<q>[\"']?)"
    r"(?P<val>[^\s\"'#,;]{6,})",
    re.IGNORECASE,
)

# Values that are clearly references or placeholders, not literal secrets.
_PLACEHOLDER_PREFIXES = (
    "$", "{{", "<", "[", "%", "os.", "os[", "getenv", "env", "process.", "secrets.",
    "config", "settings", "your", "change", "placeholder", "example", "xxx", "***",
    "none", "null", "true", "false", "todo", "dummy", "fake",
)  # fmt: skip


def _is_placeholder(value: str) -> bool:
    return value.lower().startswith(_PLACEHOLDER_PREFIXES)


def _assignment_sub(match: re.Match[str]) -> str:
    value, quote = match.group("val"), match.group("q")
    if _is_placeholder(value):
        return match.group(0)
    if not quote and "(" in value:  # unquoted function call / expression, not a literal
        return match.group(0)
    return f"{match.group('key')}{match.group('sep')}{quote}{REDACTED}"


def _redact_line(line: str) -> str:
    line = _URL_PASSWORD.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:{REDACTED}@", line)
    line = _URL_TOKEN_ONLY.sub(lambda m: f"{m.group('scheme')}{REDACTED}@", line)
    for label, pattern in _TOKEN_PATTERNS:
        line = pattern.sub(f"[REDACTED:{label}]", line)
    return _ASSIGNMENT.sub(_assignment_sub, line)


def redact_text(text: str) -> str:
    """Return text with secrets masked. The number of lines is unchanged."""
    out: list[str] = []
    in_private_key = False
    for line in text.split("\n"):
        if in_private_key:
            if _PK_END.search(line):
                in_private_key = False
                out.append(line)
            else:
                out.append(REDACTED)
            continue
        line = _PK_INLINE.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(2)}", line)
        if _PK_BEGIN.search(line) and not _PK_END.search(line):
            in_private_key = True
            out.append(line)
            continue
        out.append(_redact_line(line))
    return "\n".join(out)


def contains_secret(text: str) -> bool:
    """True if redaction would change the text."""
    return redact_text(text) != text


def redact_url(url: str) -> str:
    """Remove any user:password@ part from a URL."""
    try:
        parts = urlsplit(url)
        if parts.username is None and parts.password is None:
            return url
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except ValueError:
        return "[unparseable-url]"
