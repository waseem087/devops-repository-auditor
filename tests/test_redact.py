from redact import REDACTED, contains_secret, redact_text, redact_url

from .conftest import fake_aws_key, fake_github_token


def test_github_token_redacted():
    tok = fake_github_token()
    out = redact_text(f"token = '{tok}'")
    assert tok not in out and "REDACTED" in out


def test_aws_key_redacted_with_label():
    key = fake_aws_key()
    out = redact_text(f"aws = {key}")
    assert key not in out and "aws-access-key" in out


def test_generic_assignment_redacted_but_key_kept():
    out = redact_text('password = "hunter2hunter2"')
    assert "hunter2" not in out and out.startswith("password")


def test_json_style_assignment():
    out = redact_text('{"api_key": "zzzzzzzz12345"}')
    assert "zzzzzzzz12345" not in out


def test_placeholders_and_env_references_untouched():
    for line in [
        'password = os.environ["DB_PASSWORD"]',
        "token: ${{ secrets.GITHUB_TOKEN }}",
        'API_KEY = "your-key-here"',
        "password = get_password()",
    ]:
        assert redact_text(line) == line


def test_url_credentials_redacted():
    out = redact_text("git clone https://bob:s3cr3tpass@github.com/x/y")
    assert "s3cr3tpass" not in out and "bob" in out
    assert "tokenvalue123" not in redact_text("https://tokenvalue123@github.com/x/y")


def test_private_key_block_redacted_and_line_count_preserved():
    text = "a\n-----BEGIN RSA PRIVATE KEY-----\nMIIEsecretbody\nmorebody\n-----END RSA PRIVATE KEY-----\nz"
    out = redact_text(text)
    assert "MIIEsecretbody" not in out and "morebody" not in out
    assert out.count("\n") == text.count("\n")


def test_inline_private_key():
    out = redact_text('k = "-----BEGIN PRIVATE KEY-----ABCDEF-----END PRIVATE KEY-----"')
    assert "ABCDEF" not in out and REDACTED in out


def test_plain_text_unchanged_and_contains_secret():
    assert redact_text("hello world\nauthor = 'Jo'") == "hello world\nauthor = 'Jo'"
    assert contains_secret(f"x = {fake_github_token()}")
    assert not contains_secret("nothing here")


def test_redact_url():
    assert redact_url("https://user:pw@github.com/a/b.git") == "https://github.com/a/b.git"
    assert redact_url("https://github.com/a/b") == "https://github.com/a/b"
