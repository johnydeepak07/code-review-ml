# tests/conftest.py
import pytest

from analyzer import suggestion_generator


class _FakeMessage:
    content = "1. Mock suggestion for testing.\n2. Mock suggestion.\n3. Mock suggestion."


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


@pytest.fixture(autouse=True)
def mock_openai(monkeypatch):
    """
    Replace the OpenAI call with a canned response for every test.
    Without this, any test whose code triggers an issue (missing docstring,
    low naming entropy, etc.) makes a real, billed API call — the suite
    would cost money, require network + a valid key, and be non-deterministic.
    This still exercises the real _build_issues_list -> prompt -> response
    wiring; it only stubs out the network call itself.
    """
    def _fake_create(*args, **kwargs):
        return _FakeResponse()

    monkeypatch.setattr(
        suggestion_generator.client.chat.completions, "create", _fake_create
    )
