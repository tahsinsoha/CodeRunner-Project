"""Offline tests for the LLM client (no network, no key needed)."""

from types import SimpleNamespace

import pytest

from wordle_fibble.llm_client import LLMClient, load_key


class FakeError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClient:
    """Fails with the given errors first, then returns a reply."""

    def __init__(self, errors=()):
        self.errors = list(errors)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        msg = SimpleNamespace(content=" CRANE\nbecause common letters ", reasoning_content="thought")
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("wordle_fibble.llm_client.time.sleep", lambda s: None)


def test_reply_fields():
    r = LLMClient("deepseek", client=FakeClient()).chat([{"role": "user", "content": "hi"}])
    assert r.text == "CRANE\nbecause common letters"
    assert r.reasoning == "thought" and r.input_tokens == 120 and r.output_tokens == 30


def test_default_models():
    assert LLMClient("deepseek", client=FakeClient()).model == "deepseek-flash"
    assert LLMClient("ollama", client=FakeClient()).model == "gpt-oss:20b"


def test_deepseek_thinking_switch():
    for thinking, expected in [(True, "enabled"), (False, "disabled")]:
        fake = FakeClient()
        LLMClient("deepseek", "deepseek-v4-pro", client=fake, thinking=thinking,
                  reasoning_effort="high").chat([])
        kw = fake.calls[0]
        assert kw["model"] == "deepseek-v4-pro" and kw["reasoning_effort"] == "high"
        assert kw["extra_body"] == {"thinking": {"type": expected}}


def test_default_request_has_no_extras():
    fake = FakeClient()
    LLMClient("deepseek", client=fake).chat([])
    assert set(fake.calls[0]) == {"model", "messages", "max_tokens"}


def test_custom_thinking_switch():
    fake = FakeClient()
    LLMClient("custom", "gemma", client=fake, thinking=False, temperature=0).chat([])
    assert fake.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert fake.calls[0]["temperature"] == 0


def test_ollama_thinking_switch_not_supported():
    with pytest.raises(ValueError):
        LLMClient("ollama", client=FakeClient(), thinking=False)


def test_unknown_provider():
    with pytest.raises(ValueError):
        LLMClient("openai", client=FakeClient())


def test_retries_rate_limit_and_server_errors():
    fake = FakeClient([FakeError(429), FakeError(503)])
    r = LLMClient("deepseek", client=fake).chat([])
    assert r.attempts == 3 and len(fake.calls) == 3


def test_does_not_retry_client_errors():
    with pytest.raises(FakeError):
        LLMClient("deepseek", client=FakeClient([FakeError(401)])).chat([])


def test_gives_up_after_max_retries():
    fake = FakeClient([FakeError(429)] * 5)
    with pytest.raises(FakeError):
        LLMClient("deepseek", client=fake, max_retries=2).chat([])
    assert len(fake.calls) == 3


def test_env_file_from_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env = tmp_path / "keys.env"
    env.write_text("DEEPSEEK_API_KEY=sk-file\n")
    monkeypatch.setenv("LLM_ENV_FILE", str(env))
    monkeypatch.setattr("wordle_fibble.llm_client.REPO_DIR", tmp_path / "no-repo")
    assert load_key("DEEPSEEK_API_KEY") == "sk-file"


def test_load_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nDEEPSEEK_API_KEY='sk-abc'\n")
    assert load_key("DEEPSEEK_API_KEY", env_files=[env]) == "sk-abc"
    assert load_key("DEEPSEEK_API_KEY", env_files=[tmp_path / "missing"]) is None
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    assert load_key("DEEPSEEK_API_KEY", env_files=[env]) == "sk-env"


def test_missing_key_message(monkeypatch):
    monkeypatch.setattr("wordle_fibble.llm_client.load_key", lambda var: None)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        LLMClient("deepseek")


def test_ollama_needs_no_key(monkeypatch):
    monkeypatch.setattr("wordle_fibble.llm_client.load_key", lambda var: None)
    assert LLMClient("ollama").model == "gpt-oss:20b"  # builds a real client, no request sent
