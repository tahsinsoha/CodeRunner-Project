"""Connection to LLMs through OpenAI-compatible chat APIs.

Providers (none of these models were tested in the paper):

    deepseek : hosted DeepSeek API (paid). Key in DEEPSEEK_API_KEY.
               Models: deepseek-flash (V4.1 Flash), deepseek-v4-pro.
    ollama   : an Ollama server for an open-weight model such as gpt-oss:20b.
               Address in OLLAMA_BASE_URL (default http://localhost:11434/v1).
               No key needed.
    custom   : any other OpenAI-compatible server. Address in LLM_BASE_URL,
               key in LLM_API_KEY.

Keys come from the environment or from a .env file, which is never committed:
the repo's .env, or the file named in LLM_ENV_FILE.

    DEEPSEEK_API_KEY=sk-...

    from wordle_fibble.llm_client import LLMClient
    llm = LLMClient("deepseek", "deepseek-flash", thinking=False)
    reply = llm.chat([{"role": "user", "content": "Name one 5-letter word."}])
    print(reply.text, reply.output_tokens, reply.seconds)
"""

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent

PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_var": "DEEPSEEK_API_KEY",
        "models": ["deepseek-flash", "deepseek-v4-pro"],
        "key_help": "get one at https://platform.deepseek.com",
    },
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_var": None,  # Ollama ignores the key, but the client needs some value
        "models": ["gpt-oss:20b"],
        "key_help": "",
    },
    "custom": {
        "base_url": os.environ.get("LLM_BASE_URL", ""),
        "key_var": "LLM_API_KEY",
        "models": [os.environ.get("LLM_MODEL", "")],
        "key_help": "and set LLM_BASE_URL and LLM_MODEL",
    },
}


def load_key(var, env_files=None):
    """Key from the environment, else from a .env file (repo root, or LLM_ENV_FILE)."""
    if os.environ.get(var):
        return os.environ[var].strip()
    if env_files is None:
        env_files = [REPO_DIR / ".env"]
        if os.environ.get("LLM_ENV_FILE"):
            env_files.append(Path(os.environ["LLM_ENV_FILE"]))
    for path in env_files:
        path = Path(path)
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.partition("=")
                if sep and key.strip() == var and value.strip():
                    return value.strip().strip("'\"")
    return None


def thinking_options(provider, thinking):
    """Request fields that switch reasoning on/off for each provider."""
    if thinking is None:
        return {}
    if provider == "deepseek":
        return {"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
    if provider == "custom":  # vLLM-style servers (Qwen3, Gemma, ...)
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}}
    raise ValueError(f"no thinking switch implemented for provider {provider!r}; "
                     "leave thinking=None (model default)")


@dataclass
class LLMReply:
    text: str
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    attempts: int = 1
    raw: object = field(default=None, repr=False)


class LLMClient:
    """One model from one provider, with retries on rate limits and outages.

    thinking: None = provider default, True/False = reasoning on/off.
    reasoning_effort: "low" / "high" / "max" for DeepSeek thinking mode.
    temperature: has no effect in DeepSeek thinking mode.
    """

    def __init__(self, provider, model=None, thinking=None, reasoning_effort=None,
                 temperature=None, max_tokens=4096, timeout=300, max_retries=8,
                 base_url=None, key=None, client=None):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)}")
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.model = model or cfg["models"][0]
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        thinking_options(provider, thinking)  # fail early on unsupported settings
        if client is None:
            from openai import OpenAI

            if cfg["key_var"]:
                key = key or load_key(cfg["key_var"])
                if not key:
                    raise RuntimeError(f"no API key: set {cfg['key_var']} or add it to .env "
                                       f"({cfg['key_help']})")
            client = OpenAI(api_key=key or "ollama", base_url=base_url or cfg["base_url"],
                            timeout=timeout, max_retries=0)
        self.client = client

    @property
    def label(self):
        mode = {None: "", True: "+thinking", False: "-thinking"}[self.thinking]
        return f"{self.provider}/{self.model}{mode}"

    def request_kwargs(self, messages):
        kwargs = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        kwargs.update(thinking_options(self.provider, self.thinking))
        return kwargs

    def chat(self, messages):
        """Send messages; retry 429s, server errors and timeouts with growing waits
        (2, 4, 8, ... s, max 120 s, plus jitter)."""
        kwargs = self.request_kwargs(messages)
        for attempt in range(1, self.max_retries + 2):
            start = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # the openai package raises several error classes
                if not _retryable(e) or attempt > self.max_retries:
                    raise
                wait = min(120, 2 ** attempt) + random.uniform(0, 1)
                print(f"[{self.label}] {type(e).__name__}; retry {attempt} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        return LLMReply(
            text=(msg.content or "").strip(),
            reasoning=(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""),
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            seconds=round(time.perf_counter() - start, 2),
            attempts=attempt,
            raw=resp,
        )


def _retryable(error):
    status = getattr(error, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return type(error).__name__ in {"APIConnectionError", "APITimeoutError",
                                    "RateLimitError", "InternalServerError"}


def list_models(provider, base_url=None, key=None):
    """Model ids the provider serves right now."""
    from openai import OpenAI

    cfg = PROVIDERS[provider]
    key = key or (load_key(cfg["key_var"]) if cfg["key_var"] else "ollama")
    client = OpenAI(api_key=key, base_url=base_url or cfg["base_url"], max_retries=2)
    return sorted(m.id for m in client.models.list().data)
