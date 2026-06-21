"""
LLM client abstraction.

Controlled entirely by env vars — no code change needed to switch backends:

  LLM_PROVIDER=ollama      (default) → talks to Ollama
  LLM_PROVIDER=claude               → talks to Anthropic Claude API
  LLM_PROVIDER=openai               → talks to any OpenAI-compatible server

Ollama env vars:
  OLLAMA_HOST      default: http://localhost:11434
                   change to http://<your-server-ip>:11434 to use a remote machine
  OLLAMA_MODEL     default: llama3

Claude env vars:
  ANTHROPIC_API_KEY   required
  CLAUDE_MODEL        default: claude-haiku-4-5-20251001

OpenAI-compatible env vars (works with vLLM, LM Studio, Together, Groq, etc.):
  OPENAI_API_KEY      required
  OPENAI_BASE_URL     e.g. http://<server>:8000/v1
  OPENAI_MODEL        e.g. mistral-7b
"""
import os

_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def chat(prompt: str, max_tokens: int = 8192) -> str:
    """Send a prompt and return the response string. Provider-agnostic."""
    if _PROVIDER == "ollama":
        return _ollama_chat(prompt, max_tokens)
    if _PROVIDER == "claude":
        return _claude_chat(prompt, max_tokens)
    if _PROVIDER == "openai":
        return _openai_chat(prompt, max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER: {_PROVIDER!r}. Choose ollama | claude | openai")


# ---------------------------------------------------------------------------
# Ollama — local or remote server
# ---------------------------------------------------------------------------
def _ollama_chat(prompt: str, max_tokens: int) -> str:
    import ollama

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": max_tokens},
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Claude (Anthropic API)
# ---------------------------------------------------------------------------
def _claude_chat(prompt: str, max_tokens: int) -> str:
    import anthropic

    model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    msg = client.messages.create(
        model=model,
        max_tokens=min(max_tokens, 8096),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------------------
# OpenAI-compatible (vLLM, LM Studio, Together AI, Groq, etc.)
# ---------------------------------------------------------------------------
def _openai_chat(prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    )
    model = os.getenv("OPENAI_MODEL", "default")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
