"""
Hosted answer generation for the maize advisory chat
====================================================

The advisory pipeline used to write answers with a local
``Qwen2.5-1.5B-Instruct``. That needs about 3 GB for the weights alone, and a
further 2.4 GB for the NLLB translation model, which is well over the ~2.7 GB
that Streamlit Community Cloud gives an app. The cloud server also has only
about two CPU cores, so a local 1.5B model takes minutes to write one answer.

This module keeps the language model *off* the app server: the pipeline sends
the grounded prompt to a hosted provider and receives the answer as text. Groq
is the default because its free tier is fast and needs no card. Google Gemini
and the Hugging Face Inference Providers router speak the same OpenAI-style
protocol and are supported as alternatives.

Configuration (environment variables, or the same keys in Streamlit secrets):

``ADVISORY_LLM_PROVIDER``
    ``auto`` (default), ``groq``, ``gemini`` or ``hf``. ``auto`` picks the first
    provider that has a key, preferring Groq, then Gemini. Hugging Face is
    opt-in (``ADVISORY_LLM_PROVIDER=hf``) because it needs inference credits.
``GROQ_API_KEY`` / ``GROQ_MODEL``
    Free key from https://console.groq.com/keys. Default model
    ``qwen/qwen3.8-27b``.
``GEMINI_API_KEY`` / ``GEMINI_MODEL``
    Free key from https://aistudio.google.com/apikey. Default model
    ``gemini-2.5-flash``.
``HF_TOKEN`` / ``ADVISORY_HF_MODEL``
    Hugging Face Inference Providers. Note that Hugging Face now requires
    pre-paid inference credits even on the free plan, so this is the fallback
    for accounts that have them.

The module uses only the standard library, so importing it costs nothing and it
does not pull another dependency into the Streamlit image.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

#: Environment keys that may also arrive through ``st.secrets`` on Community
#: Cloud, where secrets are not exported as environment variables.
SECRET_ENV_KEYS = (
    "ADVISORY_LLM_PROVIDER",
    "ADVISORY_HF_MODEL",
    "ADVISORY_LLM_TIMEOUT",
    "ADVISORY_TRANSLATION_BACKEND",
    "ADVISORY_CT2_TRANSLATION_MODEL",
    "ADVISORY_TRANSLATION_BEAM",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GOOGLE_TRANSLATE_API_KEY",
    "HF_TOKEN",
)

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"

#: Sent on every request; a bare urllib user agent is blocked by some CDNs.
USER_AGENT = "maize-advisory/1.0 (+https://github.com/WillardDev/Farmers-Assistant-RAG-)"

PROVIDER_LABELS = {"groq": "Groq", "gemini": "Google Gemini", "hf": "Hugging Face"}

#: Short, farmer-readable reason shown when no provider is configured.
SETUP_HINT = (
    "No hosted answer model is configured. Add a free GROQ_API_KEY "
    "(https://console.groq.com/keys) or GEMINI_API_KEY "
    "(https://aistudio.google.com/apikey) to the app secrets, then reboot the app."
)


class HostedModelError(RuntimeError):
    """Raised when the hosted provider cannot answer, with a human hint."""


def _hydrate_from_streamlit_secrets() -> None:
    """Copy advisory/LLM secrets into ``os.environ`` when running on Cloud.

    Local runs use ``.env`` (loaded by ``rag_pipeline``). Community Cloud keeps
    secrets in ``st.secrets`` instead, and they are not environment variables,
    so a correctly filled-in dashboard would otherwise look unconfigured. A real
    environment variable always wins. Top-level keys and an optional
    ``[advisory]`` / ``[llm]`` / ``[api]`` table are both accepted.
    """
    try:
        import streamlit as st

        secrets = st.secrets
    except Exception:  # streamlit not installed, or no secrets.toml on this host
        return

    def section(name: str):
        try:
            block = secrets.get(name)
        except Exception:
            return None
        return block if hasattr(block, "get") else None

    for key in SECRET_ENV_KEYS:
        if os.environ.get(key) is not None:
            continue
        value = None
        try:
            value = secrets.get(key)
        except Exception:
            value = None
        if value is None:
            for name in ("advisory", "llm", "api"):
                block = section(name)
                if block and key in block:
                    value = block[key]
                    break
        if value is not None:
            os.environ[key] = str(value)


_hydrate_from_streamlit_secrets()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def resolve_provider() -> str:
    """Return the provider to use, or ``""`` when none is configured.

    ``auto`` deliberately ignores ``HF_TOKEN``: a Hugging Face token is also
    used for downloading public models, and Hugging Face Inference Providers
    need pre-paid credits even on free accounts. Opt in to that path with
    ``ADVISORY_LLM_PROVIDER=hf`` so a plain token does not turn into a
    confusing "no credits" error.
    """
    requested = _env("ADVISORY_LLM_PROVIDER", "auto").lower()
    if requested in PROVIDER_LABELS:
        return requested
    if _env("GROQ_API_KEY"):
        return "groq"
    if _env("GEMINI_API_KEY"):
        return "gemini"
    return ""


def model_name(provider: str) -> str:
    if provider == "groq":
        return _env("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if provider == "gemini":
        return _env("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    if provider == "hf":
        return _env("ADVISORY_HF_MODEL", DEFAULT_HF_MODEL)
    return ""


def provider_label() -> str:
    """``"Groq · qwen/qwen3.8-27b"`` for the UI, or ``""``."""
    provider = resolve_provider()
    if not provider:
        return ""
    return f"{PROVIDER_LABELS[provider]} · {model_name(provider)}"


def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: int | None = None) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    # Some providers sit behind a CDN that rejects the default "Python-urllib"
    # signature with HTTP 403 before the API key is even checked.
    request.add_header("User-Agent", USER_AGENT)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(
                request, timeout=timeout or int(_env("ADVISORY_LLM_TIMEOUT", "120"))) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise HostedModelError(_explain_http_error(exc.code, detail)) from exc
    except urllib.error.URLError as exc:
        raise HostedModelError(
            "Could not reach the hosted answer model. Check the app's internet "
            f"access and try again ({exc.reason}).") from exc
    except TimeoutError as exc:
        raise HostedModelError(
            "The hosted answer model took too long to reply. Try again, or set "
            "ADVISORY_LLM_TIMEOUT to a larger number of seconds.") from exc


def _explain_http_error(status: int, detail: str) -> str:
    if status in (401, 403):
        return ("The hosted answer model rejected the API key. Re-check the key "
                "in the app secrets and reboot the app. (" + detail + ")")
    if status == 402:
        return ("The Hugging Face account has no Inference Providers credits. "
                "Use a free GROQ_API_KEY or GEMINI_API_KEY instead. (" + detail + ")")
    if status == 429:
        return ("The hosted answer model is rate-limited right now. Wait a minute "
                "and try again, or configure a second provider. (" + detail + ")")
    if status in (400, 404) and ("not supported" in detail.lower()
                                 or "does not exist" in detail.lower()
                                 or "model_not_found" in detail.lower()):
        return ("The selected model is not available from your provider account. "
                "Use the default GROQ_MODEL/GEMINI_MODEL, or set a model your "
                "account can access. (" + detail + ")")
    if status >= 500:
        return ("The hosted answer model is temporarily unavailable. Try again "
                "shortly. (" + detail + ")")
    return f"The hosted answer model returned HTTP {status}: {detail}"


def _openai_chat(url: str, api_key: str, model: str, prompt: str,
                 max_tokens: int, temperature: float,
                 extra_payload: dict | None = None) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "stream": False,
    }
    payload.update(extra_payload or {"max_tokens": max_tokens})
    data = _post_json(url, payload, {"Authorization": f"Bearer {api_key}"})
    try:
        choices = data["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise HostedModelError(
            f"The hosted provider returned an unexpected response: {str(data)[:300]}") from exc
    if not content or not content.strip():
        raise HostedModelError(
            "The hosted answer model returned an empty answer. Try again.")
    return content.strip()


def _groq_chat(prompt: str, max_tokens: int, temperature: float) -> str:
    api_key = _env("GROQ_API_KEY")
    if not api_key:
        raise HostedModelError(SETUP_HINT)
    model = model_name("groq")
    extra = {"max_tokens": max_tokens}
    if "gpt-oss" in model:
        # Reasoning tokens count towards the completion budget on Groq, so
        # leave headroom and ask for brief reasoning: this is grounded
        # extraction from the retrieved context, not open-ended problem solving.
        extra = {"max_completion_tokens": max_tokens + 300,
                 "reasoning_effort": "low"}
    return _openai_chat("https://api.groq.com/openai/v1/chat/completions",
                        api_key, model, prompt, max_tokens, temperature,
                        extra_payload=extra)


def _hf_chat(prompt: str, max_tokens: int, temperature: float) -> str:
    token = _env("HF_TOKEN") or _env("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise HostedModelError(SETUP_HINT)
    return _openai_chat("https://router.huggingface.co/v1/chat/completions",
                        token, model_name("hf"), prompt, max_tokens, temperature)


def _gemini_chat(prompt: str, max_tokens: int, temperature: float) -> str:
    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        raise HostedModelError(SETUP_HINT)
    model = model_name("gemini")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
           f":generateContent?key={api_key}")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature,
                             "maxOutputTokens": max_tokens},
    }
    data = _post_json(url, payload)
    try:
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        raise HostedModelError(
            f"{PROVIDER_LABELS['gemini']} returned an unexpected response: "
            f"{str(data)[:300]}") from exc
    if not content.strip():
        raise HostedModelError(
            f"{PROVIDER_LABELS['gemini']} returned an empty answer. Try again.")
    return content.strip()


_CHAT_FUNCTIONS = {"groq": _groq_chat, "gemini": _gemini_chat, "hf": _hf_chat}


def chat(prompt: str, max_tokens: int = 400, temperature: float = 0.0) -> str:
    """Send one grounded prompt to the configured provider and return the text."""
    provider = resolve_provider()
    if not provider:
        raise HostedModelError(SETUP_HINT)
    return _CHAT_FUNCTIONS[provider](prompt, max_tokens, temperature)


_LANGUAGE_NAMES = {"en": "English", "sw": "Kiswahili", "luo": "Dholuo"}

_TRANSLATE_PROMPT = """Translate the text below from {source} to {target}.
Return only the translation - no notes, no quotes, no explanations.
Keep names, numbers and units unchanged.

Text:
{text}"""


def translate_with_llm(text: str, source_lang: str, target_lang: str) -> str:
    """Fallback translation through the chat model when NLLB is not available."""
    prompt = _TRANSLATE_PROMPT.format(
        source=_LANGUAGE_NAMES.get(source_lang, source_lang),
        target=_LANGUAGE_NAMES.get(target_lang, target_lang),
        text=text,
    )
    return chat(prompt, max_tokens=512).strip()


def google_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate through the Google Cloud Translation v2 REST API.

    Google's language list does include Luo (``luo``), so this is a fully
    hosted alternative to the local NLLB model. It needs a
    ``GOOGLE_TRANSLATE_API_KEY`` with the Cloud Translation API enabled.
    """
    api_key = _env("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        raise HostedModelError(
            "GOOGLE_TRANSLATE_API_KEY is not set, so hosted translation is "
            "unavailable.")
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    payload = {
        "q": [text],
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    data = _post_json(url, payload)
    try:
        return data["data"]["translations"][0]["translatedText"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise HostedModelError(
            f"Google Translate returned an unexpected response: {str(data)[:300]}") from exc
