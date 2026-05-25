from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import openai

from utils.api_config import DEPRECATED_MODELS, PROVIDERS, detect_api_setup, get_client_config, resolve_model
from utils.http_client import create_openai_client
from utils.ssl_fix import configure_ssl

configure_ssl()

APIConnectionError = getattr(openai, "APIConnectionError", Exception)
AuthenticationError = getattr(openai, "AuthenticationError", Exception)
PermissionDeniedError = getattr(openai, "PermissionDeniedError", Exception)
InvalidRequestError = getattr(openai, "InvalidRequestError", Exception)
OpenAIError = getattr(openai, "OpenAIError", Exception)
RateLimitError = getattr(openai, "RateLimitError", Exception)

FORBIDDEN_MODELS = frozenset({"grok-beta", "grok-2-1212", "llama3-70b-8192", "llama3-8b-8192"})


def sanitize_model(provider: str, model: Optional[str]) -> str:
    resolved = resolve_model(provider, model)
    if resolved in FORBIDDEN_MODELS:
        resolved = DEPRECATED_MODELS.get(resolved, PROVIDERS[provider]["default_model"])
    if resolved not in PROVIDERS[provider]["models"]:
        resolved = PROVIDERS[provider]["default_model"]
    return resolved


def get_client(provider: str, api_key: str):
    config = get_client_config(provider, api_key)
    return create_openai_client(config["api_key"], config["base_url"]), config


def friendly_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "certificate verify failed" in text or "ssl" in text:
        return (
            "SSL connection failed. Restart the app after running: pip install truststore. "
            "If you use a corporate VPN/proxy, try disabling it briefly."
        )
    if "connection error" in text or "connect" in text and "refused" in text:
        return "Cannot reach the API server. Check your internet connection and firewall."
    if "403" in text or "permission" in text or "credits" in text or "licenses" in text:
        return (
            "Your xAI/Groq account has no API credits or permission. "
            "Add credits at https://console.x.ai or use a GROQ_API_KEY from https://console.groq.com"
        )
    if "401" in text or "authentication" in text or "invalid api key" in text:
        return "Invalid API key. Check GROK_API_KEY or GROQ_API_KEY in your .env file."
    if "model not found" in text:
        return str(exc)
    return str(exc)


def chat_completion(
    provider: str,
    api_key: str,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    client, config = get_client(provider, api_key)
    active_model = sanitize_model(provider, model or config["model"])

    try:
        response = client.chat.completions.create(
            model=active_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = response.choices[0].message.content.strip() if response.choices else ""
        return {"text": text, "model": active_model, "provider": config["provider_name"]}
    except PermissionDeniedError as exc:
        return {"error": "permission_denied", "message": friendly_error_message(exc), "model": active_model}
    except AuthenticationError as exc:
        return {"error": "authentication_error", "message": friendly_error_message(exc), "model": active_model}
    except RateLimitError as exc:
        return {"error": "rate_limit", "message": friendly_error_message(exc), "model": active_model}
    except APIConnectionError as exc:
        return {"error": "network_error", "message": friendly_error_message(exc), "model": active_model}
    except InvalidRequestError as exc:
        return {"error": "invalid_request", "message": friendly_error_message(exc), "model": active_model}
    except OpenAIError as exc:
        return {"error": "openai_error", "message": friendly_error_message(exc), "model": active_model}
    except Exception as exc:
        return {"error": "unexpected_error", "message": friendly_error_message(exc), "model": active_model}


def test_connection(provider: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    if not provider or not api_key:
        provider, api_key, message = detect_api_setup()
        if not provider or not api_key:
            return {"ok": False, "message": message}
    result = chat_completion(
        provider,
        api_key,
        [{"role": "user", "content": "Reply with exactly: OK"}],
        model=model,
        max_tokens=10,
    )
    if result.get("error"):
        return {"ok": False, "message": result.get("message"), "model": result.get("model")}
    return {"ok": True, "message": result.get("text", ""), "model": result.get("model")}
