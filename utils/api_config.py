from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

PROVIDERS: Dict[str, Dict] = {
    "xai": {
        "name": "xAI (Grok)",
        "base_url": "https://api.x.ai/v1",
        "env_keys": ("GROK_API_KEY", "XAI_API_KEY"),
        "default_model": "grok-3-mini",
        "models": (
            "grok-3-mini",
            "grok-4.3",
            "grok-4.20-0309-non-reasoning",
            "grok-build-0.1",
        ),
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "env_keys": ("GROQ_API_KEY",),
        "default_model": "llama-3.3-70b-versatile",
        "models": (
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ),
    },
}

DEPRECATED_MODELS = {
    "grok-beta": "grok-3-mini",
    "grok-2-1212": "grok-3-mini",
    "llama3-70b-8192": "llama-3.3-70b-versatile",
    "llama3-8b-8192": "llama-3.1-8b-instant",
}


def _key_prefix_provider(api_key: str) -> Optional[str]:
    if not api_key:
        return None
    lowered = api_key.strip().lower()
    if lowered.startswith("xai-"):
        return "xai"
    if lowered.startswith("gsk_"):
        return "groq"
    return None


def resolve_model(provider: str, model: Optional[str] = None) -> str:
    config = PROVIDERS[provider]
    chosen = (model or os.getenv("SUMMARY_MODEL") or config["default_model"]).strip()
    return DEPRECATED_MODELS.get(chosen, chosen)


def get_provider_models(provider: str) -> List[str]:
    return list(PROVIDERS[provider]["models"])


def detect_api_setup() -> Tuple[Optional[str], Optional[str], str]:
    """Return (provider_id, api_key, message)."""
    return resolve_credentials()


def resolve_credentials(
    session_groq_key: Optional[str] = None,
    provider_choice: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str]:
    """Pick API provider and key. Groq is preferred when available (free tier)."""
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    xai_key = (os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip()

    if session_groq_key and session_groq_key.strip():
        groq_key = session_groq_key.strip()

    env_prefer = (os.getenv("API_PROVIDER") or os.getenv("PREFER_GROQ", "")).strip().lower()
    if env_prefer in ("1", "true", "yes", "groq"):
        provider_choice = "groq"
    elif env_prefer == "xai":
        provider_choice = "xai"

    choice = (provider_choice or "").strip().lower()
    if choice in ("groq", "groq (free)"):
        if groq_key:
            return "groq", groq_key, ""
        return None, None, "Groq selected but no GROQ_API_KEY. Paste a key in the sidebar or add it to .env."
    if choice in ("xai", "xai (grok)"):
        if xai_key:
            return "xai", xai_key, ""
        return None, None, "xAI selected but no GROK_API_KEY in .env."

    if groq_key:
        return "groq", groq_key, ""
    if xai_key:
        return "xai", xai_key, ""

    return None, None, "No API key found. Set GROQ_API_KEY (free) or GROK_API_KEY in .env."


def get_client_config(provider: str, api_key: str, model: Optional[str] = None) -> Dict[str, str]:
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    config = PROVIDERS[provider]
    resolved_model = resolve_model(provider, model)
    if resolved_model in DEPRECATED_MODELS:
        resolved_model = DEPRECATED_MODELS[resolved_model]
    if resolved_model not in config["models"]:
        resolved_model = config["default_model"]
    return {
        "provider": provider,
        "provider_name": config["name"],
        "api_key": api_key,
        "base_url": config["base_url"],
        "model": resolved_model,
    }
