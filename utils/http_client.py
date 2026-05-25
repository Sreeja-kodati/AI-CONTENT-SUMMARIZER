from __future__ import annotations

import httpx
import openai

from utils.ssl_fix import configure_ssl


def create_openai_client(api_key: str, base_url: str, timeout_seconds: float = 120.0) -> openai.OpenAI:
    configure_ssl()
    timeout = httpx.Timeout(timeout_seconds, connect=30.0)
    http_client = httpx.Client(timeout=timeout)
    return openai.OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
