from __future__ import annotations
import os
import logging
import re
from typing import Any, Dict, List, Optional
import openai
from sentence_transformers import SentenceTransformer
from utils.api_config import detect_api_setup, get_client_config
from utils.chunking import chunk_text

APIConnectionError = getattr(openai, "APIConnectionError", Exception)
AuthenticationError = getattr(openai, "AuthenticationError", Exception)
InvalidRequestError = getattr(openai, "InvalidRequestError", Exception)
OpenAIError = getattr(openai, "OpenAIError", Exception)
RateLimitError = getattr(openai, "RateLimitError", Exception)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_MODEL = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding_model() -> SentenceTransformer:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL
MAX_TOTAL_TOKENS = 8000
MAX_RESPONSE_TOKENS = 1200
MAX_INPUT_TOKENS = 6800
CHUNK_SIZE = 3000
CHUNK_OVERLAP = 300
PROMPT = (
    "You are an advanced AI content summarization assistant.\n\n"
    "Analyze the provided content carefully and generate:\n\n"
    "1. A concise summary (100 words)\n"
    "2. A detailed summary\n"
    "3. Important bullet points\n"
    "4. Key topics discussed\n"
    "5. Overall sentiment (Positive, Negative, Neutral)\n"
    "6. Important insights and takeaways\n\n"
    "Rules:\n"
    "- Keep summaries factual\n"
    "- Preserve important information\n"
    "- Avoid hallucinations\n"
    "- Use clean formatting\n"
    "- Highlight technical terms if present\n"
    "- If the content is lengthy, prioritize the most meaningful sections\n"
)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4.0))


def generate_embedding(text: str) -> List[float]:
    if not text:
        return []
    return get_embedding_model().encode(text).tolist()


def get_openai_client(api_key: str, base_url: str) -> Optional["openai.OpenAI"]:
    if not api_key:
        logger.error("Missing API key environment variable.")
        return None
    try:
        from utils.http_client import create_openai_client

        return create_openai_client(api_key, base_url)
    except Exception as exc:
        logger.exception("Failed to initialize OpenAI client: %s", exc)
        return None


def safe_extract_response(response: Any) -> str:
    if not response:
        return ""
    try:
        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"].strip()
        return response.choices[0].message["content"].strip()
    except Exception as exc:
        logger.exception("Failed to extract response text: %s", exc)
        return ""


def extract_list_items(value: str) -> List[str]:
    if not value:
        return []
    lines = [line.strip(" -•*\t") for line in value.splitlines() if line.strip()]
    items = [line for line in lines if line]
    if not items:
        parts = re.split(r"[\n;,]", value)
        items = [part.strip(" -•*") for part in parts if part.strip()]
    return items


def parse_response(message: str) -> Dict[str, str]:
    sections = {
        "concise_summary": "",
        "detailed_summary": "",
        "bullet_points": "",
        "key_topics": "",
        "sentiment": "",
        "insights": "",
        "title": "AI Content Summary",
    }
    current_key = None
    for line in message.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("1.") or "concise summary" in lower:
            current_key = "concise_summary"
            sections[current_key] += re.sub(r"^1\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if lower.startswith("2.") or "detailed summary" in lower:
            current_key = "detailed_summary"
            sections[current_key] += re.sub(r"^2\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if lower.startswith("3.") or "bullet points" in lower:
            current_key = "bullet_points"
            sections[current_key] += re.sub(r"^3\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if lower.startswith("4.") or "key topics" in lower:
            current_key = "key_topics"
            sections[current_key] += re.sub(r"^4\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if lower.startswith("5.") or "overall sentiment" in lower:
            current_key = "sentiment"
            sections[current_key] += re.sub(r"^5\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if lower.startswith("6.") or "important insights" in lower:
            current_key = "insights"
            sections[current_key] += re.sub(r"^6\.\s*", "", line, flags=re.I).strip() + "\n"
            continue
        if current_key:
            sections[current_key] += line + "\n"
    for key, value in sections.items():
        sections[key] = value.strip()
    return sections


def structure_summary(parsed: Dict[str, str]) -> Dict[str, Any]:
    bullet_points_list = extract_list_items(parsed.get("bullet_points", ""))
    if not bullet_points_list and parsed.get("detailed_summary"):
        bullet_points_list = extract_list_items(parsed.get("detailed_summary", ""))[:8]

    topics_list = extract_list_items(parsed.get("key_topics", ""))
    if not topics_list and parsed.get("detailed_summary"):
        topics_list = extract_list_items(parsed.get("detailed_summary", ""))[:6]

    return {
        "short_summary": parsed.get("concise_summary", ""),
        "detailed_summary": parsed.get("detailed_summary", ""),
        "bullet_points": parsed.get("bullet_points", ""),
        "bullet_points_list": bullet_points_list,
        "key_topics": parsed.get("key_topics", ""),
        "topics_list": topics_list,
        "sentiment": parsed.get("sentiment", "Neutral"),
        "insights": parsed.get("insights", ""),
        **parsed,
    }


def call_chat_api(
    client: "openai.OpenAI",
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> Dict[str, Any]:
    if not client:
        return {"error": "client_unavailable", "message": "API client is not initialized."}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        logger.info("Chat API response received for model %s", model)

        return {"response": response}
    except AuthenticationError as exc:
        logger.error("Authentication error: %s", exc)
        print(exc)
        return {"error": "authentication_error", "message": str(exc)}
    except RateLimitError as exc:
        logger.error("Rate limit error: %s", exc)
        print(exc)
        return {"error": "rate_limit", "message": str(exc)}
    except APIConnectionError as exc:
        logger.error("Network error: %s", exc)
        print(exc)
        return {"error": "network_error", "message": str(exc)}
    except InvalidRequestError as exc:
        logger.error("Invalid request: %s", exc)
        print(exc)
        return {"error": "invalid_request", "message": str(exc)}
    except OpenAIError as exc:
        logger.error("OpenAI error: %s", exc)
        print(exc)
        return {"error": "openai_error", "message": str(exc)}
    except Exception as exc:
        logger.exception("Unexpected error calling chat API: %s", exc)
        print(exc)
        return {"error": "unexpected_error", "message": str(exc)}


def summarize_content(
    client: "openai.OpenAI",
    model: str,
    content: str,
    chunk_index: int = 1,
    total_chunks: int = 1,
) -> Dict[str, Any]:
    trimmed = content.strip()
    if not trimmed:
        return {}
    instructions = (
        f"Summarize the provided content chunk {chunk_index} of {total_chunks}. "
        "Follow the required output format and keep the response factual."
    )
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{instructions}\n\nCONTENT:\n{trimmed}"},
    ]
    result = call_chat_api(client, model, messages)
    if result.get("error"):
        return {"error": result["error"], "message": result.get("message", "")}
    response = result.get("response")
    ai_text = safe_extract_response(response)
    if not ai_text:
        return {"error": "empty_response", "message": "The model returned an empty response."}
    parsed = parse_response(ai_text)
    return structure_summary(parsed)


def merge_chunk_summaries(
    client: "openai.OpenAI", model: str, summaries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    if len(summaries) == 1:
        return summaries[0]
    if len(summaries) > 6:
        merged_groups = []
        for i in range(0, len(summaries), 6):
            group = summaries[i : i + 6]
            merged_groups.append(merge_chunk_summaries(client, model, group))
        return merge_chunk_summaries(client, model, merged_groups)

    merged_parts = []
    for index, summary in enumerate(summaries, start=1):
        merged_parts.append(
            f"Chunk {index} short summary:\n{summary.get('short_summary', '')}\n"
            f"Detailed summary:\n{summary.get('detailed_summary', '')}\n"
            f"Bullet points:\n{summary.get('bullet_points', '')}\n"
            f"Key topics:\n{summary.get('key_topics', '')}\n"
            f"Insights:\n{summary.get('insights', '')}\n"
        )

    combine_prompt = (
        "Combine the following chunk summaries into a final cohesive summary. "
        "Keep the same output structure: concise summary, detailed summary, bullet points, key topics, sentiment, and insights."
    )
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"{combine_prompt}\n\n" + "\n\n".join(merged_parts)},
    ]
    result = call_chat_api(client, model, messages)
    if result.get("error"):
        logger.error("Failed to merge chunk summaries: %s", result.get("message"))
        return fallback_merge_summaries(summaries)
    response = result.get("response")
    ai_text = safe_extract_response(response)
    if not ai_text:
        return fallback_merge_summaries(summaries)
    parsed = parse_response(ai_text)
    return structure_summary(parsed)


def fallback_merge_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    bullet_points = []
    topics = []
    insights = []
    details = []
    for summary in summaries:
        bullet_points.extend(summary.get("bullet_points_list", []))
        topics.extend(summary.get("topics_list", []))
        if summary.get("insights"):
            insights.append(summary.get("insights"))
        if summary.get("detailed_summary"):
            details.append(summary.get("detailed_summary"))
    return {
        "short_summary": summaries[0].get("short_summary", ""),
        "detailed_summary": "\n\n".join(details),
        "bullet_points": "\n".join(dict.fromkeys(bullet_points)),
        "bullet_points_list": list(dict.fromkeys(bullet_points)),
        "key_topics": ", ".join(dict.fromkeys(topics)),
        "topics_list": list(dict.fromkeys(topics)),
        "sentiment": summaries[-1].get("sentiment", "Neutral"),
        "insights": "\n\n".join(insights),
    }


def generate_ai_summary(
    content: str,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if api_key and provider:
        try:
            client_settings = get_client_config(provider, api_key, model)
        except ValueError as exc:
            return {"error": "invalid_provider", "message": str(exc)}
    else:
        detected_provider, detected_key, message = detect_api_setup()
        if not detected_provider or not detected_key:
            logger.error(message)
            return {"error": "missing_api_key", "message": message}
        api_key = detected_key
        provider = detected_provider
        client_settings = get_client_config(provider, api_key, model)

    content = content.strip()
    if not content:
        return {"error": "empty_content", "message": "No content was provided for summarization."}

    client = get_openai_client(client_settings["api_key"], client_settings["base_url"])
    if not client:
        return {
            "error": "client_init_failed",
            "message": f"Unable to initialize {client_settings['provider_name']} API client.",
        }

    active_model = client_settings["model"]
    token_count = estimate_tokens(content)
    if token_count > MAX_INPUT_TOKENS:
        chunks = chunk_text(content, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        if not chunks:
            return {"error": "chunking_failed", "message": "Unable to split content into chunks."}
        chunk_summaries = []
        for index, chunk in enumerate(chunks, start=1):
            summary = summarize_content(client, active_model, chunk, index, len(chunks))
            if summary.get("error"):
                logger.error("Chunk summarization failed for chunk %s: %s", index, summary.get("message"))
                continue
            chunk_summaries.append(summary)
        if not chunk_summaries:
            return {"error": "chunk_summarization_failed", "message": "All chunk summarization calls failed."}
        final_summary = merge_chunk_summaries(client, active_model, chunk_summaries)
        final_summary["model_used"] = active_model
        final_summary["provider"] = client_settings["provider_name"]
        return final_summary

    summary = summarize_content(client, active_model, content)
    if summary.get("error"):
        logger.error("Summarization failed: %s", summary.get("message"))
        return summary
    summary["model_used"] = active_model
    summary["provider"] = client_settings["provider_name"]
    return summary
