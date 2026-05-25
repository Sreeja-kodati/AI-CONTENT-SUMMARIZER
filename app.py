import os
import time
import json
import streamlit as st
from dotenv import load_dotenv

from utils.ssl_fix import configure_ssl

configure_ssl()
from utils.api_config import (
    DEPRECATED_MODELS,
    get_client_config,
    get_provider_models,
    resolve_credentials,
    resolve_model,
)
from utils.file_parser import extract_text_from_pdf, extract_text_from_docx, extract_text_from_url, clean_text
from utils.chunking import chunk_text
from utils.pinecone_db import PineconeClient
from utils.summarizer import generate_embedding, generate_ai_summary
from utils.sentiment import analyze_sentiment
from utils.llm_client import test_connection
from utils.helpers import get_word_count, get_reading_time, safe_filename

load_dotenv()

APP_VERSION = "2.1.0"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "summary-index")

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        body { background-color: #0e1117; color: #e7edf5; }
        .main { background-color: #111827; }
        .stButton>button { background-color: #2563eb; color: white; border: none; }
        .stButton>button:hover { background-color: #1d4ed8; }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            background-color: #1f2937; color: #e2e8f0;
        }
        .stSelectbox>div>div>div>div { background-color: #1f2937; color: #e2e8f0; }
        .stDownloadButton>button { background-color: #059669; }
        .stDownloadButton>button:hover { background-color: #047857; }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        .result-card {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
        }
        .result-card h3 { margin-top: 0; color: #f8fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def is_no_credits_error(message: str) -> bool:
    lowered = (message or "").lower()
    return "403" in lowered or "credits" in lowered or "licenses" in lowered or "permission" in lowered


def render_no_credits_help():
    st.error("Your xAI account has no API credits yet.")
    st.markdown(
        """
        **Fix (choose one):**

        1. **Free option — Groq (recommended)**  
           - Go to [console.groq.com](https://console.groq.com) and create a free API key (`gsk_...`)  
           - Paste it in the sidebar under **Groq API key**, select **Groq (free)**, then click **Test API connection**

        2. **Paid option — xAI**  
           - Add credits for your team at [console.x.ai](https://console.x.ai)  
           - Or open your team billing link from the error message

        3. **Permanent setup** — add to `.env` and restart the app:
        ```
        GROQ_API_KEY=gsk_your_key_here
        API_PROVIDER=groq
        ```
        (You can comment out or remove `GROK_API_KEY` if you only use Groq.)
        """
    )


def render_sidebar():
    st.sidebar.header("AI Content Summarizer")
    st.sidebar.caption(f"Build **v{APP_VERSION}** — uses xAI/Groq with current models only.")

    for key in list(st.session_state.keys()):
        if "grok-beta" in str(st.session_state.get(key, "")):
            del st.session_state[key]

    has_xai = bool((os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or "").strip())
    has_groq_env = bool((os.getenv("GROQ_API_KEY") or "").strip())

    if has_xai and not has_groq_env:
        st.sidebar.warning(
            "xAI key detected but your team has **no credits**. Use a free **Groq** key below."
        )

    session_groq = st.sidebar.text_input(
        "Groq API key (free)",
        type="password",
        placeholder="gsk_... from console.groq.com",
        help="Paste a Groq key to avoid xAI billing. Saved for this browser session only.",
        key="session_groq_key",
    )

    default_provider_index = 0 if (session_groq.strip() or has_groq_env) else (1 if has_xai else 0)
    provider_label = st.sidebar.radio(
        "API provider",
        ["Groq (free)", "xAI (Grok)"],
        index=default_provider_index,
        help="Groq has a generous free tier and works without xAI credits.",
    )
    provider_choice = "groq" if provider_label.startswith("Groq") else "xai"

    provider, api_key, cred_message = resolve_credentials(
        session_groq_key=session_groq,
        provider_choice=provider_choice,
    )
    if not provider or not api_key:
        st.sidebar.error(cred_message)
        return None, None, None, None, None, None, None, None, cred_message

    client_config = get_client_config(provider, api_key)
    st.sidebar.success(f"Using **{client_config['provider_name']}** — model `{client_config['model']}`")

    models = get_provider_models(provider)
    default_model = resolve_model(provider)
    default_index = models.index(default_model) if default_model in models else 0
    selected_model = st.sidebar.selectbox(
        "AI model",
        options=models,
        index=default_index,
        key="ai_model_v2",
        help="Never use grok-beta — it was retired. Default is grok-3-mini.",
    )
    selected_model = resolve_model(provider, selected_model)

    if st.sidebar.button("Test API connection"):
        with st.spinner("Testing API..."):
            test_result = test_connection(
                provider=provider,
                api_key=api_key,
                model=selected_model,
            )
        if test_result.get("ok"):
            st.sidebar.success(f"API OK — model `{test_result.get('model')}`")
        else:
            msg = test_result.get("message", "API test failed")
            st.sidebar.error(msg)
            if is_no_credits_error(msg):
                st.sidebar.info("Switch to **Groq (free)** and paste a `gsk_` key above.")

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        help="Upload a PDF or Word document for summarization.",
    )
    url_input = st.sidebar.text_input(
        "Paste article / blog URL",
        placeholder="https://example.com/article",
    )
    raw_text = st.sidebar.text_area(
        "Paste long text",
        height=160,
        help="Paste text directly if you are not uploading a file.",
    )
    top_k = st.sidebar.slider("Relevant chunks to retrieve", min_value=1, max_value=10, value=5)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quick tips**")
    st.sidebar.write("1. Add your content on the left.")
    st.sidebar.write("2. Pick a supported model above.")
    st.sidebar.write("3. Click **Analyze Content**.")
    submit_button = st.sidebar.button("Analyze Content", type="primary")
    return (
        provider,
        api_key,
        uploaded_file,
        url_input,
        raw_text,
        top_k,
        submit_button,
        selected_model,
        "",
    )


def display_metrics(word_count: int, reading_time: str, chunks: int, relevant_chunks: int):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Word count", f"{word_count}")
    col2.metric("Estimated reading time", reading_time)
    col3.metric("Document chunks", f"{chunks}")
    col4.metric("Retrieved chunks", f"{relevant_chunks}")


def get_summary_text(summary_data: dict, keys: list[str], separator: str = "\n") -> str:
    for key in keys:
        value = summary_data.get(key)
        if isinstance(value, list):
            return separator.join(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_download_text(summary_data: dict) -> tuple[str, str]:
    text = [f"Title: {summary_data.get('title', 'AI Content Summary')}\n"]
    text.append(f"Concise Summary:\n{summary_data.get('concise_summary', '')}\n")
    text.append(f"Detailed Summary:\n{summary_data.get('detailed_summary', '')}\n")
    text.append(f"Bullet Points:\n{summary_data.get('bullet_points', '')}\n")
    text.append(f"Key Topics:\n{summary_data.get('key_topics', '')}\n")
    text.append(f"Sentiment:\n{summary_data.get('sentiment', '')}\n")
    text.append(f"Insights:\n{summary_data.get('insights', '')}\n")
    output = "\n".join(text)
    return output, safe_filename("summary.txt")


def format_api_error(message: str, model: str) -> str:
    if is_no_credits_error(message):
        return (
            "xAI returned 403: your team has no credits. "
            "Use Groq (free) in the sidebar, or add credits at console.x.ai."
        )
    lowered = (message or "").lower()
    if "model not found" in lowered or "decommissioned" in lowered:
        replacement = DEPRECATED_MODELS.get(model)
        hint = f" Try `{replacement}` instead." if replacement else ""
        return f"The selected model is unavailable.{hint} Choose another model in the sidebar."
    if "unavailable" in lowered:
        return "Requested model is unavailable. Please select a different model in the sidebar."
    return message or "The AI could not generate a summary. Please try again."


def render_output(summary_data: dict, stats: dict):
    if not summary_data:
        st.warning("No summary data available. Please provide content and click Analyze.")
        return

    st.markdown(f"## {summary_data.get('title', 'AI Content Summary')}")
    display_metrics(
        stats["word_count"],
        stats["reading_time"],
        stats["chunk_count"],
        stats["retrieved_count"],
    )

    concise_summary = get_summary_text(summary_data, ["concise_summary", "short_summary"])
    detailed_summary = get_summary_text(summary_data, ["detailed_summary"])
    bullet_points_text = get_summary_text(summary_data, ["bullet_points"])
    bullet_points_list = summary_data.get("bullet_points_list", [])
    if not bullet_points_text and bullet_points_list:
        bullet_points_text = "\n".join(bullet_points_list)
    topics_text = get_summary_text(summary_data, ["key_topics"])
    topics_list = summary_data.get("topics_list", [])
    if not topics_text and topics_list:
        topics_text = ", ".join(topics_list)
    sentiment = summary_data.get("sentiment", "Neutral") or "Neutral"
    insights = summary_data.get("insights", "")

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Summary")
    if concise_summary:
        st.markdown("**Concise**")
        st.write(concise_summary)
    if detailed_summary:
        st.markdown("**Detailed**")
        st.write(detailed_summary)
    if not concise_summary and not detailed_summary:
        st.write("No summary was generated.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Key Points")
    if bullet_points_text:
        for line in bullet_points_text.split("\n"):
            if line.strip():
                st.markdown(f"1. {line.strip().lstrip('-•1234567890. ')}")
    else:
        st.write("No key points were generated.")
    if topics_text:
        st.markdown("**Key topics**")
        st.write(topics_text)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 💭 Sentiment Analysis")
    st.write(f"**AI sentiment:** {sentiment}")
    if stats.get("lexical_sentiment"):
        st.write(f"**Lexical sentiment (TextBlob):** {stats.get('lexical_sentiment')}")
    st.markdown("</div>", unsafe_allow_html=True)

    if insights:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Insights")
        st.write(insights)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown("### 💾 Export")
    download_text, file_name = build_download_text(summary_data)
    st.download_button(
        label="Download summary as TXT",
        data=download_text,
        file_name=file_name,
        mime="text/plain",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Processing statistics"):
        st.write(f"**Chunks created:** {stats['chunk_count']}")
        st.write(f"**Relevant chunks retrieved:** {stats['retrieved_count']}")
        st.write(f"**Text length:** {stats['char_count']} characters")
        st.write(f"**Input source:** {stats['source']}")
        st.write(f"**Provider:** {summary_data.get('provider', stats.get('provider', 'Unknown'))}")
        st.write(f"**Model used:** {summary_data.get('model_used', stats.get('model_used', 'Unknown'))}")


def main():
    inject_css()
    st.title("AI-Powered Content Summarizer")
    st.caption(f"Version {APP_VERSION} — if you still see `grok-beta` errors, stop all Streamlit windows and run `run_app.ps1`.")
    st.write(
        "Summarize documents, articles, and long text with AI — including key points, topics, and sentiment."
    )

    sidebar_result = render_sidebar()
    if sidebar_result[-1]:
        st.error(sidebar_result[-1])
        st.info(
            "Get a **free** Groq key at https://console.groq.com (starts with `gsk_`), "
            "paste it in the sidebar, or add `GROQ_API_KEY=...` to `.env`."
        )
        return

    (
        provider,
        api_key,
        uploaded_file,
        url_input,
        raw_text,
        top_k,
        submit_button,
        selected_model,
        _,
    ) = sidebar_result

    client_config = get_client_config(provider, api_key)

    stats = {
        "word_count": 0,
        "reading_time": "0 min",
        "chunk_count": 0,
        "retrieved_count": 0,
        "char_count": 0,
        "source": "None",
        "lexical_sentiment": "Unknown",
        "provider": client_config["provider_name"],
        "model_used": selected_model,
    }

    if submit_button:
        text_input = ""
        source_type = ""
        if uploaded_file is not None:
            content_type = uploaded_file.type
            file_bytes = uploaded_file.read()
            if content_type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                text_input = extract_text_from_pdf(file_bytes)
                source_type = "PDF upload"
            elif "word" in content_type or uploaded_file.name.lower().endswith(".docx"):
                text_input = extract_text_from_docx(file_bytes)
                source_type = "DOCX upload"
            else:
                st.error("Unsupported file type. Please upload a PDF or DOCX file.")
                return

        if not text_input and url_input:
            text_input = extract_text_from_url(url_input)
            source_type = "URL article"
            if not text_input:
                st.warning(
                    "Unable to extract text from the provided URL. Try a different URL or paste text directly."
                )

        if not text_input and raw_text:
            text_input = raw_text
            source_type = "Raw text"

        if not text_input:
            st.error("Please upload a PDF/DOCX, paste a URL, or enter text before analyzing.")
            return

        try:
            cleaned_text = clean_text(text_input)
            if not cleaned_text:
                st.error("The uploaded content could not be parsed or is empty after cleaning.")
                return

            with st.spinner("Creating text chunks and computing embeddings..."):
                chunks = chunk_text(cleaned_text)
                stats["chunk_count"] = len(chunks)
                stats["char_count"] = len(cleaned_text)
                stats["source"] = source_type
                stats["word_count"] = get_word_count(cleaned_text)
                stats["reading_time"] = get_reading_time(cleaned_text)
                stats["lexical_sentiment"] = analyze_sentiment(cleaned_text)

            pinecone_client = PineconeClient(
                index_name=PINECONE_INDEX,
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENV,
            )

            retrieved_chunks = []
            if pinecone_client.enabled:
                try:
                    chunk_payload = []
                    for idx, chunk in enumerate(chunks):
                        vector = generate_embedding(chunk)
                        chunk_payload.append(
                            {
                                "id": f"chunk-{int(time.time())}-{idx}",
                                "values": vector,
                                "metadata": {"text": chunk, "source": source_type, "chunk_index": idx},
                            }
                        )
                    pinecone_client.upsert_chunks(chunk_payload)
                    query_vector = generate_embedding(cleaned_text)
                    retrieved = pinecone_client.query_similar_chunks(query_vector, top_k=top_k)
                    retrieved_chunks = [item["metadata"]["text"] for item in retrieved]
                    stats["retrieved_count"] = len(retrieved_chunks)
                except Exception as error:
                    st.warning(
                        f"Pinecone storage or retrieval failed: {error}. Continuing without semantic retrieval."
                    )

            if not retrieved_chunks:
                retrieved_chunks = chunks[:top_k]
                stats["retrieved_count"] = len(retrieved_chunks)

            summarization_source = "\n\n".join(retrieved_chunks)
            prompt_content = (
                "Use the following relevant document chunks to generate the requested outputs:\n\n"
                f"{summarization_source}\n\n"
                "If the content is lengthy, prioritize the most meaningful sections."
            )

            with st.spinner(f"Generating summary with {selected_model}..."):
                summary_data = generate_ai_summary(
                    prompt_content,
                    api_key=api_key,
                    provider=provider,
                    model=selected_model,
                )

            if not summary_data or summary_data.get("error"):
                raw_message = summary_data.get("message", "") if summary_data else ""
                error_message = format_api_error(raw_message, selected_model)
                if is_no_credits_error(raw_message):
                    render_no_credits_help()
                else:
                    st.error(error_message)
                return

            stats["model_used"] = summary_data.get("model_used", selected_model)
            stats["provider"] = summary_data.get("provider", client_config["provider_name"])
            st.success("Content analyzed successfully!")
            render_output(summary_data, stats)

        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")

    else:
        st.info("Upload content, paste a URL, or enter text, then click **Analyze Content** to start.")


if __name__ == "__main__":
    main()
