import io
import re
from typing import Optional
from PyPDF2 import PdfReader
from docx import Document

try:
    from newspaper import Article
    NEWSPAPER_ENABLED = True
except Exception:
    Article = None
    NEWSPAPER_ENABLED = False

try:
    import requests
    from bs4 import BeautifulSoup
    FALLBACK_ENABLED = True
except Exception:
    requests = None
    BeautifulSoup = None
    FALLBACK_ENABLED = False


def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
        return "\n\n".join(paragraphs).strip()
    except Exception:
        return ""


def _clean_html(html_content: str) -> str:
    if not html_content:
        return ""
    if BeautifulSoup:
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception:
            pass
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.I | re.S)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_text_from_url(url: str) -> str:
    if not url or not url.strip():
        return ""

    if NEWSPAPER_ENABLED and Article is not None:
        try:
            article = Article(url)
            article.download()
            article.parse()
            result = article.text.strip()
            if result:
                return result
        except Exception:
            pass

    if FALLBACK_ENABLED and requests is not None:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return _clean_html(response.text)
        except Exception:
            return ""

    return ""


def clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\u00a0", " ")
    text = text.strip()
    return text
