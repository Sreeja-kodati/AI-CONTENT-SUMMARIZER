import re
import unicodedata


def get_word_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def get_reading_time(text: str, words_per_minute: int = 200) -> str:
    words = get_word_count(text)
    minutes = max(1, round(words / words_per_minute))
    return f"{minutes} min"


def safe_filename(filename: str) -> str:
    filename = unicodedata.normalize("NFKD", filename)
    filename = re.sub(r"[^a-zA-Z0-9._-]", "_", filename)
    return filename
