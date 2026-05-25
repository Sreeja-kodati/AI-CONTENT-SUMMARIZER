from textblob import TextBlob


def analyze_sentiment(text: str) -> str:
    if not text:
        return "Neutral"
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "Positive"
        if polarity < -0.1:
            return "Negative"
        return "Neutral"
    except Exception:
        return "Neutral"
