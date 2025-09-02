# sentiment.py

from textblob import TextBlob
import news_fetcher
import time

# Cache to reduce API calls
_cached_news = None
_cached_time = 0
CACHE_DURATION = 300  # seconds


def get_sentiment_score(symbol: str) -> float:
    """Calculate average sentiment score for news related to a given symbol."""
    global _cached_news, _cached_time

    now = time.time()
    if _cached_news is None or now - _cached_time > CACHE_DURATION:
        _cached_news = news_fetcher.fetch_news(query="crypto")  # Default: crypto
        _cached_time = now

    if not _cached_news:
        return 0.0  # Neutral if no news

    relevant_news = [
        n for n in _cached_news
        if symbol.split('/')[0].lower() in (n.get('title', '') + n.get('description', '')).lower()
    ]

    if not relevant_news:
        return 0.0

    scores = []
    for article in relevant_news:
        text = article.get("description") or article.get("title") or ""
        if not text.strip():
            continue
        blob = TextBlob(text)
        scores.append(blob.sentiment.polarity)  # Range: -1 to 1

    return round(sum(scores) / len(scores), 3) if scores else 0.0