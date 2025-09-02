# nlp_boost.py

import time
import news_fetcher
from sentiment import get_sentiment_score

# Cache fetched news
news_cache = []
last_fetch_time = 0
CACHE_DURATION = 300  # 5 minutes


def get_recent_news():
    """Return cached news, refetching if cache expired."""
    global news_cache, last_fetch_time
    now = time.time()
    if now - last_fetch_time > CACHE_DURATION:
        news_cache = news_fetcher.fetch_news()
        last_fetch_time = now
    return news_cache


def apply_news_boost(symbol: str, confidence: float) -> float:
    """Adjust confidence score based on news sentiment."""
    sentiment_score = get_sentiment_score(symbol)

    # Sentiment polarity scaling
    if sentiment_score > 0.1:
        boost = 0.05
    elif sentiment_score < -0.1:
        boost = -0.05
    else:
        boost = 0.0

    adjusted = confidence + boost
    adjusted = min(max(adjusted, 0), 1)  # Clamp 0-1
    return round(adjusted, 2)