# news_fetcher.py

import requests
import config
import logger

BASE_URL = "https://newsdata.io/api/1/news"


def fetch_news(query="crypto", language="en", max_results=10):
    """Fetch latest crypto or stock news articles from NewsData.io."""
    try:
        params = {
            "apikey": config.NEWSDATA_API_KEY,
            "q": query,
            "language": language,
        }
        response = requests.get(BASE_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])[:max_results]
        else:
            logger.log_error(f"News API error: {response.status_code} {response.text}")
            return []
    except Exception as e:
        logger.log_error(f"Error fetching news: {e}")
        return []