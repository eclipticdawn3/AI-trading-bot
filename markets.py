# ================
# markets.py
# Market registry + helpers (KuCoin via ccxt, Stocks/ETFs/Gold via Yahoo)
# ================

from dataclasses import dataclass
from typing import List, Dict
import config

@dataclass(frozen=True)
class Market:
    symbol: str         # display symbol used everywhere
    source: str         # 'kucoin' or 'yahoo'
    key: str            # raw symbol for the source (e.g., same for KuCoin, Yahoo ticker for yfinance)
    quote: str          # 'USDT' or 'USD' etc.

def get_markets() -> List[Market]:
    mkts: List[Market] = []
    # Crypto on KuCoin
    for sym in config.CRYPTO_SYMBOLS:
        # assume quote is USDT in our list
        mkts.append(Market(symbol=sym, source="kucoin", key=sym, quote="USDT"))

    # Yahoo-backed (if enabled)
    if getattr(config, "ENABLE_YAHOO", False):
        for ysym in config.YAHOO_SYMBOLS:
            # display symbol as given; quote is typically USD
            mkts.append(Market(symbol=ysym, source="yahoo", key=ysym, quote="USD"))
    return mkts

def by_source(markets: List[Market]) -> Dict[str, List[Market]]:
    buckets: Dict[str, List[Market]] = {"kucoin": [], "yahoo": []}
    for m in markets:
        if m.source in buckets:
            buckets[m.source].append(m)
    return buckets

def display_symbols() -> List[str]:
    return [m.symbol for m in get_markets()]