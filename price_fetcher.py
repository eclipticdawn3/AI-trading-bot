import time
import math
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import ccxt
import requests
from datetime import datetime

import config
import logger
from markets import get_markets, by_source
from strategy import generate_signal  # generate_signal(symbol, prices, sentiment)
from sentiment import get_sentiment_score
import trader
import virtual_trader
import notifier

LOG = logger.get_logger()

# ---------- small utils ----------
def _to_ms(dt_str: str) -> int:
    # FMP returns "YYYY-MM-DD HH:MM:SS" (UTC)
    try:
        return int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)
    except Exception:
        # Try date-only
        try:
            return int(datetime.strptime(dt_str, "%Y-%m-%d").timestamp() * 1000)
        except Exception:
            return 0

def _sleep_limit():
    time.sleep(max(0.0, float(getattr(config, "FMP_RATE_LIMIT_SEC", 0.25))))

class PriceFetcher:
    """
    Hybrid fetcher:
      - Crypto from KuCoin (CCXT)
      - Stocks/ETFs/Gold/Forex from FMP (Financial Modeling Prep)
    Produces:
      - self.last_ohlcv[symbol][timeframe] -> List[[ts, o, h, l, c, v]]
      - self.last_prices[symbol] -> float (last close/last trade)
    """
    def __init__(self, interval: int = 60):
        self.interval = interval
        self.markets = get_markets()
        self.by_src = by_source(self.markets)

        self.exchange = None
        if getattr(config, "ENABLE_CCXT", True):
            self.exchange = ccxt.kucoin({
                'enableRateLimit': True,
                'apiKey': getattr(config, 'API_KEY', ''),
                'secret': getattr(config, 'API_SECRET', ''),
                'password': getattr(config, 'API_PASSPHRASE', ''),
                'options': {'defaultType': 'spot'}
            })

        self.fmp_key = getattr(config, "FMP_API_KEY", None)
        if getattr(config, "ENABLE_FMP", False) and not self.fmp_key:
            LOG.warning("ENABLE_FMP=True but FMP_API_KEY is missing")

        # caches / stores
        self.last_ohlcv: Dict[str, Dict[str, List[List[float]]]] = {}
        self.last_prices: Dict[str, float] = {}

    # ---------- CCXT helper ----------
    def _fetch_ccxt_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None):
        if not self.exchange:
            return []
        if limit is None:
            limit = getattr(config, 'DEFAULT_LIMIT', 200)
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception as e:
            LOG.error(f"CCXT OHLCV error {symbol} {timeframe}: {e}")
            return []

    def _fetch_ccxt_ticker_last(self, symbol: str) -> Optional[float]:
        if not self.exchange:
            return None
        try:
            t = self.exchange.fetch_ticker(symbol)
            if t and t.get("last") is not None:
                return float(t["last"])
        except Exception as e:
            LOG.error(f"CCXT ticker error {symbol}: {e}")
        return None

    # ---------- FMP helpers ----------
    def _fmp_hist_chart(self, symbol: str, interval: str = "1hour", limit: int = 200) -> List[List[float]]:
        """
        FMP "historical-chart" endpoint.
        Works for stocks/ETFs (e.g., AAPL, SPY, QQQ) and forex metals (XAUUSD).
        Intervals: 1min, 5min, 15min, 30min, 1hour, 4hour
        """
        key = self.fmp_key
        if not key:
            return []
        url = f"https://financialmodelingprep.com/api/v3/historical-chart/{interval}/{symbol}?apikey={key}"
        try:
            _sleep_limit()
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or not data:
                return []

            # FMP returns newest-first; we want oldest-first
            data = list(reversed(data))
            rows = []
            for row in data[-limit:]:
                ts = _to_ms(str(row.get("date", "")))
                o = float(row.get("open", 0) or 0)
                h = float(row.get("high", 0) or 0)
                l = float(row.get("low", 0) or 0)
                c = float(row.get("close", 0) or 0)
                v = float(row.get("volume", 0) or 0)
                # Ensure valid
                if ts <= 0 or min(o, h, l, c) <= 0:
                    continue
                rows.append([ts, o, h, l, c, v])
            return rows
        except Exception as e:
            LOG.error(f"FMP historical error {symbol} ({interval}): {e}")
            return []

    def _fmp_last_price(self, symbol: str) -> Optional[float]:
        key = self.fmp_key
        if not key:
            return None
        # Try quote endpoint first
        url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={key}"
        try:
            _sleep_limit()
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            js = r.json()
            if isinstance(js, list) and js:
                price = js[0].get("price")
                if price is None:
                    price = js[0].get("previousClose")
                return float(price) if price is not None else None
        except Exception:
            pass
        # Fallback to last from historical chart (1h)
        rows = self._fmp_hist_chart(symbol, interval="1hour", limit=5)
        return float(rows[-1][4]) if rows else None

    # ---------- ATR ----------
    @staticmethod
    def compute_atr_from_ohlcv(ohlcv: list, period: int = 14) -> Optional[float]:
        try:
            if not ohlcv or len(ohlcv) < max(3, period):
                return None
            highs = np.array([row[2] for row in ohlcv], dtype=float)
            lows  = np.array([row[3] for row in ohlcv], dtype=float)
            closes= np.array([row[4] for row in ohlcv], dtype=float)

            # True Range: max(high-low, abs(high-prevClose), abs(low-prevClose))
            prev_close = np.roll(closes, 1)
            prev_close[0] = closes[0]
            tr1 = highs - lows
            tr2 = np.abs(highs - prev_close)
            tr3 = np.abs(lows - prev_close)
            tr = np.maximum(tr1, np.maximum(tr2, tr3))

            if len(tr) < period:
                return float(np.mean(tr))
            series = pd.Series(tr)
            val = series.rolling(window=period).mean().iloc[-1]
            return float(val) if not np.isnan(val) else float(np.mean(tr))
        except Exception as e:
            LOG.error(f"ATR compute error: {e}")
            return None

    # ---------- aggregate fetch ----------
    def fetch_all(self):
        """
        Populate self.last_ohlcv and self.last_prices for all markets defined in markets.py
        """
        # 1) KuCoin (crypto)
        if getattr(config, "ENABLE_CCXT", True):
            for m in self.by_src.get('kucoin', []):
                symbol = m.symbol
                try:
                    ohlcv_5m = self._fetch_ccxt_ohlcv(symbol, '5m', limit=getattr(config, 'DEFAULT_LIMIT', 200))
                    ohlcv_1h = self._fetch_ccxt_ohlcv(symbol, '1h', limit=getattr(config, 'DEFAULT_LIMIT', 200))
                    ohlcv_4h = self._fetch_ccxt_ohlcv(symbol, '4h', limit=getattr(config, 'DEFAULT_LIMIT', 200))
                    tf_map = {}
                    if ohlcv_5m: tf_map['5m'] = ohlcv_5m
                    if ohlcv_1h: tf_map['1h'] = ohlcv_1h
                    if ohlcv_4h: tf_map['4h'] = ohlcv_4h
                    if tf_map:
                        self.last_ohlcv[symbol] = tf_map
                    lp = self._fetch_ccxt_ticker_last(symbol)
                    if lp is not None:
                        self.last_prices[symbol] = lp
                except Exception as e:
                    LOG.error(f"Fetch error (CCXT) {symbol}: {e}")

        # 2) FMP (stocks/ETFs/forex metals)
        if getattr(config, "ENABLE_FMP", False):
            for m in self.by_src.get('fmp', []):
                symbol = m.symbol  # display name in your system
                key = m.key        # actual FMP symbol (e.g., AAPL, SPY, QQQ, NVDA, XAUUSD)
                try:
                    rows_1h = self._fmp_hist_chart(key, interval="1hour", limit=getattr(config, 'DEFAULT_LIMIT', 200))
                    rows_4h = self._fmp_hist_chart(key, interval="4hour", limit=min(200, getattr(config, 'DEFAULT_LIMIT', 200)))
                    tf_map = {}
                    if rows_1h: tf_map['1h'] = rows_1h
                    if rows_4h: tf_map['4h'] = rows_4h
                    if tf_map:
                        self.last_ohlcv[symbol] = tf_map
                    lp = self._fmp_last_price(key)
                    if lp is not None:
                        self.last_prices[symbol] = lp
                except Exception as e:
                    LOG.error(f"Fetch error (FMP) {symbol}: {e}")

    # ---------- one live/demo cycle ----------
    def cycle_once(self):
        """
        One cycle:
         - fetch market data
         - compute ATRs
         - compute sentiment
         - generate signals and act (virtual or live)
        """
        start = time.time()
        LOG.info("Cycle: fetching market data...")
        self.fetch_all()

        # compute ATR map for symbols (prefer finest TF we have)
        atr_map: Dict[str, float] = {}
        for sym, data in list(self.last_ohlcv.items()):
            o5 = data.get('5m') or []
            o1 = data.get('1h') or []
            base = o5 if len(o5) >= 20 else o1
            atr = self.compute_atr_from_ohlcv(base, period=getattr(config, 'ATR_PERIOD', 14)) if base else None
            if atr:
                atr_map[sym] = atr

        # sentiment cache
        LOG.info("Cycle: fetching news/sentiment...")
        sentiment_cache: Dict[str, float] = {}
        for sym in list(self.last_ohlcv.keys()):
            try:
                base = sym.split('/')[0] if '/' in sym else sym
                s = get_sentiment_score(base)
                sentiment_cache[sym] = s
            except Exception as e:
                LOG.error(f"Sentiment fetch failed for {sym}: {e}")
                sentiment_cache[sym] = 0.0

        # strategy loop
        LOG.info("Cycle: running strategy and handling signals...")
        for sym, tf_map in list(self.last_ohlcv.items()):
            try:
                base_ohlcv = tf_map.get('5m') or tf_map.get('1h') or tf_map.get('4h') or []
                if not base_ohlcv:
                    continue
                last_candle = base_ohlcv[-1]
                prices = {
                    "timestamp": last_candle[0],
                    "open":  float(last_candle[1]),
                    "high":  float(last_candle[2]),
                    "low":   float(last_candle[3]),
                    "close": float(last_candle[4]),
                    "volume": float(last_candle[5]),
                    "avg_volume": float(np.mean([c[5] for c in base_ohlcv[-20:]])) if len(base_ohlcv) >= 5 else float(last_candle[5])
                }

                sentiment = float(sentiment_cache.get(sym, 0.0))
                decision = generate_signal(sym, prices, sentiment)
                if not decision:
                    continue

                conf = float(decision.get('confidence', 0))
                norm_conf = max(0.0, min(1.0, conf / 100.0))

                if norm_conf < getattr(config, 'CONFIDENCE_THRESHOLD', 0.65):
                    LOG.info(f"Skip {sym}: low confidence {norm_conf:.2f}")
                    continue

                last_price = float(self.last_prices.get(sym) or prices["close"])

                # execution: virtual or live
                if getattr(config, 'VIRTUAL_TRADING', True):
                    atr = atr_map.get(sym)
                    virtual_trader.handle_signal(sym, decision, last_price, atr)
                else:
                    side = decision.get('filtered', 'HOLD')
                    if side not in ('BUY', 'SELL'):
                        continue
                    qty = trader.calculate_dynamic_amount(sym)
                    if qty <= 0:
                        LOG.info(f"Live skip {sym}: insufficient balance or zero qty")
                        continue
                    trader.place_order(sym, side.lower(), qty)

                # notify via telegram
                if getattr(config, 'TELEGRAM_ENABLED', True):
                    try:
                        msg = (f"🔔 Trade Signal: {decision.get('filtered')} | {sym}\n"
                               f"Conf: {norm_conf*100:.0f}% | Price: {last_price:.6f}\n"
                               f"Sentiment: {sentiment:.3f}")
                        notifier.send_telegram_message(msg)
                    except Exception as e:
                        LOG.error(f"Telegram notify failed: {e}")

            except Exception as e:
                LOG.error(f"Error processing {sym}: {e}", exc_info=config.PRINT_STACK_TRACES)

        elapsed = time.time() - start
        LOG.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {self.interval}s.")
        time.sleep(max(0.5, self.interval))

    # ---------- walk-forward backtest (demo/virtual only) ----------
    def run_backtest(self, days: int = 30):
        """
        Walk-forward backtest using the virtual_trader engine (KuCoin + FMP).
        Uses 1h candles for speed. Records trades to virtual_trader logs.
        """
        LOG.info("Starting backtest (walk-forward) for %d days...", days)

        # Build per-source lists
        ku_markets = [m for m in self.markets if m.source == "kucoin"]
        fmp_markets = [m for m in self.markets if m.source == "fmp"]

        # CCXT window
        if self.exchange:
            ms_per_day = 24 * 3600 * 1000
            now_ms = self.exchange.milliseconds()
            since_ms = now_ms - int(days * ms_per_day)

        # 1) Backtest KuCoin symbols
        for m in ku_markets:
            sym = m.symbol
            try:
                if not self.exchange:
                    LOG.info("No CCXT exchange, skipping %s", sym)
                    continue

                LOG.info("Backtesting %s ...", sym)
                hist = self.exchange.fetch_ohlcv(sym, timeframe='1h', since=since_ms,
                                                 limit=min(days*24 + 200, 5000))
                if not hist:
                    LOG.info("No history for %s", sym)
                    continue

                for i in range(24, len(hist)):
                    window_1h = hist[max(0, i - 48): i + 1]   # last 48 bars
                    last = window_1h[-1]
                    prices = {
                        "timestamp": last[0],
                        "open": float(last[1]),
                        "high": float(last[2]),
                        "low":  float(last[3]),
                        "close":float(last[4]),
                        "volume": float(last[5]),
                        "avg_volume": float(np.mean([c[5] for c in window_1h[-20:]])) if len(window_1h) >= 5 else float(last[5])
                    }
                    atr_val = self.compute_atr_from_ohlcv(window_1h, period=getattr(config, 'ATR_PERIOD', 14))
                    sentiment = 0.0  # keep neutral in historical

                    decision = generate_signal(sym, prices, sentiment)
                    if not decision:
                        continue
                    conf = float(decision.get('confidence', 0))
                    if conf/100.0 < getattr(config, 'CONFIDENCE_THRESHOLD', 0.65):
                        continue

                    last_price = float(prices["close"])
                    virtual_trader.handle_signal(sym, decision, last_price, atr_val)
                    virtual_trader.update_positions({sym: last_price}, atr_map={sym: atr_val})
                LOG.info("Finished backtesting %s", sym)

            except Exception as e:
                LOG.error(f"Backtest error for {sym}: {e}")

        # 2) Backtest FMP symbols (1h)
        for m in fmp_markets:
            sym = m.symbol
            key = m.key
            try:
                LOG.info("Backtesting (FMP) %s ...", sym)
                hist = self._fmp_hist_chart(key, interval="1hour", limit=days*24 + 200)
                if not hist:
                    LOG.info("No FMP history for %s", sym)
                    continue

                for i in range(24, len(hist)):
                    window_1h = hist[max(0, i - 48): i + 1]
                    last = window_1h[-1]
                    prices = {
                        "timestamp": last[0],
                        "open": float(last[1]),
                        "high": float(last[2]),
                        "low":  float(last[3]),
                        "close":float(last[4]),
                        "volume": float(last[5]),
                        "avg_volume": float(np.mean([c[5] for c in window_1h[-20:]])) if len(window_1h) >= 5 else float(last[5])
                    }
                    atr_val = self.compute_atr_from_ohlcv(window_1h, period=getattr(config, 'ATR_PERIOD', 14))
                    sentiment = 0.0

                    decision = generate_signal(sym, prices, sentiment)
                    if not decision:
                        continue
                    conf = float(decision.get('confidence', 0))
                    if conf/100.0 < getattr(config, 'CONFIDENCE_THRESHOLD', 0.65):
                        continue

                    last_price = float(prices["close"])
                    virtual_trader.handle_signal(sym, decision, last_price, atr_val)
                    virtual_trader.update_positions({sym: last_price}, atr_map={sym: atr_val})
                LOG.info("Finished backtesting (FMP) %s", sym)

            except Exception as e:
                LOG.error(f"Backtest (FMP) error for {sym}: {e}")

        LOG.info("Backtest finished for %d days. Virtual balance: %.4f", days, virtual_trader.balance)