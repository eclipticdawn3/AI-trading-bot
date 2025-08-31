# ================
# price_fetcher.py — robust multi-market fetcher + execution loop + backtest
# ================

import time
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import ccxt
import yfinance as yf

import config
import logger
from markets import get_markets, by_source, Market
from strategy import generate_signal           # generate_signal(symbol, prices, sentiment)
from sentiment import get_sentiment_score
import trader
import virtual_trader
import notifier

LOG = logger.get_logger()

# ---------------------------
# Config defaults (safe if missing in config.py)
# ---------------------------
DEF_FETCH_LIMIT = int(getattr(config, "DEFAULT_LIMIT", 300))
CONF_THRESH = float(getattr(config, "CONFIDENCE_THRESHOLD", 0.65))
VIRTUAL = bool(getattr(config, "VIRTUAL_TRADING", True))
ATR_PERIOD = int(getattr(config, "ATR_PERIOD", 14))
SLEEP_SEC = int(getattr(config, "MAIN_LOOP_INTERVAL", 60))

MAX_CONCURRENT_POS = int(getattr(config, "MAX_CONCURRENT_POS", 5))     # risk: cap simultaneous positions
MIN_ATR = float(getattr(config, "MIN_ATR_FILTER", 0.0))                 # if >0, skip very low ATR symbols
MIN_AVG_VOL = float(getattr(config, "MIN_AVG_VOLUME", 0.0))             # if >0, skip dead candles
COOLDOWN_SEC = int(getattr(config, "SIGNAL_COOLDOWN_SEC", 300))         # avoid churn per symbol

TELE_ENABLED = bool(getattr(config, "TELEGRAM_ENABLED", True))

# YF: for FX (=X) and some futures, intraday often unavailable; we gracefully fallback
YF_STOCK_INTERVAL = getattr(config, "YF_STOCK_INTERVAL", "1h")
YF_STOCK_PERIOD   = getattr(config, "YF_STOCK_PERIOD",   "14d")
YF_FALLBACK_INTERVAL = getattr(config, "YF_FALLBACK_INTERVAL", "1d")
YF_FALLBACK_PERIOD   = getattr(config, "YF_FALLBACK_PERIOD",   "60d")

# Optional: custom tickers that behave like FX (=X) or Futures (=F)
YF_FORCE_DAILY: Tuple[str, ...] = tuple(getattr(
    config, "YF_FORCE_DAILY", ("XAUUSD=X", "EURUSD=X", "JPY=X", "GC=F", "SI=F", "CL=F")
))


def _is_crypto_symbol(sym: str) -> bool:
    """Heuristic: KuCoin spot pairs look like 'BTC/USDT' etc."""
    return "/" in sym


def _ts_ms_from_index(idx: pd.DatetimeIndex) -> np.ndarray:
    """DatetimeIndex -> milliseconds since epoch (int64)."""
    # convert nanoseconds to milliseconds safely
    return (idx.view("int64") // 1_000_000).astype("int64")


def _flatten_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo can return MultiIndex columns in many edge-cases.
    We flatten them and try to keep standard OHLCV names present.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in c if str(x) != ""]) for c in df.columns.values]
    # normalize exact col picks (first matching token)
    colmap = {}
    for want in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        candidates = [c for c in df.columns if c.split()[0].lower() == want.lower()]
        if candidates:
            colmap[want] = candidates[0]
    # If Adj Close exists but Close missing, use Adj Close as Close
    if "Close" not in colmap and "Adj Close" in colmap:
        colmap["Close"] = colmap["Adj Close"]
    missing = [k for k in ["Open", "High", "Low", "Close"] if k not in colmap]
    if missing:
        raise ValueError(f"Missing OHLC columns from Yahoo data: {missing}")
    # Build slim DF with standard names
    out = pd.DataFrame(index=df.index)
    for k in ["Open", "High", "Low", "Close"]:
        out[k] = df[colmap[k]]
    out["Volume"] = df[colmap["Volume"]] if "Volume" in colmap else 0.0
    return out.dropna(how="any")


def _yf_download_single(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Wrapper around yfinance.download with sane defaults and clearer errors.
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=False,   # keep raw OHLC
        actions=False
    )
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} with {period}/{interval}")
    return _flatten_yf_df(df)


def _df_to_ohlcv_rows(df: pd.DataFrame) -> List[List[float]]:
    """
    Convert standardized DF(Open,High,Low,Close,Volume) to CCXT-like rows.
    """
    idx_ms = _ts_ms_from_index(df.index)
    o = df["Open"].to_numpy(dtype="float64")
    h = df["High"].to_numpy(dtype="float64")
    l = df["Low"].to_numpy(dtype="float64")
    c = df["Close"].to_numpy(dtype="float64")
    v = df.get("Volume", pd.Series(np.zeros(len(df), dtype="float64"))).to_numpy(dtype="float64")
    rows = np.column_stack([idx_ms, o, h, l, c, v]).tolist()
    return rows


class PriceFetcher:
    def __init__(self, interval: int = SLEEP_SEC):
        self.interval = interval
        self.markets: List[Market] = get_markets()
        self.by_src = by_source(self.markets)

        self.exchange = ccxt.kucoin({
            "enableRateLimit": True,
            "apiKey": getattr(config, "API_KEY", ""),
            "secret": getattr(config, "API_SECRET", ""),
            "password": getattr(config, "API_PASSPHRASE", "")
        })

        # data stores
        self.last_ohlcv: Dict[str, Dict[str, Any]] = {}
        self.last_prices: Dict[str, float] = {}
        self.atr_cache: Dict[str, float] = {}

        # throttle churn
        self._last_action_ts: Dict[str, float] = {}

    # -------- KuCoin ----------
    def _fetch_ccxt_ohlcv(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> list:
        limit = limit or DEF_FETCH_LIMIT
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []
        except Exception as e:
            LOG.error(f"KuCoin OHLCV error {symbol} {timeframe}: {e}")
            return []

    def _fetch_ccxt_ticker_last(self, symbol: str) -> Optional[float]:
        try:
            t = self.exchange.fetch_ticker(symbol)
            last = t.get("last", None)
            return float(last) if last is not None else None
        except Exception:
            return None

    # -------- Yahoo ----------
    def _fetch_yahoo_ohlcv(
        self,
        ticker: str,
        period: str = YF_STOCK_PERIOD,
        interval: str = YF_STOCK_INTERVAL
    ) -> list:
        """
        Try intraday first; fallback to daily if Yahoo doesn’t provide intraday for this symbol.
        """
        try:
            # Force daily path for tickers known to lack intraday
            if ticker in YF_FORCE_DAILY or ticker.endswith("=X"):
                df = _yf_download_single(ticker, YF_FALLBACK_PERIOD, YF_FALLBACK_INTERVAL)
                return _df_to_ohlcv_rows(df)

            # primary attempt (intraday)
            df = _yf_download_single(ticker, period, interval)
            return _df_to_ohlcv_rows(df)

        except Exception as e1:
            LOG.info(f"Yahoo {ticker} intraday failed: {e1} — falling back to daily")
            try:
                df = _yf_download_single(ticker, YF_FALLBACK_PERIOD, YF_FALLBACK_INTERVAL)  # daily
                return _df_to_ohlcv_rows(df)
            except Exception as e2:
                LOG.error(f"Yahoo fetch error for {ticker}: {e2}")
                return []

    # -------- Aggregate fetch ----------
    def fetch_all(self):
        """
        Populate last_ohlcv and last_prices across all markets.
        """
        # KuCoin (crypto)
        for m in self.by_src.get("kucoin", []):
            symbol = m.symbol
            try:
                o5  = self._fetch_ccxt_ohlcv(symbol, "5m", limit=200)
                o1  = self._fetch_ccxt_ohlcv(symbol, "1h", limit=200)
                o4  = self._fetch_ccxt_ohlcv(symbol, "4h", limit=200)
                self.last_ohlcv[symbol] = {"5m": o5, "1h": o1, "4h": o4}

                last = self._fetch_ccxt_ticker_last(symbol)
                if last is not None:
                    self.last_prices[symbol] = last
            except Exception as e:
                LOG.error(f"Fetch error (KuCoin) for {symbol}: {e}")

        # Yahoo (stocks/ETFs/FX/etc.)
        if getattr(config, "ENABLE_YAHOO", False):
            for m in self.by_src.get("yahoo", []):
                symbol = m.symbol
                try:
                    rows = self._fetch_yahoo_ohlcv(m.key, period=YF_STOCK_PERIOD, interval=YF_STOCK_INTERVAL)
                    if rows:
                        # we only store 1h & 4h views (approx for daily via same list)
                        self.last_ohlcv[symbol] = {
                            "1h": rows,
                            "4h": rows[-100:] if len(rows) > 100 else rows
                        }
                        self.last_prices[symbol] = float(rows[-1][4])
                    else:
                        LOG.info(f"No Yahoo data for {symbol}")
                except Exception as e:
                    LOG.error(f"Yahoo fetch error for {symbol}: {e}")

    # -------- Indicators ----------
    @staticmethod
    def compute_atr_from_ohlcv(ohlcv: list, period: int = ATR_PERIOD) -> Optional[float]:
        """
        ATR approximation using simple high-low range SMA.
        """
        try:
            if not ohlcv or len(ohlcv) < max(3, period):
                return None
            highs = np.array([row[2] for row in ohlcv], dtype="float64")
            lows  = np.array([row[3] for row in ohlcv], dtype="float64")
            ranges = highs - lows
            if len(ranges) < period:
                return float(np.mean(ranges)) if len(ranges) > 0 else None
            series = pd.Series(ranges)
            val = series.rolling(window=period).mean().iloc[-1]
            if np.isnan(val):
                return float(np.mean(ranges))
            return float(val)
        except Exception as e:
            LOG.error(f"ATR compute error: {e}")
            return None

    # -------- Risk/Churn helpers ----------
    def _on_cooldown(self, symbol: str, now: float) -> bool:
        last = self._last_action_ts.get(symbol, 0.0)
        return (now - last) < COOLDOWN_SEC

    def _mark_action(self, symbol: str, now: float):
        self._last_action_ts[symbol] = now

    def _open_positions_count(self) -> int:
        try:
            return len(virtual_trader.virtual_trader.positions)
        except Exception:
            # fallback for older alias
            return len(getattr(virtual_trader, "positions", {}))

    # -------- Main cycle ----------
    def cycle_once(self):
        start = time.time()
        LOG.info("Cycle: fetching market data...")
        self.fetch_all()

        # ATR per symbol
        atr_map: Dict[str, float] = {}
        for sym, data in self.last_ohlcv.items():
            o5 = data.get("5m") or []
            o1 = data.get("1h") or []
            ohl = o5 if len(o5) >= 20 else o1
            atr = self.compute_atr_from_ohlcv(ohl, period=ATR_PERIOD)
            if atr is not None:
                atr_map[sym] = atr

        # Sentiment cache
        LOG.info("Cycle: fetching news/sentiment...")
        sentiment_cache: Dict[str, float] = {}
        for sym in list(self.last_ohlcv.keys()):
            try:
                base = sym.split("/")[0] if "/" in sym else sym
                s = get_sentiment_score(base)
                sentiment_cache[sym] = s
            except Exception as e:
                LOG.error(f"Sentiment fetch failed for {sym}: {e}")
                sentiment_cache[sym] = 0.0

        # Strategy + execution
        LOG.info("Cycle: running strategy and handling signals...")
        now = time.time()
        markets_to_process = list(self.last_ohlcv.keys())

        for sym in markets_to_process:
            try:
                if self._on_cooldown(sym, now):
                    continue

                ohl = self.last_ohlcv.get(sym) or {}
                base_ohlcv = ohl.get("5m") or ohl.get("1h") or ohl.get("4h") or []
                if not base_ohlcv:
                    continue

                last_candle = base_ohlcv[-1]
                prices = {
                    "timestamp": int(last_candle[0]),
                    "open": float(last_candle[1]),
                    "high": float(last_candle[2]),
                    "low": float(last_candle[3]),
                    "close": float(last_candle[4]),
                    "volume": float(last_candle[5]),
                    "avg_volume": float(np.mean([c[5] for c in base_ohlcv[-20:]])) if len(base_ohlcv) >= 20 else float(last_candle[5])
                }

                # basic quality filters
                if MIN_AVG_VOL > 0 and prices["avg_volume"] < MIN_AVG_VOL:
                    continue
                atr_val = atr_map.get(sym)
                if MIN_ATR > 0 and (atr_val is None or atr_val < MIN_ATR):
                    continue

                sentiment = sentiment_cache.get(sym, 0.0)
                decision = generate_signal(sym, prices, sentiment)
                if not decision:
                    continue

                conf = float(decision.get("confidence", 0))
                norm_conf = max(0.0, min(1.0, conf / 100.0))
                if norm_conf < CONF_THRESH:
                    LOG.info(f"Skip {sym}: low confidence {norm_conf:.2f}")
                    continue

                # risk: cap concurrent positions
                if VIRTUAL and decision.get("filtered") in ("BUY", "SELL"):
                    if self._open_positions_count() >= MAX_CONCURRENT_POS:
                        LOG.info(f"Skip {sym}: max concurrent positions reached")
                        continue

                last_price = float(self.last_prices.get(sym, prices["close"]))

                # execute
                if VIRTUAL:
                    try:
                        virtual_trader.handle_signal(sym, decision, last_price, atr_val)
                        virtual_trader.update_positions({sym: last_price}, atr_map={sym: atr_val or 0.0})
                        self._mark_action(sym, now)
                    except Exception as e:
                        LOG.error(f"Virtual trade failed for {sym}: {e}")
                else:
                    side = decision.get("filtered", "HOLD")
                    if side in ("BUY", "SELL"):
                        qty = trader.calculate_dynamic_amount(sym)
                        if qty > 0:
                            trader.place_order(sym, side.lower(), qty)
                            self._mark_action(sym, now)

                # notify
                if TELE_ENABLED:
                    try:
                        msg = (f"🔔 Trade Signal: {decision.get('filtered')} | {sym}\n"
                               f"Conf: {norm_conf*100:.0f}% | Price: {last_price:.6f}\n"
                               f"Sentiment: {sentiment:.3f}")
                        notifier.send_telegram_message(msg)
                    except Exception:
                        pass

            except Exception as e:
                LOG.error(f"Error processing {sym}: {e}", exc_info=getattr(config, "PRINT_STACK_TRACES", False))

        elapsed = time.time() - start
        LOG.info(f"Cycle finished in {elapsed:.2f}s. Sleeping {self.interval}s.")
        time.sleep(max(0.2, self.interval))

    # -------- Backtest ----------
    def run_backtest(self, days: int = 30):
        """
        Walk-forward backtest for KuCoin symbols (1h candles),
        recording trades via virtual_trader (CSV + JSON).
        """
        LOG.info("Starting backtest (walk-forward) for %d days...", days)
        ms_per_day = 24 * 3600 * 1000
        now_ms = self.exchange.milliseconds()
        since_ms = now_ms - int(days * ms_per_day)

        for m in self.markets:
            sym = m.symbol
            try:
                if m.source != "kucoin":
                    LOG.info("Backtest skipping non-kucoin: %s", sym)
                    continue

                LOG.info("Backtesting %s ...", sym)
                hist = self.exchange.fetch_ohlcv(sym, timeframe="1h", since=since_ms, limit=days * 24 + 50)
                if not hist:
                    LOG.info("No history for %s", sym)
                    continue

                for i in range(24, len(hist)):
                    window_1h = hist[max(0, i - 48): i + 1]  # last 48h
                    last = window_1h[-1]
                    prices = {"timestamp": int(last[0]),
                        "open": float(last[1]),
                        "high": float(last[2]),
                        "low": float(last[3]),
                        "close": float(last[4]),
                        "volume": float(last[5]),
                        "avg_volume": float(np.mean([c[5] for c in window_1h[-20:]])) if len(window_1h) >= 20 else float(last[5])
                    }

                    atr_val = self.compute_atr_from_ohlcv(window_1h, period=ATR_PERIOD) or 0.0
                    # no historical live sentiment
                    sentiment = 0.0

                    decision = generate_signal(sym, prices, sentiment)
                    if not decision:
                        continue

                    conf = float(decision.get("confidence", 0))
                    norm_conf = max(0.0, min(1.0, conf / 100.0))
                    if norm_conf < CONF_THRESH:
                        continue

                    last_price = float(prices["close"])

                    try:
                        virtual_trader.handle_signal(sym, decision, last_price, atr_val)
                        virtual_trader.update_positions({sym: last_price}, atr_map={sym: atr_val})
                    except Exception as e:
                        LOG.error("Virtual trade failed in backtest for %s: %s", sym, e)

                LOG.info("Finished backtesting %s", sym)

            except Exception as e:
                LOG.error(f"Backtest error for {sym}: {e}")

        try:
            bal = getattr(virtual_trader.virtual_trader, "balance", None)
        except Exception:
            bal = None
        LOG.info("Backtest finished for %d days. Virtual balance: %s", days, f"{bal:.4f}" if bal is not None else "N/A")