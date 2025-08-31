# strategy.py
"""
Conservative multi-layer strategy.

Entry point:
    decision = generate_signal(symbol, prices, sentiment)

- prices: dict with keys: timestamp, open, high, low, close, volume, avg_volume (preferred)
  Optionally it can include a key 'history' with the recent OHLCV list (list of rows [ts, o,h,l,c,v])
  OR call update_symbol_history(symbol, ohlcv_list) to store history centrally.

- sentiment: float in [-1.0, 1.0]
- Returns a dict:
    {
      "symbol": symbol,
      "raw": "BUY"/"SELL"/"HOLD",
      "filtered": "BUY"/"SELL"/"HOLD",
      "confidence": int(0..100),
      "strength": float (multiplier),
      "components": { ... }   # explanation
    }
"""

from typing import List, Dict, Optional
import math
import collections
import time

# Try to use ta if available for robust indicators; otherwise fallback to simple logic.
try:
    import pandas as pd
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

import config
import logger

LOG = logger.get_logger()

# === Configurable weights & thresholds (tweak these later via backtest) ===
WEIGHTS = {
    "trend": 2.0,     # EMA regime / higher TF trend
    "momentum": 1.5,  # RSI / MACD
    "vol": 0.8,       # volume confirmation
    "candle": 1.0,    # candle pattern confirmation
    "sentiment": 1.0, # news sentiment multiplier
    "noise": 1.0      # noise filters (wick/body, flatness)
}

# thresholds (these produce a conservative bot)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

EMA_FAST = 12
EMA_SLOW = 26
EMA_REGIME_SHORT = 50
EMA_REGIME_LONG = 200

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

MIN_CONF_TO_TRADE = float(getattr(config, "CONFIDENCE_THRESHOLD", 0.65))  # 0..1
# note: PriceFetcher expects confidence as 0..100 and then compares normalized value; we return 0..100

# store lightweight history per-symbol if PriceFetcher doesn't pass it
_symbol_history = {}  # symbol -> list of rows [ts, o, h, l, c, v]

# helper: normalize confidence to 0..100
def _to_pct(x: float) -> int:
    return max(0, min(100, int(round(x * 100))))

# helper: safe division
def _safe_div(a, b, default=0.0):
    try:
        return a / b if b else default
    except Exception:
        return default

# === public helpers ===

def update_symbol_history(symbol: str, ohlcv: List[List[float]]) -> None:
    """
    Store OHLCV history for a symbol (useful for when PriceFetcher can pass historical candles).
    Each ohlcv row is [timestamp_ms, open, high, low, close, volume]
    """
    if not ohlcv:
        return
    _symbol_history[symbol] = list(ohlcv[-1000:])  # keep bounded

def clear_symbol_history(symbol: str) -> None:
    _symbol_history.pop(symbol, None)

def explain_signal(decision: Dict) -> str:
    """Return a human readable explanation of why decision fired."""
    parts = [f"{decision.get('filtered', 'HOLD')} ({decision.get('confidence',0)}%)"]
    comps = decision.get("components", {})
    for k, v in comps.items():
        parts.append(f"{k}:{v}")
    return " | ".join(parts)

# === Indicator helpers (use pandas+ta if available) ===

def _compute_indicators_from_history(ohlcv: List[List[float]]) -> Optional[Dict]:
    """
    Expect ohlcv as list of [ts,o,h,l,c,v] with newest last.
    Returns dict with: rsi, ema_fast, ema_slow, ema_regime_short, ema_regime_long, macd, macd_signal, atr
    """
    if not ohlcv or len(ohlcv) < 6:
        return None
    if not HAS_TA:
        # fallback: simple computations
        closes = [row[4] for row in ohlcv]
        highs = [row[2] for row in ohlcv]
        lows = [row[3] for row in ohlcv]
        # simple RSI approximation not implemented here: return None to force conservative path
        return None

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=EMA_FAST).ema_indicator()
        df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=EMA_SLOW).ema_indicator()
        df['ema50'] = ta.trend.EMAIndicator(df['close'], window=EMA_REGIME_SHORT).ema_indicator()
        df['ema200'] = ta.trend.EMAIndicator(df['close'], window=EMA_REGIME_LONG).ema_indicator()
        macd = ta.trend.MACD(df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        df['atr'] = atr
    except Exception as e:
        LOG.error(f"[strategy] TA compute error: {e}")
        return None

    # pick latest
    last = df.iloc[-1]
    indicators = {
        'rsi': float(last['rsi']) if not pd.isna(last['rsi']) else None,
        'ema_fast': float(last['ema_fast']) if not pd.isna(last['ema_fast']) else None,
        'ema_slow': float(last['ema_slow']) if not pd.isna(last['ema_slow']) else None,
        'ema50': float(last['ema50']) if not pd.isna(last['ema50']) else None,
        'ema200': float(last['ema200']) if not pd.isna(last['ema200']) else None,
        'macd': float(last['macd']) if not pd.isna(last['macd']) else None,
        'macd_signal': float(last['macd_signal']) if not pd.isna(last['macd_signal']) else None,
        'atr': float(last['atr']) if not pd.isna(last['atr']) else None
    }
    return indicators

# simple candle pattern checks (micro confirmation)
def _candle_pattern_check(prices: Dict) -> Optional[str]:
    """
    Given single candle prices dict (open, high, low, close), try to detect a simple pattern:
    - bullish engulfing => 'BUY'
    - bearish engulfing => 'SELL'
    - hammer / shooting star => small signal
    This is intentionally conservative: returns None if unsure.
    """
    o = float(prices.get('open', 0.0))
    h = float(prices.get('high', 0.0))
    l = float(prices.get('low', 0.0))
    c = float(prices.get('close', 0.0))
    body = abs(c - o)
    total = h - l if (h - l) != 0 else 1e-9
    body_ratio = body / total

    # hammer-like: small body, long lower wick
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    if body_ratio < 0.35 and lower_wick > 2 * body:
        return 'BUY'
    if body_ratio < 0.35 and upper_wick > 2 * body:
        return 'SELL'
    # engulfing pattern requires prior candle -> can't detect here reliably
    return None

# wick/body noise filter
def _pass_noise_filters(prices: Dict) -> bool:
    o = float(prices.get('open', 0.0))
    h = float(prices.get('high', 0.0))
    l = float(prices.get('low', 0.0))
    c = float(prices.get('close', 0.0))
    body = abs(c - o)
    wick = (h - l) - body
    if body <= 0:
        return False
    ratio = wick / body if body != 0 else float('inf')
    if ratio > getattr(config, 'WICK_BODY_MAX_RATIO', 4.0):
        return False
    return True

# confidence combining helper
def _combine_confidence(components: Dict[str, float], sentiment: float) -> float:
    """
    components: dict name -> score in 0..1 (how strongly that component supports the direction)
    sentiment: -1..1 (we'll treat >0 as boost for BUY, <0 for SELL)
    returns 0..1
    """
    # weighted average by WEIGHTS; normalize by total weight
    total_weight = 0.0
    score = 0.0
    for k, v in components.items():
        w = WEIGHTS.get(k, 1.0)
        total_weight += w
        score += (v * w)
    if total_weight == 0:
        base = 0.0
    else:
        base = score / total_weight  # in 0..1 roughly

    # sentiment multiplier: +/-
    s_weight = WEIGHTS.get('sentiment', 1.0)
    # map sentiment to multiplier: small influence (0.75..1.25)
    sentiment_mult = 1.0 + (sentiment * 0.25)
    base = base * sentiment_mult

    # clamp
    base = max(0.0, min(1.0, base))
    return base

# === Main generator ===

def generate_signal(symbol: str, prices: Dict, sentiment: float = 0.0) -> Optional[Dict]:
    """
    Main entry. Returns decision dict or None.
    prices: must contain 'open','high','low','close','volume' and may contain 'history' (list of ohlcv rows)
    sentiment: -1..1
    """
    try:
        # build base decision
        decision = {
            "symbol": symbol,
            "raw": "HOLD",
            "filtered": "HOLD",
            "confidence": 0,
            "strength": 0.0,
            "components": {}
        }

        # quick sanity
        if not prices or 'close' not in prices:
            return decision

        # if history passed in prices['history'], prefer it; otherwise see if module-level cache has it
        history = None
        if isinstance(prices.get('history', None), list):
            history = prices.get('history')
        elif symbol in _symbol_history:
            history = _symbol_history[symbol]

        indicators = None
        if history:
            indicators = _compute_indicators_from_history(history)

        # Conservative fallback path if no indicators: require clear candle+sentiment alignment
        if not indicators:
            # micro rules: candle direction + volume surge + sentiment
            o = float(prices.get('open', 0.0))
            c = float(prices.get('close', 0.0))
            vol = float(prices.get('volume', 0.0) or 0.0)
            avg_vol = float(prices.get('avg_volume', 0.0) or 0.0)
            # require noise pass
            if not _pass_noise_filters(prices):
                return decision

            candle_dir = 'BUY' if c > o else 'SELL' if c < o else 'HOLD'
            # require volume at least 1x avg_volume if avg known, else allow small trades
            vol_ok = True if avg_vol == 0 else (vol >= 0.8 * avg_vol)
            # require sentiment alignment
            sent_align = (sentiment > 0.15 and candle_dir == 'BUY') or (sentiment < -0.15 and candle_dir == 'SELL')
            # only produce a weak signal if both candle and sentiment align strongly
            if candle_dir in ('BUY', 'SELL') and vol_ok and sent_align:
                base_strength = 0.5  # weak baseline
                decision['raw'] = candle_dir
                decision['strength'] = base_strength
                decision['components'] = {
                    'candle': 1.0,
                    'volume': 1.0 if vol_ok else 0.0,
                    'sentiment': sentiment
                }
                conf = _combine_confidence({'candle':1.0,'vol':1.0}, sentiment)
                decision['confidence'] = _to_pct(conf)
                decision['filtered'] = candle_dir if conf >= MIN_CONF_TO_TRADE else 'HOLD'
                return decision
            else:
                return decision

        # --- when indicators are available: multi-layer analysis ---
        rsi = indicators.get('rsi') if indicators else None
        ema_fast = indicators.get('ema_fast') if indicators else None
        ema_slow = indicators.get('ema_slow') if indicators else None
        ema50 = indicators.get('ema50') if indicators else None
        ema200 = indicators.get('ema200') if indicators else None
        macd = indicators.get('macd') if indicators else None
        macd_sig = indicators.get('macd_signal') if indicators else None
        atr = indicators.get('atr') if indicators else None

        # component scores (0..1)
        comps = {}

        # Trend/regime (EMA50 vs EMA200)
        if ema50 and ema200:
            up_regime = ema50 > ema200
            down_regime = ema50 < ema200
            comps['trend'] = 1.0 if up_regime else 0.0 if down_regime else 0.5
        else:
            comps['trend'] = 0.5  # neutral if missing

        # Momentum (RSI)
        if rsi is not None:
            if rsi < RSI_OVERSOLD:
                comps['momentum'] = 1.0
            elif rsi > RSI_OVERBOUGHT:
                comps['momentum'] = 0.0
            else:
                # reward mid-range momentum in direction of price change minimally
                comps['momentum'] = _safe_div(RSI_NEUTRAL_HIGH - rsi, RSI_NEUTRAL_HIGH - RSI_NEUTRAL_LOW, 0.5) if rsi < RSI_NEUTRAL_HIGH else _safe_div(rsi - RSI_NEUTRAL_LOW, RSI_NEUTRAL_HIGH - RSI_NEUTRAL_LOW, 0.5)
                comps['momentum'] = max(0.0, min(1.0, comps['momentum']))
        else:comps['momentum'] = 0.5

        # MACD signal - simple crossover strength
        if macd is not None and macd_sig is not None:
            comps['macd'] = 1.0 if macd > macd_sig else 0.0 if macd < macd_sig else 0.5
        else:
            comps['macd'] = 0.5

        # Volume: require close above average volume for stronger signals (price_fetcher provides avg_volume)
        vol = float(prices.get('volume', 0) or 0)
        avg_vol = float(prices.get('avg_volume', 0) or 0)
        if avg_vol and vol:
            comps['vol'] = 1.0 if vol >= avg_vol * 1.0 else 0.5 if vol >= avg_vol * 0.6 else 0.0
        else:
            comps['vol'] = 0.5

        # Candle micro confirmation
        pat = _candle_pattern_check(prices)
        if pat == 'BUY':
            comps['candle'] = 1.0
        elif pat == 'SELL':
            comps['candle'] = 0.0
        else:
            comps['candle'] = 0.5

        # Noise filter (wick/body) - pass is 1.0 else 0
        comps['noise'] = 1.0 if _pass_noise_filters(prices) else 0.0

        # Combine a directional leaning score (0..1) for BUY bias. We'll compute both buy_score and sell_score.
        # Map components to a directional value: for components that are binary, interpret >0.5 as bullish; else bearish.
        buy_votes = 0.0
        sell_votes = 0.0

        # Trend: if trend==1 => buy, if 0 => sell
        if comps['trend'] >= 0.75:
            buy_votes += WEIGHTS['trend']
        elif comps['trend'] <= 0.25:
            sell_votes += WEIGHTS['trend']

        # Momentum: if momentum high (1) -> buy, if low (0) -> sell
        if comps['momentum'] >= 0.75:
            buy_votes += WEIGHTS['momentum']
        elif comps['momentum'] <= 0.25:
            sell_votes += WEIGHTS['momentum']

        # MACD
        if comps['macd'] >= 0.75:
            buy_votes += WEIGHTS['momentum'] * 0.8
        elif comps['macd'] <= 0.25:
            sell_votes += WEIGHTS['momentum'] * 0.8

        # Volume & candle support (multiply to bias)
        if comps['vol'] >= 0.75:
            buy_votes += WEIGHTS['vol']
        elif comps['vol'] <= 0.25:
            sell_votes += WEIGHTS['vol']

        # candle confirmation
        if comps['candle'] >= 0.75:
            buy_votes += WEIGHTS['candle']
        elif comps['candle'] <= 0.25:
            sell_votes += WEIGHTS['candle']

        # noise lowers confidence if failing
        if comps['noise'] == 0.0:
            # penalize both sides but keep decision conservative
            buy_votes *= 0.5
            sell_votes *= 0.5

        # base_strength measures absolute agreement among indicators (0..1)
        total_possible = WEIGHTS['trend'] + WEIGHTS['momentum'] + WEIGHTS['vol'] + WEIGHTS['candle']
        score_raw = max(0.0, (buy_votes - sell_votes) / (total_possible + 1e-9))  # -1..1
        # map to 0..1 confidence magnitude
        strength = abs(score_raw)
        # direction
        direction = 'HOLD'
        if buy_votes > sell_votes and strength > 0:
            direction = 'BUY'
        elif sell_votes > buy_votes and strength > 0:
            direction = 'SELL'
        else:
            direction = 'HOLD'

        # combine into confidence with sentiment
        # create per-component support dict (0..1)
        comp_for_conf = {
            "trend": comps['trend'],
            "momentum": comps['momentum'],
            "vol": comps['vol'],
            "candle": comps['candle'],
            "noise": comps['noise']
        }
        base_conf = _combine_confidence(comp_for_conf, sentiment)  # 0..1

        # bias by directional strength
        final_conf = base_conf * (0.6 + 0.4 * strength)  # prefer higher when strong agreement
        # avoid tiny signals: require minimum coverage
        if final_conf < 0.05:
            # too weak
            decision['filtered'] = 'HOLD'
            decision['confidence'] = 0
            decision['strength'] = float(strength)
            decision['components'] = comp_for_conf
            decision['raw'] = 'HOLD'
            return decision

        decision['raw'] = direction
        decision['strength'] = float(strength)
        decision['components'] = comp_for_conf

        pct_conf = _to_pct(final_conf)
        decision['confidence'] = pct_conf

        # Filter: if direction is BUY but trend strongly down -> block (safeguard)
        if direction == 'BUY' and comps['trend'] < 0.3:
            decision['filtered'] = 'HOLD'
        elif direction == 'SELL' and comps['trend'] > 0.7:
            decision['filtered'] = 'HOLD'
        else:
            # apply normalized threshold
            decision['filtered'] = direction if final_conf >= MIN_CONF_TO_TRADE else 'HOLD'

        return decision

    except Exception as e:
        LOG.error(f"[strategy] generate_signal exception for {symbol}: {e}")
        return {
            "symbol": symbol,
            "raw": "HOLD",
            "filtered": "HOLD",
            "confidence": 0,
            "strength": 0.0,
            "components": {}
        }