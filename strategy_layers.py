# strategy_layers.py
import logging
import statistics

logger = logging.getLogger(__name__)

def apply_filters(signal: dict, prices: dict) -> dict:
    """
    Apply advanced noise filters and return updated signal dict (keeps fields).
    If filtered out, sets signal['filtered'] = 'HOLD' and logs reason.
    """

    try:
        high = float(prices["high"])
        low = float(prices["low"])
        close = float(prices["close"])
        open_ = float(prices["open"])
        volume = float(prices.get("volume", 0) or 0)
        avg_vol = float(prices.get("avg_volume", volume) or volume)

        body = abs(close - open_)
        wick = max((high - low) - body, 0.0)

        # Wick filter (avoid fakeouts)
        if body > 0 and wick / body > 2:
            logger.info(f"Filtered {signal.get('raw')} for {signal.get('symbol')} due to long wick")
            signal['filtered'] = 'HOLD'
            signal['filter_reason'] = 'wick_long'
            return signal

        # Volatility filter (sudden huge candle size)
        # compute a small recent average range fallback
        try:
            avg_range = statistics.mean([abs(high - low) for _ in range(3)])  # naive fallback
        except Exception:
            avg_range = abs(high - low)
        if avg_range > 0 and (high - low) > 3 * avg_range:
            logger.info(f"Filtered {signal.get('raw')} for {signal.get('symbol')} due to volatility spike")
            signal['filtered'] = 'HOLD'
            signal['filter_reason'] = 'vol_spike'
            return signal

        # Volume anomaly filter
        if avg_vol > 0 and volume > 2 * avg_vol:
            logger.info(f"Filtered {signal.get('raw')} for {signal.get('symbol')} due to abnormal volume")
            signal['filtered'] = 'HOLD'
            signal['filter_reason'] = 'vol_anomaly'
            return signal

        # Passed all filters: keep original 'raw' for now; leave filtered field untouched here
        return signal

    except Exception as e:
        logger.error(f"Filter error: {e}")
        signal['filtered'] = 'HOLD'
        signal['filter_reason'] = 'filter_exception'
        return signal


def calculate_confidence(signal: dict) -> int:
    """
    Confidence scoring based on the full signal dict.
    Input:
      signal: {
        'symbol':..., 'raw':'BUY'|'SELL'|'HOLD', 'filtered':'BUY'|'SELL'|'HOLD',
        'strength': float >=0 (e.g. 1.0 baseline), 'sentiment': -1..1
      }
    Returns: 0..100 integer
    """
    try:
        base = 50

        filtered = signal.get("filtered", signal.get("raw", "HOLD"))
        strength = float(signal.get("strength", 1.0))

        if filtered == "BUY" or filtered == "SELL":
            base += 20
        else:
            base -= 20

        # Strength should scale gently; avoid extreme jumps
        # strength usually is around 1.0; apply a soft multiplier
        adj = base * max(0.0, min(3.0, strength))

        # Sentiment is already in signal['sentiment'] if present - use small modifier
        sentiment = float(signal.get("sentiment", 0.0))
        adj += sentiment * 10.0  # ±10 points at most for strong sentiment

        # Clamp between 0 and 100
        conf = max(0, min(100, int(round(adj))))
        return conf

    except Exception as e:
        logger.error(f"Confidence scoring error: {e}")
        return 0