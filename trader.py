# trader.py — live trading helper (respects VIRTUAL_TRADING flag)

import ccxt
import time
import config
import logger

# Build exchange only if we might place live orders
_exchange = None
def _get_exchange():
    global _exchange
    if _exchange is None:
        _exchange = ccxt.kucoin({
            'apiKey': getattr(config, 'API_KEY', ''),
            'secret': getattr(config, 'API_SECRET', ''),
            'password': getattr(config, 'API_PASSPHRASE', ''),
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
    return _exchange

def _safe_fetch_balance():
    try:
        return _get_exchange().fetch_balance()
    except Exception as e:
        logger.log_error(f"Live balance fetch failed: {e}")
        return {'total': {}}

def calculate_dynamic_amount(symbol: str) -> float:
    """
    % of available USDT based on TRADE_AMOUNT_PERCENT.
    Returns base currency amount (e.g., BTC) rounded to 6 dp.
    """
    try:
        if getattr(config, 'VIRTUAL_TRADING', True):
            # When simulating, sizing is handled in virtual_trader.
            return 0.0

        ex = _get_exchange()
        bal = _safe_fetch_balance()
        usdt = float(bal.get('total', {}).get('USDT', 0) or 0)
        if usdt <= 0:
            return 0.0

        pct = float(getattr(config, 'TRADE_AMOUNT_PERCENT', 25)) / 100.0
        spend = usdt * pct
        ticker = ex.fetch_ticker(symbol)
        last = float(ticker['last'])
        qty = max(0.0, round(spend / last, 6))
        return qty
    except Exception as e:
        logger.log_error(f"calculate_dynamic_amount error: {e}")
        return 0.0

def place_order(symbol: str, side: str, amount: float) -> None:
    """
    Places a market order if VIRTUAL_TRADING=False.
    """
    try:
        if getattr(config, 'VIRTUAL_TRADING', True):
            logger.log_info(f"[LIVE DISABLED] Would place {side.upper()} {amount} {symbol}")
            return

        if amount <= 0:
            logger.log_warning(f"Skip live order {symbol}: amount<=0")
            return

        _get_exchange().create_order(symbol=symbol, type='market', side=side, amount=amount)
        logger.log_info(f"✅ LIVE {side.upper()} {amount} {symbol}")
        time.sleep(0.5)  # gentle pacing
    except Exception as e:
        logger.log_error(f"Live order failed {symbol} {side} {amount}: {e}")    