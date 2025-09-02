# ================
# logger.py
# Rotating file logger + helpers
# ================

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import config

_logger = None

def _build_logger():
    logger = logging.getLogger("bot")
    logger.setLevel(logging.INFO)

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(fmt)

    # File (rotating)
    fh = RotatingFileHandler(config.LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)
    return logger

def get_logger():
    global _logger
    if _logger is None:
        _logger = _build_logger()
    return _logger

def log_info(msg: str):
    get_logger().info(msg)

def log_warning(msg: str):
    get_logger().warning(msg)

def log_error(msg: str):
    get_logger().error(msg, exc_info=config.PRINT_STACK_TRACES)

def log_signal(symbol: str, side: str, price: float, rsi: float = None, confidence: float = None):
    ts = datetime.utcnow().isoformat()
    parts = [f"{ts} | SIGNAL | {symbol} | {side.upper()} | price={price:.6f}"]
    if rsi is not None:
        parts.append(f"rsi={rsi:.2f}")
    if confidence is not None:
        parts.append(f"conf={confidence:.2f}")
    line = " | ".join(parts)
    get_logger().info(line)
    try:
        with open(config.SIGNALS_LOG, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        get_logger().error(f"Failed to write to signals log: {e}", exc_info=config.PRINT_STACK_TRACES)

def log_trade(event: str, symbol: str, side: str, amount: float, price: float,
              pnl: float = None, meta: dict = None):
    ts = datetime.utcnow().isoformat()
    parts = [f"{ts} | TRADE | {event} | {symbol} | {side.upper()} | amount={amount:.6f} | price={price:.6f}"]
    if pnl is not None:
        parts.append(f"pnl={pnl:.6f}")
    if meta:
        parts.append(f"meta={meta}")
    line = " | ".join(parts)
    get_logger().info(line)
    try:
        with open(config.TRADES_LOG, "a") as f:
            f.write(line + "\n")
    except Exception as e:
        get_logger().error(f"Failed to write to trades log: {e}", exc_info=config.PRINT_STACK_TRACES)