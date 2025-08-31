
# ================
# config.py  (Phase 2 complete foundation)
# ================

# --- Exchange API (KuCoin) ---
API_KEY = '---'
API_SECRET = '-----'
API_PASSPHRASE = '------'

USE_SANDBOX = False  # KuCoin sandbox requires separate keys if you use it

# --- Telegram ---
TELEGRAM_BOT_TOKEN = '--:--'
TELEGRAM_CHAT_ID = '--'
TELEGRAM_ENABLED = True  # flip False to silence alerts

# --- News / Sentiment (Phase 2.5 ready) ---
NEWSDATA_API_KEY = "---"  # newsdata.io
NEWS_ENABLED = True
SENTIMENT_THRESHOLD = 0.30   # min absolute sentiment to matter
SENTIMENT_WEIGHT = 0.25      # boost cap (0..1) applied to confidence

# --- Timeframes & candles ---
TIMEFRAMES = ["5m", "15m", "1h", "4h"]
DEFAULT_LIMIT = 200  # candles per timeframe (fetch depth)

# --- Bot behavior ---
MAX_TRADES_PER_DAY = 10
REENTRY_COOLDOWN_MIN = 30       # min minutes before same symbol re-entry
CONFIDENCE_THRESHOLD = 0.65     # required to place a trade
SMART_FILTERS_ENABLED = True    # wick/body, volume, flatness filters

# --- Position sizing (live) ---
TRADE_AMOUNT_PERCENT = 25.0     # % of available USDT per live trade

# --- Virtual trading (demo/paper) ---
VIRTUAL_TRADING = True
VIRTUAL_START_BALANCE = 1000.0
VIRTUAL_FEE_RATE = 0.001        # 0.10% taker assumed in simulation

# --- Risk management (ATR TP/SL + partials + trailing) ---
ATR_PERIOD = 14
ATR_TP_MULT = 1.2
ATR_SL_MULT = 1.0
MIN_TP_SL_DIST = 0.002          # 0.2% minimum spacing safeguard
OPPOSITE_SIGNAL_EXIT = True

# Partial take-profits: list of (R multiple, fraction_to_close)
PARTIAL_TP_LEVELS = [(0.5, 0.25), (1.0, 0.25), (1.5, 0.25)]
TRAIL_STOP_AT_R = 1.0           # start trailing when R >= 1.0
TRAIL_MULT_ATR = 1.0            # trail distance = ATR * this

# --- Noise filter thresholds ---
WICK_BODY_MAX_RATIO = 2.5
VOL_ANOMALY_ZSCORE = 2.0
MIN_VOL_PCT_OF_AVG = 0.35
MIN_INTRADAY_ATR_PCT = 0.10     # ATR/price < 10% considered too flat for crypto

# --- Markets (Phase 2 adds Yahoo-backed symbols) ---
# Crypto on KuCoin (spot)
CRYPTO_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT",
    "XMR/USDT", "LTC/USDT", "DOT/USDT", "AVAX/USDT", "ALGO/USDT",
    "TRX/USDT", "APT/USDT", "INJ/USDT"
]

# Yahoo Finance tickers (indices/commodities/stocks/ETFs)
# Examples: gold spot (XAUUSD), S&P 500 ETF (SPY), Apple (AAPL)
YAHOO_SYMBOLS = [
    "XAUUSD=X",   # Gold spot (USD)
    "SPY",        # S&P 500 ETF
    "AAPL",       # Apple
    "NVDA",       # Nvidia
    "QQQ"         # Nasdaq 100 ETF
]

# Set True to enable Yahoo-backed markets in virtual trading & signals
ENABLE_YAHOO = True

# --- File paths (logs & state) ---
LOG_FILE = "bot.log"
SIGNALS_LOG = "signals.log"
TRADES_LOG = "virtual_trades.csv"
OPEN_POSITIONS_FILE = "OpenPositions.json"   # optional legacy
VIRTUAL_STATE_FILE = "VirtualState.json"
VIRTUAL_META_FILE = "VirtualMeta.json"

# --- Debug ---
PRINT_STACK_TRACES = False