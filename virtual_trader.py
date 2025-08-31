# virtual_trader.py — corrected version (bugfixes + robustness)

import json
import csv
from datetime import datetime
import math

import config
import logger
try:
    import notifier
except Exception:
    notifier = None  # optional

# ---------- Config helpers & paths ----------
STATE_FILE = getattr(config, 'VIRTUAL_STATE_FILE', 'virtual_state.json')
META_FILE  = getattr(config, 'VIRTUAL_META_FILE',  'virtual_meta.json')
TRADES_CSV = getattr(config, 'TRADES_LOG',         'virtual_trades.csv')

FEE = float(getattr(config, 'VIRTUAL_FEE_RATE', 0.001))
START_BAL = float(getattr(config, 'VIRTUAL_START_BALANCE', 1000.0))
TP_MULT = float(getattr(config, 'ATR_TP_MULT', 1.2))
SL_MULT = float(getattr(config, 'ATR_SL_MULT', 1.0))
MIN_DIST = float(getattr(config, 'MIN_TP_SL_DIST', 0.002))  # as fraction, e.g. 0.002 = 0.2%
PARTIALS = list(getattr(config, 'PARTIAL_TP_LEVELS', [(0.5, 0.25), (1.0, 0.25), (1.5, 0.25)]))
OPPOSITE_EXIT = bool(getattr(config, 'OPPOSITE_SIGNAL_EXIT', True))
TRAIL_AT_R = float(getattr(config, 'TRAIL_STOP_AT_R', 1.0))
TRAIL_ATR_MULT = float(getattr(config, 'TRAIL_MULT_ATR', 1.0))

# ---------- Utilities ----------
def _utcnow() -> str:
    return datetime.utcnow().isoformat()

def _safe_read_json(path: str, default):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path: str, payload) -> None:
    try:
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
    except Exception:
        # best-effort only
        pass

def _append_trade_csv(row: dict):
    new_file = False
    try:
        with open(TRADES_CSV, 'r'):
            pass
    except Exception:
        new_file = True
    with open(TRADES_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp','action','symbol','side','amount','price',
            'realized_pnl','reason','balance_after'
        ])
        if new_file:
            writer.writeheader()
        writer.writerow(row)

# ---------- Engine ----------
class VirtualTrader:
    def __init__(self):
        meta = _safe_read_json(META_FILE, {"balance": START_BAL, "trade_count": 0})
        state = _safe_read_json(STATE_FILE, {"balance": START_BAL, "positions": {}, "trade_history": []})

        self.balance = float(state.get('balance', meta.get('balance', START_BAL)))
        self.positions = state.get('positions', {})   # symbol -> dict
        self.trade_history = state.get('trade_history', [])
        self.trade_count = int(meta.get('trade_count', 0))

        # normalize positions
        for s, p in list(self.positions.items()):
            p.setdefault('side', 'buy')
            p.setdefault('amount', 0.0)
            p.setdefault('entry_price', 0.0)
            p.setdefault('tp', None)
            p.setdefault('sl', None)
            p.setdefault('opened_at', _utcnow())
            p.setdefault('confidence', 0.0)
            p.setdefault('partials', [])   # reached partial R levels
            p.setdefault('trail_active', False)

        self._persist_full()

    # ---------- Persistence ----------
    def _persist_full(self):
        _safe_write_json(STATE_FILE, {
            "balance": round(self.balance, 8),
            "positions": self.positions,
            "trade_history": self.trade_history[-5000:],  # cap growth
        })
        _safe_write_json(META_FILE, {
            "balance": round(self.balance, 8),
            "trade_count": self.trade_count
        })

    # ---------- Sizing ----------
    def _calc_amount_for_price(self, price: float) -> float:
        """Use TRADE_AMOUNT_PERCENT of cash balance."""
        if price <= 0:
            return 0.0
        pct = float(getattr(config, 'TRADE_AMOUNT_PERCENT', 25)) / 100.0
        spend = max(0.0, self.balance * pct)
        qty = max(0.0, round(spend / price, 6))
        return qty

    # ---------- TP/SL helpers ----------
    def _derive_tp_sl(self, entry: float, side: str, atr: float | None) -> tuple:
        """
        Returns (tp, sl, r_per_point).
        If ATR missing, falls back to MIN_DIST% of price.
        """
        base = atr if atr and atr > 0 else max(MIN_DIST * entry, 1e-8)

        if side == 'buy':
            tp = entry + max(base * TP_MULT, MIN_DIST * entry)
            sl = entry - max(base * SL_MULT, MIN_DIST * entry)
            r_value = entry - sl
        else:
            tp = entry - max(base * TP_MULT, MIN_DIST * entry)
            sl = entry + max(base * SL_MULT, MIN_DIST * entry)
            r_value = sl - entry

        r_value = max(r_value, 1e-8)
        return (round(tp, 8), round(sl, 8), r_value)

    # ---------- Public API ----------
    def can_trade(self, symbol: str) -> bool:
        return symbol not in self.positions

    def open_position(self, symbol: str, side: str, price: float, confidence: float = 0.0,
                      atr: float | None = None, amount: float | None = None):
        if symbol in self.positions:
            logger.log_warning(f"[VIRTUAL] {symbol} already open, skip.")
            return False
        if price <= 0:
            logger.log_warning(f"[VIRTUAL] Bad price for {symbol}, skip.")
            return False

        amt = amount if amount and amount > 0 else self._calc_amount_for_price(price)
        if amt <= 0:
            logger.log_warning(f"[VIRTUAL] No buying power for {symbol}.")
            return False

        cost = price * amt
        fee_cost = cost * FEE
        if cost + fee_cost > self.balance + 1e-12:
            logger.log_warning(f"[VIRTUAL] Insufficient balance for {symbol}. Need {cost+fee_cost:.4f}, have {self.balance:.4f}")
            return False

        tp, sl, r_per_point = self._derive_tp_sl(price, side, atr)

        self.balance -= (cost + fee_cost)
        self.positions[symbol] = {
            "side": side,
            "amount": amt,
            "entry_price": price,
            "tp": tp,
            "sl": sl,
            "opened_at": _utcnow(),
            "confidence": float(confidence),
            "r_per_point": r_per_point,
            "partials": [],    # list of R levels already taken
            "trail_active": False
        }
        self.trade_count += 1

        # history & csv
        evt = {"timestamp": _utcnow(), "action": "open", "symbol": symbol, "side": side,
               "amount": amt, "price": price}
        self.trade_history.append(evt)
        _append_trade_csv({
            "timestamp": evt["timestamp"], "action": "open", "symbol": symbol, "side": side,
            "amount": amt, "price": price, "realized_pnl": "", "reason": "open",
            "balance_after": round(self.balance, 8)
        })

        logger.log_info(f"📈 [VIRTUAL] OPEN {side.upper()} {symbol} amt={amt:.6f} entry={price:.6f} TP={tp} SL={sl} conf={confidence:.2f}")
        if notifier and getattr(config, 'TELEGRAM_ENABLED', True):
            try:
                notifier.send_telegram_message(
                    f"📈 <b>VIRTUAL OPEN</b>\n{symbol} | {side.upper()}\n"
                    f"Amount: {amt:.6f}\nEntry: {price:.6f}\nTP: {tp}\nSL: {sl}\nConf: {confidence:.2f}"
                )
            except Exception:
                pass

        self._persist_full()
        return True

    def _maybe_partial_take(self, symbol: str, price: float):
        """Execute partial exits when R thresholds reached."""
        pos = self.positions.get(symbol)
        if not pos:
            return

        side = pos["side"]
        entry = pos["entry_price"]
        amount = pos["amount"]
        r_val = pos["r_per_point"]
        reached_R = 0.0

        if r_val <= 0:
            return

        if side == "buy":
            reached_R = (price - entry) / r_val
        else:
            reached_R = (entry - price) / r_val

        for r_level, frac in PARTIALS:
            if r_level in pos.get("partials", []):
                continue
            if reached_R >= r_level and amount > 0:
                cut = round(amount * frac, 6)
                if cut <= 0:
                    continue
                # call private slice closer
                self._close_slice(symbol, cut, price, reason=f"partial_{r_level}R")
                pos.setdefault("partials", []).append(r_level)

    def _update_trailing_stop(self, symbol: str, price: float, atr: float | None):
        pos = self.positions.get(symbol)
        if not pos:
            return

        # Activate trailing after TRAIL_AT_R
        side = pos["side"]
        entry = pos["entry_price"]
        r_val = pos["r_per_point"]
        if TRAIL_AT_R <= 0:
            return
        cur_R = (price - entry) / r_val if side == "buy" else (entry - price) / r_val
        if cur_R < TRAIL_AT_R:
            return

        base_dist = (atr if atr and atr > 0 else entry * MIN_DIST) * TRAIL_ATR_MULT
        if side == "buy":
            new_sl = max(pos["sl"], price - base_dist)
            if new_sl > pos["sl"]:
                pos["sl"] = round(new_sl, 8)
        else:
            new_sl = min(pos["sl"], price + base_dist)
            if new_sl < pos["sl"]:
                pos["sl"] = round(new_sl, 8)

    def _close_slice(self, symbol: str, amount: float, price: float, reason: str):
        """Close part of a position."""
        pos = self.positions.get(symbol)
        if not pos or amount <= 0:
            return

        amount = min(amount, pos["amount"])
        side = pos["side"]
        entry = pos["entry_price"]

        # Proportional cost basis and PnL
        gross = price * amount
        fee = (price + entry) * amount * FEE * 0.5  # approx both sides
        if side == "buy":
            pnl = (price - entry) * amount - fee
        else:
            pnl = (entry - price) * amount - fee

        self.balance += gross
        pos["amount"] = round(pos["amount"] - amount, 6)

        evt = {
            "timestamp": _utcnow(), "action": "close", "symbol": symbol, "side": side,
            "amount": amount, "price": price, "realized_pnl": round(pnl, 8),
            "reason": reason
        }
        self.trade_history.append(evt)
        _append_trade_csv({
            "timestamp": evt["timestamp"], "action": "close", "symbol": symbol, "side": side,
            "amount": amount, "price": price, "realized_pnl": evt["realized_pnl"],
            "reason": reason, "balance_after": round(self.balance, 8)
        })

        logger.log_info(f"📉 [VIRTUAL] CLOSE {side.upper()} {symbol} amt={amount:.6f} px={price:.6f} pnl={pnl:.6f} reason={reason}")
        if notifier and getattr(config, 'TELEGRAM_ENABLED', True):
            try:
                notifier.send_telegram_message(
                    f"📉 <b>VIRTUAL CLOSE</b>\n{symbol} | {side.upper()}\n"
                    f"Amount: {amount:.6f}\nPrice: {price:.6f}\nPnL: {pnl:.6f}\nReason: {reason}\n"
                    f"Bal: {self.balance:.4f}"
                )
            except Exception:
                pass

        if pos["amount"] <= 0:
            # fully closed
            self.positions.pop(symbol, None)

    def close_position_full(self, symbol: str, price: float, reason: str):
        pos = self.positions.get(symbol)
        if not pos:
            return
        amt = pos["amount"]
        if amt > 0:
            self._close_slice(symbol, amt, price, reason)

    # ---------- Ticking ----------
    def update_positions(self, prices: dict, atr_map: dict | None = None):
        """
        prices: { 'BTC/USDT': 62000.0, ... }
        atr_map: optional {symbol: atr_value}
        """
        for symbol, pos in list(self.positions.items()):
            if symbol not in prices:
                continue
            try:
                price = float(prices[symbol])
            except Exception:
                continue
            side = pos["side"]

            # Check TP/SL
            hit_tp = (side == "buy" and pos["tp"] is not None and price >= pos["tp"]) or \
                     (side == "sell" and pos["tp"] is not None and price <= pos["tp"])
            hit_sl = (side == "buy" and pos["sl"] is not None and price <= pos["sl"]) or \
                     (side == "sell" and pos["sl"] is not None and price >= pos["sl"])

            if hit_tp:
                self.close_position_full(symbol, price, "TP")
                continue
            if hit_sl:
                self.close_position_full(symbol, price, "SL")
                continue

            # Partial exits
            self._maybe_partial_take(symbol, price)

            # Trailing stop
            atr = (atr_map or {}).get(symbol)
            self._update_trailing_stop(symbol, price, atr)

        self._persist_full()

    # ---------- Signal-driven interface ----------
    def handle_signal(self, symbol: str, decision: dict, price: float, atr: float | None = None):
        """
        decision: dict from strategy.generate_signal(...)
          expects keys: filtered ("BUY"/"SELL"/"HOLD"), confidence (0..100)
        """
        act = (decision or {}).get("filtered", "HOLD")
        conf = float((decision or {}).get("confidence", 0))
        if act == "HOLD":
            return

        # Opposite signal exit
        if OPPOSITE_EXIT and symbol in self.positions:
            cur_side = self.positions[symbol]["side"]
            if (cur_side == "buy" and act == "SELL") or (cur_side == "sell" and act == "BUY"):
                self.close_position_full(symbol, price, "opposite_signal")

        # One position per symbol
        if symbol in self.positions:
            return  # already handled above or still same side -> ignore

        side = "buy" if act == "BUY" else "sell"
        self.open_position(symbol, side, price, confidence=conf, atr=atr)

# create a module-level instance and backward-compatible alias
_instance = VirtualTrader()
virtual_trader = _instance
def update_positions(prices: dict, atr_map: dict | None = None): return _instance.update_positions(prices, atr_map)
def open_position(*args, **kwargs): return _instance.open_position(*args, **kwargs)
def close_position_full(*args, **kwargs): return _instance.close_position_full(*args, **kwargs)
def can_trade(symbol: str) -> bool: return _instance.can_trade(symbol)
def handle_signal(*args, **kwargs): return _instance.handle_signal(*args, **kwargs)
