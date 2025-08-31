# test_connection.py

import ccxt
import config

exchange = ccxt.kucoin({
    'apiKey': config.API_KEY,
    'secret': config.API_SECRET,
    'password': config.API_PASSPHRASE,
    'enableRateLimit': True
})

try:
    balance = exchange.fetch_balance()
    print("✅ Connected to KuCoin.")
    print("💰 USDT balance:", balance['total'].get('USDT', 0))
except Exception as e:
    print("❌ Connection failed:", str(e))