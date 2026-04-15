# crypto_app/services/market_data.py

import requests
import yfinance as yf
import time
from .utils import retry, safe_execute
import pandas as pd
import logging
logger = logging.getLogger(__name__)

CACHE = {}
CACHE_TTL = 10


def get_cached(key):
    data = CACHE.get(key)
    if data:
        value, ts = data
        if time.time() - ts < CACHE_TTL:
            return value
    return None


def set_cache(key, value):
    CACHE[key] = (value, time.time())


# ===============================
# 📈 STOCK FETCH (FIXED)
# ===============================
@retry()
def _fetch_stock(symbol):
    time.sleep(0.3)

    stock = yf.Ticker(symbol)

    hist = stock.history(period="5d", interval="1h")

    if hist.empty or len(hist) < 2:
        raise ValueError(f"No data for {symbol}")

    latest = hist.iloc[-1]
    prev = hist.iloc[-2]

    price = float(latest["Close"])
    prev_price = float(prev["Close"])

    price_change = (price - prev_price) / max(prev_price, 1e-9)

    returns = hist["Close"].pct_change()
    volatility = float(returns.std())

    return {
        "price": price,
        "volume": float(latest["Volume"]),
        "price_change": price_change,
        "volatility": abs(volatility)
    }


def get_stock_data(symbol):
    cached = get_cached(symbol)
    if cached:
        return cached

    data = safe_execute(lambda: _fetch_stock(symbol), fallback=None)

    if data:
        set_cache(symbol, data)
        return data

    # ❌ NO FALLBACK
    logger.error(f"Stock data unavailable: {symbol}")
    return None


# ===============================
# 🪙 CRYPTO FETCH (CLEAN)
# ===============================
@retry()
def _fetch_crypto(symbol):
    symbol = symbol.strip().lower()

   

    url = "https://api.coingecko.com/api/v3/simple/price"

    try:
        time.sleep(1.2) 
        res = requests.get(
            url,
            params={
                "ids": symbol,
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
                "include_24hr_change": "true"
            },
            timeout=5
        )
        if res.status_code == 429:
            raise Exception("RATE_LIMIT")
        
        if res.status_code != 200:
            raise ValueError(f"API error {res.status_code}")

        json_data = res.json()

        if symbol not in json_data:
            raise ValueError(f"Missing data: {symbol}")

        data = json_data[symbol]

        return {
            "price": float(data.get("usd", 0)),
            "volume": float(data.get("usd_24h_vol", 0)),
            "price_change": float(data.get("usd_24h_change", 0)) / 100,
            "volatility": abs(float(data.get("usd_24h_change", 0))) / 100
        }

    except Exception as e:
        raise ValueError(f"Crypto fetch failed: {symbol} | {str(e)}")


def get_crypto_data(symbol):
    symbol = symbol.lower()

    # 1. Check Cache
    cached = get_cached(symbol)
    if cached:
        return cached

    # 2. Fetch Data
    data = safe_execute(lambda: _fetch_crypto(symbol), fallback=None)

    # 3. Handle Result
    if data:
        set_cache(symbol, data)
        return data

    # ❌ TOTAL FAILURE (Removed fallback to bitcoin)
    # If data fails, we fail. Don't fake reality.
    return None


# ===============================
# HISTORY (UNCHANGED)
# ===============================
def get_stock_history(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="60d", interval="1h")

    if hist.empty:
        raise ValueError("No historical data")

    df = hist.reset_index()
    df = df.rename(columns={
        "Close": "price",
        "Volume": "volume"
    })

    return df[["price", "volume"]]


def get_crypto_history(symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"

    res = requests.get(url, params={
        "vs_currency": "usd",
        "days": 60
    }, timeout=10)

    data = res.json()

    prices = data.get("prices", [])
    volumes = data.get("total_volumes", [])

    if not prices:
        raise ValueError("No crypto history")

    df = pd.DataFrame({
        "price": [p[1] for p in prices],
        "volume": [v[1] for v in volumes]
    })

    return df