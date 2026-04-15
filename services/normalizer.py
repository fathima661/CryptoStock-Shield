# crypto_app/services/normalizer.py

# ==========================================
# 🪙 CRYPTO SYMBOL MAP (CoinGecko IDs)
# ==========================================
CRYPTO_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "SOL": "solana",
    "MATIC": "polygon",
    "DOT": "polkadot",
    "LTC": "litecoin",
}

# ==========================================
# 📈 STOCK NAME / SHORTCODE MAP
# ==========================================
STOCK_MAP = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "INFY": "INFY.NS",
    "HDFC BANK": "HDFCBANK.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICI BANK": "ICICIBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
}


# ==========================================
# 🔥 NORMALIZE TICKER (STRICT BUT FRIENDLY)
# ==========================================
def normalize_ticker(ticker: str):
    if not ticker:
        raise ValueError("Ticker is empty")

    raw = ticker.strip().upper()

    # ===============================
    # 🪙 CRYPTO
    # ===============================
    if raw in CRYPTO_MAP:
        return {
            "type": "crypto",
            "symbol": CRYPTO_MAP[raw],
            "original": raw
        }

    # ===============================
    # 📈 STOCK FRIENDLY INPUT
    # Supports: Reliance, TCS, Infosys, HDFC Bank...
    # ===============================
    if raw in STOCK_MAP:
        return {
            "type": "stock",
            "symbol": STOCK_MAP[raw],
            "exchange": "NSE",
            "original": raw
        }

    # ===============================
    # 📈 STOCK DIRECT INPUT
    # Supports: RELIANCE.NS, TCS.NS, INFY.NS
    # ===============================
    if raw.endswith(".NS"):
        return {
            "type": "stock",
            "symbol": raw,
            "exchange": "NSE",
            "original": raw
        }

    if raw.endswith(".BO"):
        return {
            "type": "stock",
            "symbol": raw,
            "exchange": "BSE",
            "original": raw
        }

    # ❌ HARD BLOCK
    raise ValueError(f"Invalid or unsupported ticker: {raw}")