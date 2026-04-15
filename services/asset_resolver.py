from .normalizer import normalize_ticker
from .market_data import (
    get_stock_data,
    get_crypto_data,
    get_stock_history,
    get_crypto_history
)
from crypto_app.models import Asset


# ==========================================
# RESOLVE ASSET (STRICT)
# ==========================================
def resolve_asset(ticker):
    try:
        norm = normalize_ticker(ticker)

        # ================= CRYPTO =================
        if norm["type"] == "crypto":
            data = get_crypto_data(norm["symbol"])

            if data is None or data.get("price") in [None, 0]:
                raise ValueError(f"Crypto data unavailable for {norm['symbol']}")

            asset, _ = Asset.objects.get_or_create(
                symbol=norm["symbol"],
                market_type="crypto"
            )

            try:
                history = get_crypto_history(norm["symbol"])
            except Exception:
                history = None

            return asset, history

        # ================= STOCK =================
        if norm["type"] == "stock":
            data = get_stock_data(norm["symbol"])

            if data is None or data.get("price") in [None, 0]:
                raise ValueError(f"Stock data unavailable for {norm['symbol']}")

            asset, _ = Asset.objects.get_or_create(
                symbol=norm["symbol"],
                market_type="stock"
            )

            try:
                history = get_stock_history(norm["symbol"])
            except Exception:
                history = None

            return asset, history

    except Exception as e:
        print(f"Resolver error: {e}")

    return None, None