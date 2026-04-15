import time
import threading
from services.asset_resolver import resolve_asset
from services.feature_builder import build_features_from_history
from crypto_app.ml.predict import predict_pump
from crypto_app.models import LiveSignal
import logging

logger = logging.getLogger(__name__)

WATCHLIST = [
    ("BTC", "crypto"),
    ("ETH", "crypto"),
    ("BNB", "crypto"),
    ("XRP", "crypto"),
    ("ADA", "crypto"),
    ("DOGE", "crypto"),
    ("SOL", "crypto"),
    ("MATIC", "crypto"),
    ("DOT", "crypto"),
    ("LTC", "crypto"),

    ("RELIANCE.NS", "stock"),
    ("TCS.NS", "stock"),
    ("INFY.NS", "stock"),
    ("HDFCBANK.NS", "stock"),
    ("ICICIBANK.NS", "stock"),
]

INTERVAL = 90 # seconds


def scan_once(symbol):
    try:
        # Resolve asset and fetch history
        # If get_crypto_data fails, resolve_asset will return None/None
        asset, history = resolve_asset(symbol)

        # ❌ STOP FALLBACK CORRUPTION
        # If data is missing or history is empty, we fail early.
        if not asset or history is None or history.empty:
            logger.warning(f"Skipping {symbol} — no valid data available (Reality Check)")
            return

        # Feature Engineering (Normalization Layer included inside)
        rf, seq = build_features_from_history(
            history,
            symbol=asset.symbol,
            market_type=asset.market_type
        )
        logger.info(f"{symbol} price={rf['price']} volume={rf['volume']}")


        # AI Prediction
        result = predict_pump({**rf, "sequence": seq})

        # Save Real-Time Signal to DB
        LiveSignal.objects.create(
            symbol=asset.symbol,
            market_type=asset.market_type,
            risk_score=result["risk_score"],
            anomaly_score=result["anomaly_score"],
            confidence=result["confidence"]
        )

        logger.info(f"Live updated: {symbol} | Risk: {result['risk_score']:.2f}")

    except Exception as e:
        logger.error(f"Live scan failed: {symbol} | Error: {str(e)}")


def run_engine():
    while True:
        for symbol, _ in WATCHLIST:
            scan_once(symbol)

        time.sleep(INTERVAL)


def start_live_engine():
    thread = threading.Thread(target=run_engine, daemon=True)
    thread.start()