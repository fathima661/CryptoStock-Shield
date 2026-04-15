import logging
import math

from crypto_app.models import LiveSignal
from .normalizer import normalize_ticker
from .market_data import (
    get_crypto_data,
    get_stock_data,
    get_crypto_history,
    get_stock_history
)
from .live_state import append_state, get_state
from .feature_builder import build_live_sequence_from_buffer
from crypto_app.ml.predict import predict_pump

logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    try:
        val = float(val)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except Exception:
        return default


def seed_live_buffer_if_needed(symbol, market_type):
    """
    Seeds rolling buffer from REAL recent history if buffer is too short.
    No fake data. Only real market history.
    """
    existing = get_state(symbol)
    if existing and len(existing) >= 6:
        return existing

    try:
        if market_type == "crypto":
            hist = get_crypto_history(symbol)
        else:
            hist = get_stock_history(symbol)

        if hist is None or hist.empty:
            return existing or []

        hist = hist.tail(10)

        seeded = []
        for _, row in hist.iterrows():
            price = safe_float(row.get("price"), 0.0)
            volume = safe_float(row.get("volume"), 0.0)

            if price > 0:
                seeded.append({
                    "price": price,
                    "volume": volume
                })

        for point in seeded:
            append_state(symbol, point, maxlen=20)

        return get_state(symbol) or []

    except Exception as e:
        logger.warning(f"Buffer seed failed for {symbol}: {e}")
        return existing or []


def generate_live_signal(raw_ticker):
    """
    Generates one live prediction using rolling live buffer.
    Uses REAL live + REAL recent historical market points.
    """
    try:
        norm = normalize_ticker(raw_ticker)
        symbol = norm["symbol"]
        market_type = norm["type"]

        # ===============================
        # SEED BUFFER FIRST (REAL DATA ONLY)
        # ===============================
        seed_live_buffer_if_needed(symbol, market_type)

        # ===============================
        # FETCH CURRENT MARKET POINT
        # ===============================
        if market_type == "crypto":
            live = get_crypto_data(symbol)
        else:
            live = get_stock_data(symbol)

        if not live:
            return None

        price = safe_float(live.get("price"), 0.0)
        volume = safe_float(live.get("volume"), 0.0)

        if price <= 0:
            return None

        point = {
            "price": price,
            "volume": volume
        }

        # ===============================
        # APPEND TO ROLLING BUFFER
        # ===============================
        buffer = append_state(symbol, point, maxlen=20)

        if len(buffer) < 6:
            return None

        # ===============================
        # BUILD LIVE FEATURES
        # ===============================
        rf_features, sequence = build_live_sequence_from_buffer(
            buffer,
            market_type=market_type
        )

        prediction_input = {
            **rf_features,
            "sequence": sequence
        }

        result = predict_pump(prediction_input)

        risk_score = safe_float(result.get("risk_score"), 0.0)
        anomaly_score = safe_float(result.get("anomaly_score"), 0.0)
        confidence = safe_float(result.get("confidence"), 0.0)

        # ===============================
        # DUPLICATE PROTECTION
        # ===============================
        latest = LiveSignal.objects.filter(
            symbol__iexact=symbol
        ).order_by("-created_at").first()

        if latest:
            same_risk = abs(float(latest.risk_score) - risk_score) < 0.0005
            same_anomaly = abs(float(latest.anomaly_score) - anomaly_score) < 0.0005

            if same_risk and same_anomaly:
                return latest

        # ===============================
        # SAVE NEW LIVE SIGNAL
        # ===============================
        signal = LiveSignal.objects.create(
            symbol=symbol,
            market_type=market_type,
            risk_score=risk_score,
            anomaly_score=anomaly_score,
            confidence=confidence
        )

        return signal

    except Exception as e:
        logger.exception(f"Live signal generation failed: {e}")
        return None