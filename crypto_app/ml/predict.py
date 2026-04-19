import joblib
import numpy as np
import os
import logging
import time
import pandas as pd
from functools import lru_cache
from services.model_manager import choose_model

try:
    from services.gnn_anomaly import compute_anomaly_score
except ImportError:
    compute_anomaly_score = None


logger = logging.getLogger(__name__)

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "pump_detector_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "ml_models", "model_features.pkl")


# ==========================================
# SAFE VALUE (stable version)
# ==========================================
def clean_value(val, default=0.0):
    try:
        if val is None:
            return default
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return default
        return val
    except Exception:
        return default


# ==========================================
# LOAD MODEL
# ==========================================
@lru_cache()
def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.critical("🚨 MODEL FILE MISSING")
        raise FileNotFoundError("Model not found")

    model = joblib.load(MODEL_PATH)

    if os.path.exists(FEATURE_PATH):
        features = joblib.load(FEATURE_PATH)
    else:
        features = [
            "price",
            "volume",
            "volatility_7d",
            "daily_return",
            "price_ma7",
            "price_ma30",
            "post_count"
        ]

    return model, features


# ==========================================
# TIME WINDOW
# ==========================================
def calculate_time_window(prob):
    if prob >= 0.85:
        return "1-3 hours"
    elif prob >= 0.7:
        return "2-6 hours"
    elif prob >= 0.5:
        return "6-12 hours"
    return "No immediate risk"


# ==========================================
# CONFIDENCE (stable + bounded)
# ==========================================
def compute_confidence(model_prob, anomaly_score):
    model_prob = clean_value(model_prob, 0.5)
    anomaly_score = clean_value(anomaly_score, 0.0)

    base_conf = abs(model_prob - 0.5) * 2
    agreement = 1 - abs(model_prob - anomaly_score)

    confidence = (0.6 * base_conf) + (0.4 * agreement)
    return max(0.0, min(1.0, confidence))


# ==========================================
# EXPLANATION (unchanged logic)
# ==========================================
def generate_explanation(features, model_prob, anomaly_score):
    reasons = []
    importance = {}

    if features.get("volume", 0) > 1e7:
        reasons.append("Unusual spike in trading volume")
        importance["volume"] = features["volume"]

    if abs(features.get("daily_return", 0)) > 0.05:
        reasons.append("Significant price movement detected")
        importance["daily_return"] = features["daily_return"]

    if features.get("volatility_7d", 0) > 0.05:
        reasons.append("High short-term volatility")
        importance["volatility_7d"] = features["volatility_7d"]

    if features.get("post_count", 0) > 20:
        reasons.append("Elevated social activity")
        importance["post_count"] = features["post_count"]

    if anomaly_score > 0.75:
        reasons.append("Strong anomaly spike detected")
    elif anomaly_score > 0.6:
        reasons.append("Moderate anomaly detected")
    elif anomaly_score > 0.4:
        reasons.append("Irregular behavior flagged")

    if not reasons:
        reasons.append("No strong abnormal signals detected")

    if model_prob > 0.75:
        summary = "High probability of manipulation"
    elif model_prob > 0.5:
        summary = "Moderate suspicious activity"
    else:
        summary = "Market behavior appears normal"

    return {
        "summary": summary,
        "reasons": reasons,
        "feature_importance": importance
    }


# ==========================================
# MAIN PREDICT FUNCTION (STABLE CORE)
# ==========================================
def predict_pump(input_data: dict):
    start_time = time.time()
    logger.info("🚀 Prediction started")

    try:
        model_obj = choose_model()
        model, feature_list = load_model()

        values = []
        feature_dict = {}

        missing_features = []

        # =========================
        # FEATURE HANDLING (STABLE FIX)
        # =========================
        for f in feature_list:
            val = clean_value(input_data.get(f, 0.0))

            if val == 0.0 and f not in input_data:
                missing_features.append(f)

            values.append(val)
            feature_dict[f] = val

        X = pd.DataFrame([values], columns=feature_list)

        # =========================
        # MODEL PREDICTION
        # =========================
        try:
            model_prob = model.predict_proba(X)[0][1]
            model_prob = clean_value(model_prob, 0.5)
            model_prob = np.clip(model_prob, 0.05, 0.95)
        except Exception:
            logger.exception("Model prediction failed")
            model_prob = 0.5

        # =========================
        # ANOMALY DETECTION (NORMALIZED FIX)
        # =========================
        sequence = input_data.get("sequence", [])

        try:
            if compute_anomaly_score and isinstance(sequence, list) and sequence:
                anomaly_score = compute_anomaly_score(sequence)
            else:
                anomaly_score = 0.0
        except Exception:
            anomaly_score = 0.0

        anomaly_score = clean_value(anomaly_score, 0.0)
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)  # CRITICAL FIX

        feature_dict["anomaly_score"] = anomaly_score

        # =========================
        # FUSION (STABLE WEIGHTED MODEL)
        # =========================
        risk_score = (0.75 * model_prob) + (0.25 * anomaly_score)
        risk_score = np.clip(risk_score, 0.0, 1.0)

        confidence = compute_confidence(model_prob, anomaly_score)

        # =========================
        # OUTPUT
        # =========================
        is_manipulated = risk_score >= 0.7
        time_window = calculate_time_window(risk_score)

        explanation = generate_explanation(
            feature_dict,
            model_prob,
            anomaly_score
        )

        # =========================
        # TRACKING SAFE
        # =========================
        if model_obj:
            try:
                model_obj.total_predictions += 1
                model_obj.save(update_fields=["total_predictions"])
            except Exception:
                pass

        latency = round(time.time() - start_time, 4)

        logger.info(f"✅ Prediction done | Risk={risk_score:.3f}")

        return {
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_manipulated": is_manipulated,
            "predicted_time_window": time_window,
            "model_used": str(model_obj.id) if model_obj else "fallback_model",
            "latency": latency,
            "explanation": explanation,
            "model_probability": round(model_prob, 4),
            "missing_features": missing_features
        }

    except Exception:
        logger.exception("CRITICAL FAILURE in prediction")
        raise