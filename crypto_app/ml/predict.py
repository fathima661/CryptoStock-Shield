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
# SAFE VALUE
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
# LOAD MODEL (CACHED)
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
        logger.warning("⚠️ Feature list missing → fallback used")
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
# 🎯 TIME WINDOW
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
# 🎯 CONFIDENCE
# ==========================================
def compute_confidence(model_prob, anomaly_score):
    model_prob = clean_value(model_prob, 0.5)
    anomaly_score = clean_value(anomaly_score, 0.0)

    base_conf = abs(model_prob - 0.5) * 2
    agreement = 1 - abs(model_prob - anomaly_score)

    confidence = (0.6 * base_conf) + (0.4 * agreement)
    confidence = max(0.0, min(1.0, confidence))
    return confidence


# ==========================================
# 🧠 EXPLANATION ENGINE
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
        reasons.append("Strong anomaly spike detected in recent trading behavior")
    elif anomaly_score > 0.6:
        reasons.append("Moderate anomaly detected in market structure")
    elif anomaly_score > 0.4:
        reasons.append("Graph anomaly detection flagged irregular behavior")

    if not reasons:
        reasons.append("No strong abnormal signals detected")

    if model_prob > 0.75:
        summary = "High probability of coordinated market manipulation"
    elif model_prob > 0.5:
        summary = "Moderate suspicious activity detected"
    else:
        summary = "Market behavior appears normal"

    return {
        "summary": summary,
        "reasons": reasons,
        "feature_importance": importance
    }


# ==========================================
# 🚀 MAIN FUNCTION
# ==========================================
def predict_pump(input_data: dict):
    start_time = time.time()
    logger.info("🚀 Prediction started")

    try:
        # =========================
        # MODEL SELECTION
        # =========================
        model_obj = choose_model()
        if not model_obj:
            logger.warning("No active ML model → using fallback mode")
            model_obj = None

        # =========================
        # LOAD MODEL
        # =========================
        model, feature_list = load_model()

        values = []
        feature_dict = {}

        for f in feature_list:
            val = clean_value(input_data.get(f, 0))
            values.append(val)
            feature_dict[f] = val

        X = pd.DataFrame([values], columns=feature_list)

        # =========================
        # MODEL PREDICTION
        # =========================
        try:
            model_prob = float(model.predict_proba(X)[0][1])
            model_prob = clean_value(model_prob, 0.5)
            model_prob = max(0.05, min(0.95, model_prob))
        except Exception:
            logger.exception("Model prediction failed")
            model_prob = 0.5

        # =========================
        # ANOMALY DETECTION (GNN)
        # =========================
        try:
            sequence = input_data.get("sequence", [])

            if compute_anomaly_score is None:
                anomaly_score = 0.0  # GNN disabled safely
            elif not isinstance(sequence, list) or len(sequence) == 0:
                anomaly_score = 0.0
            else:
                anomaly_score = compute_anomaly_score(sequence)

        except Exception:
            logger.exception("Anomaly failed")
            anomaly_score = 0.0

        anomaly_score = clean_value(anomaly_score, 0.0)
        feature_dict["anomaly_score"] = anomaly_score

        # =========================
        # FINAL FUSION
        # =========================
        risk_score = (0.7 * model_prob) + (0.3 * anomaly_score)

        risk_score = clean_value(risk_score, 0.0)
        risk_score = max(0.0, min(1.0, risk_score))

        confidence = compute_confidence(model_prob, anomaly_score)
        confidence = clean_value(confidence, 0.0)

        # =========================
        # OUTPUT LOGIC
        # =========================
        is_manipulated = risk_score >= 0.7
        time_window = calculate_time_window(risk_score)

        explanation = generate_explanation(
            feature_dict,
            model_prob,
            anomaly_score
        )

        # =========================
        # MODEL TRACKING
        # =========================
        if model_obj:
            try:
                model_obj.total_predictions += 1
                model_obj.save(update_fields=["total_predictions"])
            except Exception:
                logger.warning("Model tracking failed")

        latency = round(time.time() - start_time, 4)

        logger.info(f"✅ Prediction done | Risk={risk_score:.3f} | Latency={latency}s")

        return {
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "is_manipulated": is_manipulated,
            "predicted_time_window": time_window,
            "model_used": str(model_obj.id) if model_obj else "fallback_model",
            "latency": latency,
            "explanation": explanation,
            "model_probability": round(model_prob, 4)
        }

    except Exception:
        logger.exception("CRITICAL FAILURE in prediction")
        raise