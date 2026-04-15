# services/alert_engine.py

import hashlib
from django.utils.timezone import now
from datetime import timedelta
from crypto_app.models import Alert, AlertSetting

def generate_alert(scan, prediction):
    user = scan.user
    if not user:
        return

    try:
        settings = user.alerts
    except AlertSetting.DoesNotExist:
        return

    risk_percent = prediction.risk_score * 100

    # 🚫 BELOW THRESHOLD → NO ALERT
    if risk_percent < settings.risk_threshold:
        return

    # ==========================================
    # 🎯 RISK LEVEL CLASSIFICATION
    # ==========================================
    if risk_percent >= 80:
        level = "HIGH"
    elif risk_percent >= 50:
        level = "MEDIUM"
    else:
        level = "LOW"

    # ==========================================
    # 🔁 DEDUP KEY (CRITICAL)
    # ==========================================
    raw_key = f"{user.id}-{scan.asset.symbol}-{level}"
    dedup_key = hashlib.md5(raw_key.encode()).hexdigest()

    # Prevent duplicate alerts in last 10 mins
    recent = Alert.objects.filter(
        user=user,
        dedup_key=dedup_key,
        created_at__gte=now() - timedelta(minutes=10)
    ).exists()

    if recent:
        return

    # ==========================================
    # 🧠 MESSAGE GENERATION
    # ==========================================
    message = (
        f"{scan.asset.symbol}: {level} manipulation risk "
        f"({round(risk_percent,1)}%)."
    )

    # ==========================================
    # 💾 CREATE ALERT
    # ==========================================
    Alert.objects.create(
        user=user,
        prediction=prediction,
        message=message,
        risk_level=level,
        dedup_key=dedup_key
    )