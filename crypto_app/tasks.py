# crypto_app/tasks.py

from celery import shared_task
from django.utils import timezone
from datetime import timedelta

from .models import Watchlist, Alert, Prediction
from services.asset_resolver import resolve_asset
from services.feature_builder import build_features_from_history
from crypto_app.ml.predict import predict_pump

import logging
logger = logging.getLogger(__name__)


@shared_task(bind=True, autoretry_for=(Exception,), retry_backoff=5, retry_kwargs={"max_retries": 3})
def run_watchlist_scan(self):
    logger.info("🚀 Watchlist scan started")

    watchlist_items = Watchlist.objects.select_related("user", "asset").all()
    logger.info(f"📊 Total watchlist items: {watchlist_items.count()}")

    if not watchlist_items.exists():
        logger.warning("⚠️ No watchlist items found")
        return "No items"

    for item in watchlist_items:
        try:
            logger.info(f"🔍 Scanning: {item.asset.symbol} (User {item.user.id})")

            asset = item.asset
            user = item.user

            resolved_asset, history = resolve_asset(asset.symbol)

            if not resolved_asset or history is None:
                logger.warning(f"❌ Failed to resolve asset: {asset.symbol}")
                continue

            rf_features, sequence = build_features_from_history(
                history,
                symbol=resolved_asset.symbol,
                market_type=resolved_asset.market_type,
            )

            if not rf_features:
                logger.warning(f"❌ Feature build failed: {asset.symbol}")
                continue

            prediction_input = {**rf_features, "sequence": sequence}
            result = predict_pump(prediction_input)

            if not result:
                logger.warning(f"❌ Prediction failed: {asset.symbol}")
                continue

            risk_score = result.get("risk_score", 0)
            logger.info(f"📈 Risk score for {asset.symbol}: {risk_score}")

            # ==============================
            # ALERT LOGIC
            # ==============================
            if risk_score >= 0.7:
                risk_level = "HIGH"
                message = f"🚨 High risk detected for {asset.symbol}"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
                message = f"⚠️ Medium risk for {asset.symbol}"
            else:
                logger.info(f"✅ Safe: {asset.symbol}")
                continue

            # ==============================
            # ANTI-SPAM
            # ==============================
            dedup_key = f"{user.id}:{asset.symbol}:{risk_level}"

            recent_exists = Alert.objects.filter(
                user=user,
                dedup_key=dedup_key,
                created_at__gte=timezone.now() - timedelta(minutes=15)
            ).exists()

            if recent_exists:
                logger.info(f"⏭️ Skipped duplicate alert: {asset.symbol}")
                continue

            # ==============================
            # SAVE PREDICTION
            # ==============================
            prediction_obj = Prediction.objects.create(
                scan=None,
                risk_score=risk_score,
                confidence=result.get("confidence", 0),
                anomaly_score=result.get("anomaly_score", 0),
                predicted_time_window=result.get("predicted_time_window", "Unknown"),
                is_manipulated=result.get("is_manipulated", False),
                input_snapshot=prediction_input
            )

            # ==============================
            # CREATE ALERT
            # ==============================
            Alert.objects.create(
                user=user,
                prediction=prediction_obj,
                message=message,
                risk_level=risk_level,
                dedup_key=dedup_key
            )

            logger.info(f"🚨 Alert created for {asset.symbol}")

        except Exception as e:
            logger.exception(f"🔥 Scan failed for item {item.id}: {str(e)}")

    logger.info("✅ Watchlist scan completed")
    return "Done"