# ==========================================
# STANDARD LIBRARY IMPORTS
# ==========================================
import json
import logging
import time
from datetime import timedelta

# ==========================================
# THIRD-PARTY IMPORTS
# ==========================================
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64

# ==========================================
# DJANGO IMPORTS
# ==========================================
from django.contrib import messages
from django.contrib.auth import (login, logout, authenticate, get_user_model)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.tokens import default_token_generator
from django.core.cache import cache
from django.core.mail import send_mail
from django.core.paginator import Paginator
from django.db.models import Count, F, Q
from django.db.models.functions import TruncDate
from django.http import (FileResponse, JsonResponse, HttpResponse)
from django.shortcuts import (get_object_or_404, redirect, render)
from django.urls import reverse
from django.utils import timezone
from django.utils.encoding import force_bytes
from django.utils.http import (urlsafe_base64_encode, urlsafe_base64_decode)
from django.utils.timezone import localtime
from django.views.decorators.csrf import csrf_exempt
from django.template.loader import render_to_string
from django.core.paginator import Paginator
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image




 

# ==========================================
# LOCAL APP IMPORTS
# ==========================================
from .forms import (LoginForm, RegisterForm, ScanForm, BacktestForm)
from .models import (Alert, AlertSetting, Prediction, Scan,UserProfile, SavedScan, Asset, LiveSignal,BacktestRun, MLModel, Watchlist)
from .ml.predict import predict_pump
from .utils.pdf_report import generate_pdf
from .models import Explanation
from .models import User

# ==========================================
# SERVICES IMPORTS
# ==========================================
from services.asset_resolver import resolve_asset
from services.feature_builder import build_features_from_history
from services.market_data import (get_crypto_data, get_crypto_history,get_stock_data, get_stock_history)
from services.model_manager import choose_model
from services.live_state import get_state
from services.live_signal_service import generate_live_signal
from services.normalizer import normalize_ticker


# ==========================================
# LOGGER
# ==========================================
logger = logging.getLogger(__name__)

# ==========================================
# BASIC PAGES
# ==========================================
def home(request,auth_mode=None):
    return render(request,"home.html",{"auth_mode": auth_mode})


def about(request):
    return render(request, "about.html")


# ==========================================
# PROFILE
# ==========================================
@login_required
def profile(request):
    user = request.user

    total_scans = Scan.objects.filter(user=user).count()
    completed_scans = Scan.objects.filter(user=user, status="completed").count()
    pending_scans = Scan.objects.filter(user=user, status="processing").count()

    alerts = Alert.objects.filter(user=user).order_by("-created_at")[:5]
    recent_scans = Scan.objects.filter(user=user).order_by("-created_at")[:5]
    saved = SavedScan.objects.filter(user=user).select_related("prediction")[:5]

    context = {
        "total_scans": total_scans,
        "completed_scans": completed_scans,
        "pending_scans": pending_scans,
        "alerts": alerts,
        "recent_scans": recent_scans,
        "saved": saved,
    }

    return render(request, "profile.html", context)


# ==========================================
# SAFE VALUE CLEANER
# ==========================================
def clean_value(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except Exception:
        return default


# ==========================================
# CSV FEATURE EXTRACTION
# ==========================================
def build_features_from_csv(file):
    file.seek(0)  # important for uploaded file re-read
    df = pd.read_csv(file)

    if df.empty:
        raise ValueError("CSV is empty")

    df.columns = [c.lower().strip() for c in df.columns]

    if "price" not in df.columns or "volume" not in df.columns:
        raise ValueError("CSV must contain 'price' and 'volume' columns")

    price = df["price"].astype(float)
    volume = df["volume"].astype(float)

    price_change = (price.iloc[-1] - price.iloc[0]) / max(price.iloc[0], 1)
    volatility = price.pct_change().std()

    ma7 = price.rolling(7).mean().iloc[-1]
    ma30 = price.rolling(30).mean().iloc[-1]

    return {
        "price": clean_value(price.iloc[-1]),
        "volume": clean_value(volume.iloc[-1]),
        "volatility_7d": clean_value(volatility, 0.5),
        "daily_return": clean_value(price_change),
        "price_ma7": clean_value(ma7, price.iloc[-1]),
        "price_ma30": clean_value(ma30, price.iloc[-1]),
        "post_count": 5,

        # Optional legacy/fallback extras
        "volatility": clean_value(volatility, 0.5),
        "volume_spike": clean_value(volume.pct_change().abs().mean(), 0.5),
        "momentum": abs(clean_value(price_change)),
        "price_change": clean_value(price_change),
        "social_score": 0.3,
    }


# ==========================================
# MARKET FALLBACK FEATURE BUILDER
# ==========================================
def build_features_from_market(data, scan_obj=None):
    """
    Fallback only: generates features from a single data point
    when full historical data is unavailable.
    """
    price = clean_value(data.get("price"), 0)
    volume = clean_value(data.get("volume"), 0)
    volatility = clean_value(data.get("volatility"), 0.5)
    price_change = clean_value(data.get("price_change"), 0)

    ma7 = price * 0.98
    ma30 = price * 0.95

    return {
        "price": price,
        "volume": volume,
        "volatility_7d": volatility,
        "daily_return": price_change,
        "price_ma7": ma7,
        "price_ma30": ma30,
        "post_count": 0,
    }


# ==========================================
# UNIFIED FEATURE BUILDER
# ==========================================
def build_features(scan, market_data):
    if scan.csv_file:
        return build_features_from_csv(scan.csv_file)

    if market_data:
        return build_features_from_market(market_data, scan)

    return {
        "price": 0,
        "volume": 0,
        "volatility_7d": 0.5,
        "daily_return": 0,
        "price_ma7": 0,
        "price_ma30": 0,
        "post_count": 0,
        "volatility": 0.5,
        "volume_spike": 0.5,
        "momentum": 0.5,
        "price_change": 0,
        "social_score": 0.3,
    }


# ==========================================
# SCAN CONFIG
# ==========================================
MAX_ACTIVE_SCANS = 3
SCAN_TIMEOUT_SECONDS = 60


# ==========================================
# SCAN VIEW
# ==========================================
def scan(request):
    free_scans = request.session.get("free_scans", 0)
    form = ScanForm()

    if request.method == "POST":
        form = ScanForm(request.POST, request.FILES)

        if form.is_valid():
            if not request.user.is_authenticated and free_scans >= 1:
                messages.warning(request, "Please login to continue scanning.")
                form = ScanForm(request.POST, request.FILES)
                return render(request, "scan.html",{
                    "form": form,
                    "show_login_popup": True,
                    "old_input": request.POST
                })

            scan_obj = None

            try:
                input_type = form.cleaned_data.get("input_type")
                ticker = form.cleaned_data.get("ticker_input")
                csv_file = form.cleaned_data.get("csv_file")
                social_data = form.cleaned_data.get("social_link")

                selected_market = request.POST.get("market_type", "").strip().lower()
                if selected_market not in ["stock", "crypto"]:
                    messages.error(request, "Please select a valid market.")
                    return redirect("scan")

                resolved_asset = None
                history = None
                rf_features = {}
                sequence = []

                display_label = ""
                input_summary = ""
                scan_source = input_type

                # ==========================================
                # INPUT HANDLING
                # ==========================================
                if input_type == "ticker":
                    resolved_asset, history = resolve_asset(ticker)

                    if not resolved_asset:
                        messages.error(request, f"Could not resolve asset: {ticker}")
                        return redirect("scan")

                    rf_features, sequence = build_features_from_history(
                        history,
                        symbol=resolved_asset.symbol,
                        market_type=resolved_asset.market_type,
                    )

                    display_label = resolved_asset.symbol
                    input_summary = f"{resolved_asset.name or resolved_asset.symbol} ({resolved_asset.market_type})"

                elif input_type == "csv":
                    if not csv_file:
                        messages.error(request, "Please upload CSV.")
                        return redirect("scan")

                    rf_features = build_features_from_csv(csv_file)
                    sequence = []

                    resolved_asset, _ = Asset.objects.get_or_create(
                        symbol=f"CUSTOM_CSV_{selected_market.upper()}",
                        market_type=selected_market,
                        exchange="SYSTEM",
                        defaults={
                            "name": f"Custom CSV ({selected_market})",
                            "is_active": True,
                        }
                    )

                    display_label = "CSV Upload"
                    input_summary = csv_file.name

                elif input_type == "social":
                    social_text = social_data.strip() if social_data else ""

                    if not social_text:
                        messages.error(request, "No social input.")
                        return redirect("scan")

                    word_count = len(social_text.split())
                    uppercase_ratio = sum(1 for c in social_text if c.isupper()) / max(len(social_text), 1)
                    exclamations = social_text.count("!")
                    has_pump_words = any(w in social_text.lower() for w in ["pump","moon","100x","buy"])

                    rf_features = {
                        "price": 0,
                        "volume": 0,
                        "volatility_7d": 0.5,
                        "daily_return": 0,
                        "price_ma7": 0,
                        "price_ma30": 0,
                        "post_count": max(1, word_count // 10),
                        "social_score": min(1.0, 0.4 + (0.2 if has_pump_words else 0) + (uppercase_ratio * 0.4)),
                        "volatility": 0.5,
                        "volume_spike": min(1.0, exclamations / 10),
                        "momentum": 0.3 if has_pump_words else 0.0,
                        "price_change": 0.0,
                    }

                    sequence = []

                    resolved_asset, _ = Asset.objects.get_or_create(
                        symbol=f"SOCIAL_SIGNAL_{selected_market.upper()}",
                        market_type=selected_market,
                        exchange="SYSTEM",
                        defaults={
                            "name": "Social Signal",
                            "is_active": True,
                        }
                    )

                    display_label = "Social Signal"
                    input_summary = social_text[:100]

                else:
                    messages.error(request, "Invalid input type.")
                    return redirect("scan")

                # ==========================================
                # CREATE SCAN
                # ==========================================
                scan_obj = form.save(commit=False)
                scan_obj.user = request.user if request.user.is_authenticated else None
                scan_obj.asset = resolved_asset
                scan_obj.status = "processing"
                scan_obj.save()

                # ==========================================
                # RUN PREDICTION
                # ==========================================
                prediction_input = {**rf_features, "sequence": sequence}
                result = predict_pump(prediction_input)

                if result is None:
                    raise ValueError("Prediction returned None")

                explanation_data = generate_explanation(result, rf_features)

                # ==========================================
                # SAVE PREDICTION
                # ==========================================
                prediction_obj = Prediction.objects.create(
                    scan=scan_obj,
                    risk_score=result.get("risk_score", 0.5),
                    confidence=result.get("confidence", 0.0),
                    anomaly_score=result.get("anomaly_score", 0.0),
                    predicted_time_window=result.get("predicted_time_window", "Unknown"),
                    is_manipulated=result.get("is_manipulated", False),
                    input_snapshot=prediction_input,
                )

                # ==========================================
                # 🧠 CREATE EXPLANATION (CORRECT PLACE)
                # ==========================================
                Explanation.objects.create(
                    prediction=prediction_obj,
                    summary=explanation_data.get("summary", ""),
                    feature_importance=explanation_data.get("feature_importance", {})
                )

                # ==========================================
                # ALERT SYSTEM
                # ==========================================
                try:
                    if request.user.is_authenticated:

                        risk_score = result.get("risk_score", 0)

                        if risk_score >= 0.7:
                            risk_level = "HIGH"
                            message = f"🚨 High risk detected for {display_label}"

                        elif risk_score >= 0.4:
                            risk_level = "MEDIUM"
                            message = f"⚠️ Medium risk for {display_label}"

                        else:
                            risk_level = None

                        if risk_level:
                            recent_exists = Alert.objects.filter(
                                user=request.user,
                                message__icontains=display_label,
                                created_at__gte=timezone.now() - timedelta(minutes=10)
                            ).exists()

                            if not recent_exists:
                                Alert.objects.create(
                                    user=request.user,
                                    prediction=prediction_obj,
                                    message=message,
                                    risk_level=risk_level
                                )

                except Exception:
                    logger.exception("Alert creation failed")

                # ==========================================
                # SUCCESS
                # ==========================================
                scan_obj.status = "completed"
                scan_obj.save()

                if not request.user.is_authenticated:
                    request.session["free_scans"] = free_scans + 1
                    request.session["guest_scan_id"] = scan_obj.id

                request.session[f"scan_meta_{scan_obj.id}"] = {
                    "scan_source": scan_source,
                    "display_label": display_label,
                    "input_summary": input_summary,
                    "selected_market": selected_market,
                }

                return redirect("results", scan_id=scan_obj.id)

            except Exception as e:
                logger.exception("Scan crashed")

                if scan_obj:
                    scan_obj.status = "failed"
                    scan_obj.error_message = f"Unexpected: {str(e)}"
                    scan_obj.save()

                messages.error(request, "Internal error occurred.")

    return render(request, "scan.html", {"form": form})


def generate_explanation(prediction, features):
    reasons = []

    if features.get("volume_spike", 0) > 0.5:
        reasons.append("Unusual spike in trading volume detected")

    if features.get("volatility_7d", 0) > 0.5:
        reasons.append("High short-term volatility observed")

    if features.get("daily_return", 0) > 0.1:
        reasons.append("Sharp price increase within short time window")

    if prediction["risk_score"] > 0.7:
        summary = "High likelihood of coordinated market manipulation."
    elif prediction["risk_score"] > 0.4:
        summary = "Moderate irregular activity detected."
    else:
        summary = "Market behavior appears organic."

    return {
        "summary": summary,
        "feature_importance": {
            k: round(v, 4)
            for k, v in features.items()
            if isinstance(v, (int, float))
        },
        "reasons": reasons
    }

# ==========================================
# RESULTS VIEW
# ==========================================
def results(request, scan_id):
    scan = get_object_or_404(Scan, id=scan_id)

    if scan.status != "completed":
        return render(request, "processing.html")

    prediction = scan.prediction

    risk_percent = round(prediction.risk_score * 100, 1)
    confidence_percent = round(prediction.confidence * 100, 1)

    # ==========================================
    # DEFAULT DECISION
    # ==========================================
    decision = {
        "title": "Market Neutral",
        "sentence": "No significant manipulation patterns detected.",
        "color": "blue",
        "bg": "bg-blue-500/10",
        "border": "border-blue-500",
        "pulse": "",
    }

    if risk_percent >= 75:
        decision = {
            "title": "Critical Manipulation Warning",
            "sentence": "High probability of coordinated artificial price movement. Avoid entry.",
            "color": "red",
            "bg": "bg-red-500/10",
            "border": "border-red-500",
            "pulse": "animate-pulse",
        }
    elif risk_percent >= 45:
        decision = {
            "title": "Elevated Risk Detected",
            "sentence": "Unusual volatility detected. Exercise extreme caution with stop-losses.",
            "color": "yellow",
            "bg": "bg-yellow-500/10",
            "border": "border-yellow-500",
            "pulse": "",
        }
    elif risk_percent < 25:
        decision = {
            "title": "Organic Market Behavior",
            "sentence": "Price action appears healthy and driven by organic volume.",
            "color": "green",
            "bg": "bg-green-500/10",
            "border": "border-green-500",
            "pulse": "",
        }

    # ==========================================
    # REASONS
    # ==========================================
    reasons = []
    if prediction.anomaly_score > 0.6:
        reasons.append("Abnormal volume spike relative to 7D average")
    if risk_percent > 50:
        reasons.append("Price-momentum divergence detected")
    if prediction.confidence < 0.35:
        reasons.append("Model confidence is moderate due to limited supporting structure")
    if not reasons:
        reasons.append("Consistent liquidity and standard volatility")

    # ==========================================
    # EXTRA DISPLAY DATA FOR CSV / SOCIAL / TICKER
    # ==========================================
    session_meta = request.session.get(f"scan_meta_{scan.id}", {})

    scan_source = session_meta.get("scan_source", getattr(scan, "input_type", "ticker"))
    display_label = session_meta.get("display_label", scan.asset.symbol if scan.asset else "Unknown Asset")
    input_summary = session_meta.get("input_summary", scan.asset.name if scan.asset else "No summary")
    selected_market = session_meta.get("selected_market", scan.asset.market_type if scan.asset else "crypto")

    # pretty badge label
    source_badge = {
        "ticker": "Ticker Scan",
        "csv": "CSV Upload Scan",
        "social": "Social Signal Scan"
    }.get(scan_source, "AI Scan")

    # more human-friendly scan subtitle
    if scan_source == "ticker":
        subtitle = f"Live market asset analysis for {display_label}"
    elif scan_source == "csv":
        subtitle = f"Uploaded dataset analysis for {selected_market.title()} market"
    elif scan_source == "social":
        subtitle = f"Sentiment and hype-risk analysis for {selected_market.title()} market"
    else:
        subtitle = "AI market manipulation analysis"

    return render(
        request,
        "results.html",
        {
            "scan": scan,
            "prediction": prediction,
            "risk_percent": risk_percent,
            "conf_percent": confidence_percent,
            "decision": decision,
            "reasons": reasons,

            # extra display fields
            "scan_source": scan_source,
            "display_label": display_label,
            "input_summary": input_summary,
            "selected_market": selected_market,
            "source_badge": source_badge,
            "subtitle": subtitle,
        },
    )


# ==========================================
# AUTH
# ==========================================
def login_view(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # ✅ GET NEXT URL
        next_url = request.POST.get("next") or "/"

        if form.is_valid():
            email = form.cleaned_data.get("email")
            password = form.cleaned_data.get("password")

            # 🔥 HARD-CODED ADMIN LOGIN (bypass normal auth)
            if email == "admin@gmail.com" and password == "Admin@2001":
                from django.contrib.auth import get_user_model, login
                User = get_user_model()

                user = User.objects.filter(email=email).first()

                if not user:
                    user = User.objects.create_superuser(
                        email=email,
                        password=password
                    )

                login(request, user)

                if is_ajax:
                    return JsonResponse({
                        "success": True,
                        "redirect": "/admin-dashboard/"
                    })

                return redirect("/admin-dashboard/")

            # normal authentication flow
            user = authenticate(request, username=email, password=password)

            if user is not None:
                login(request, user)

                if is_ajax:
                    return JsonResponse({
                        "success": True,
                        "redirect": next_url
                    })

                return redirect(next_url)

            if is_ajax:
                return JsonResponse({
                    "success": False,
                    "error": {"__all__": ["Invalid email or password"]}
                }, status=400)

            return redirect("/")

        if is_ajax:
            return JsonResponse({
                "success": False,
                "error": form.errors.get_json_data()
            }, status=400)

        return redirect("/")

    return JsonResponse({"error": "Invalid request"}, status=400)



def logout_view(request):
    logout(request)
    return redirect("home")

def register_view(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

        # ✅ GET NEXT URL
        next_url = request.POST.get("next") or "/"

        if form.is_valid():
            user = form.save()
            login(request, user)

            if is_ajax:
                return JsonResponse({
                    "success": True,
                    "redirect": next_url
                })

            return redirect(next_url)

        if is_ajax:
            return JsonResponse({
                "success": False,
                "error": form.errors.get_json_data()
            }, status=400)

        return redirect("/")

    return JsonResponse({"error": "Invalid request"}, status=400)


@login_required
def delete_account(request):
    if request.method == "POST":
        user = request.user

        # related deletions
        Scan.objects.filter(user=user).delete()
        Alert.objects.filter(user=user).delete()
        SavedScan.objects.filter(user=user).delete()

        # FIXED RELATION NAME
        if hasattr(user, "profile"):
            user.profile.delete()

        # delete user
        user.delete()

        logout(request)

        return JsonResponse({
            "success": True,
            "redirect": "/"
        })

    return JsonResponse({
        "success": False,
        "error": "Invalid request method"
    }, status=400)



# ==========================================
# DOWNLOAD REPORT
# ==========================================
def download_report(request, scan_id):
    scan = get_object_or_404(Scan, id=scan_id)

    if scan.user:
        if request.user != scan.user:
            return redirect("/login/")
    else:
        if request.session.get("guest_scan_id") != scan.id:
            return redirect("/scan/")

    prediction = scan.prediction

    # ✅ FIXED CALL
    file_path = generate_pdf(prediction)

    return FileResponse(open(file_path, "rb"), as_attachment=True)

# ==========================================
# HISTORY PAGE
# ==========================================
@login_required(login_url='/')
def history(request):
    from django.db.models import F, Q, Count
    from django.db.models.functions import TruncDate
    from django.core.cache import cache
    import json

    user = request.user

    # ==========================================
    # CACHE KEY (SMART)
    # ==========================================
    cache_key = f"history_page_{user.id}_{request.GET.urlencode()}"
    cached = cache.get(cache_key)
    if cached:
        return render(request, "history.html", cached)

    # ==========================================
    # BASE QUERY (OPTIMIZED)
    # ==========================================
    scans = (
        Scan.objects
        .filter(user=user)
        .select_related("asset", "prediction")
        .only(
            "id", "status", "created_at",
            "asset__symbol", "asset__market_type",
            "prediction__risk_score"
        )
    )

    # ==========================================
    # GET PARAMS
    # ==========================================
    status = request.GET.get("status") or ""
    asset = request.GET.get("asset") or ""
    risk = request.GET.get("risk") or ""
    sort = request.GET.get("sort") or "newest"

    # ==========================================
    # FILTERS
    # ==========================================

    if status:
        scans = scans.filter(status__iexact=status)

    # 🔥 ASSET FILTER
    if asset:
        if asset == "__CRYPTO__":
            scans = scans.filter(asset__market_type="crypto")

        elif asset == "__STOCK__":
            scans = scans.filter(asset__market_type="stock")

        elif asset.startswith("CSV"):
            scans = scans.filter(asset__symbol__startswith="CUSTOM_CSV")

        elif asset.startswith("SOCIAL"):
            scans = scans.filter(asset__symbol__startswith="SOCIAL_SIGNAL")

        else:
            scans = scans.filter(asset__symbol__iexact=asset)

    # 🔥 RISK FILTER (ONLY COMPLETED — CORRECT)
    if risk:
        scans = scans.filter(
            status="completed",
            prediction__isnull=False
        )

        risk_filters = {
            "low": Q(prediction__risk_score__lt=0.4),
            "medium": Q(prediction__risk_score__gte=0.4, prediction__risk_score__lt=0.7),
            "high": Q(prediction__risk_score__gte=0.7),
        }

        scans = scans.filter(risk_filters.get(risk, Q()))

    # ==========================================
    # SORTING (SAFE)
    # ==========================================
    SORT_MAP = {
        "newest": "-created_at",
        "oldest": "created_at",
        "risk_high": F("prediction__risk_score").desc(nulls_last=True),
        "risk_low": F("prediction__risk_score").asc(nulls_last=True),
    }

    scans = scans.order_by(SORT_MAP.get(sort, "-created_at"))

    # ==========================================
    # PAGINATION
    # ==========================================
    paginator = Paginator(scans, 10)
    page_obj = paginator.get_page(request.GET.get("page"))

    # ==========================================
    # 📊 ANALYTICS (FIXED + CONSISTENT)
    # ==========================================
    analytics_qs = (
        Scan.objects
        .filter(user=user)
        .select_related("prediction")
        .only("created_at", "status", "prediction__risk_score")
        .order_by("-created_at")[:100]
    )

    analytics = list(analytics_qs)[::-1]

    risk_dates = []
    risk_values = []

    for s in analytics:
        if s.status == "completed" and hasattr(s, "prediction") and s.prediction:
            risk_dates.append(s.created_at.strftime("%Y-%m-%d"))
            risk_values.append(round(s.prediction.risk_score * 100, 2))

    # ==========================================
    # 📊 FREQUENCY (UNCHANGED — includes all scans)
    # ==========================================
    freq_qs = (
        Scan.objects
        .filter(user=user)
        .annotate(date=TruncDate("created_at"))
        .values("date")
        .annotate(count=Count("id"))
        .order_by("-date")[:100]
    )

    freq = list(freq_qs)[::-1]

    freq_dates = [str(x["date"]) for x in freq]
    freq_values = [x["count"] for x in freq]

    # ==========================================
    # FINAL CONTEXT
    # ==========================================
    context = {
        "page_obj": page_obj,
        "status": status,
        "asset": asset,
        "risk": risk,
        "sort": sort,
        "risk_dates": json.dumps(risk_dates),
        "risk_values": json.dumps(risk_values),
        "freq_dates": json.dumps(freq_dates),
        "freq_values": json.dumps(freq_values),
    }

    # 🔥 CACHE PAGE
    cache.set(cache_key, context, 10)

    return render(request, "history.html", context)

# ==========================================
# OPTIONAL ASYNC PDF HELPER
# ==========================================
def generate_pdf_async(prediction, file_path=None):
    generate_pdf(prediction)


# ==========================================
# HISTORY LIVE DATA API
# ==========================================
@login_required
def history_live_data(request):
    from django.core.cache import cache
    from django.utils.timezone import localtime

    cache_key = f"history_live_{request.user.id}"
    data = cache.get(cache_key)

    if data:
        return JsonResponse(data)

    # ✅ ORDER FIRST → SLICE → THEN REVERSE (correct flow)
    scans = list(
        Scan.objects
        .filter(user=request.user, status="completed")
        .select_related("prediction")
        .only("created_at", "prediction__risk_score")
        .order_by("-created_at")[:50]
    )

    # ✅ chronological order for chart
    scans.reverse()

    risk_dates = []
    risk_values = []

    for s in scans:
        if hasattr(s, "prediction") and s.prediction:
            risk_dates.append(localtime(s.created_at).strftime("%H:%M"))
            risk_values.append(round(s.prediction.risk_score * 100, 2))

    data = {
        "risk_dates": risk_dates,
        "risk_values": risk_values,
    }

    # ✅ cache to avoid DB hammer
    cache.set(cache_key, data, 5)

    return JsonResponse(data)


# ==========================================
# ALERTS API
# ==========================================
@login_required
def alerts_api(request):
    alerts = Alert.objects.filter(user=request.user)\
        .order_by("-created_at")[:10]

    return JsonResponse({
        "alerts": [
            {
                "id": a.id,
                "message": a.message,
                "risk": a.risk_level,
                "is_read": a.is_read,
                "time": a.created_at.strftime("%H:%M:%S"),
            }
            for a in alerts
        ]
    })

@login_required
def alerts_count(request):
    count = Alert.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({"count": count})


@login_required
def mark_alert_read(request, alert_id):
    alert = get_object_or_404(
        Alert,
        id=alert_id,
        user=request.user
    )

    if not alert.is_read:
        alert.is_read = True
        alert.save()

    return JsonResponse({"success": True})


# ==========================================
# LIVE PREVIEW API
# ==========================================
@csrf_exempt
def live_preview(request):
    """
    SAFE live preview endpoint:
    - ticker preview remains untouched
    - csv preview added separately
    - social preview added separately
    """

    input_type = request.GET.get("input_type") or request.POST.get("input_type", "ticker")

    # =========================================================
    # 1) TICKER PREVIEW (KEEP YOUR EXISTING WORKING LOGIC)
    # =========================================================
    if input_type == "ticker":
        raw_ticker = request.GET.get("ticker")

        if not raw_ticker:
            return JsonResponse({"error": "No ticker"}, status=400)

        try:
            norm = normalize_ticker(raw_ticker)
            ticker = norm["symbol"]
        except Exception as e:
            logger.warning(f"Ticker normalization failed for {raw_ticker}: {e}")
            return JsonResponse({"error": "Invalid ticker"}, status=400)

        try:
            generate_live_signal(raw_ticker)
        except Exception as e:
            logger.warning(f"Live signal generation failed for {raw_ticker}: {e}")

        signals = list(
            LiveSignal.objects
            .filter(symbol__iexact=ticker)
            .order_by("-created_at")[:20]
        )

        if not signals:
            return JsonResponse({"error": "No live data yet"}, status=404)

        latest = signals[0]

        if time.time() - latest.created_at.timestamp() > 120:
            return JsonResponse({"error": "Live data stale"}, status=408)

        return JsonResponse({
            "risk": float(latest.risk_score),
            "anomaly": float(latest.anomaly_score),
            "confidence": float(latest.confidence),
            "history": [float(s.risk_score) for s in reversed(signals)],
            "timestamp": latest.created_at.timestamp(),
            "warming_up": len(signals) < 3,
            "source": "ticker"
        })

    # =========================================================
    # 2) CSV PREVIEW (FIXED GRAPH SUPPORT)
    # =========================================================
    elif input_type == "csv":
        csv_file = request.FILES.get("csv_file")

        if not csv_file:
            return JsonResponse({"error": "No CSV uploaded"}, status=400)

        try:
            import pandas as pd

            # Read CSV safely
            df = pd.read_csv(csv_file)

            if "price" not in df.columns or "volume" not in df.columns:
                return JsonResponse({"error": "CSV must contain 'price' and 'volume' columns"}, status=400)

            # Build features using your existing function
            csv_file.seek(0)
            rf_features = build_features_from_csv(csv_file)

            result = predict_pump({
                **rf_features,
                "sequence": []
            })

            # Build chart history from uploaded price data
            prices = df["price"].dropna().astype(float).tolist()

            if len(prices) >= 2:
                min_p = min(prices)
                max_p = max(prices)

                if max_p == min_p:
                    history = [float(result.get("risk_score", 0.0))] * min(len(prices), 20)
                else:
                    normalized = [(p - min_p) / (max_p - min_p) for p in prices[-20:]]
                    base_risk = float(result.get("risk_score", 0.0))

                    # Convert price movement into risk-like preview trend
                    history = [
                        max(0.0, min(1.0, (0.5 * n) + (0.5 * base_risk)))
                        for n in normalized
                    ]
            else:
                # fallback if very few rows
                base_risk = float(result.get("risk_score", 0.0))
                history = [
                    max(0.0, base_risk - 0.10),
                    max(0.0, base_risk - 0.06),
                    max(0.0, base_risk - 0.03),
                    max(0.0, base_risk - 0.01),
                    base_risk
                ]

            return JsonResponse({
                "risk": float(result.get("risk_score", 0.0)),
                "anomaly": float(result.get("anomaly_score", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "history": history,
                "timestamp": time.time(),
                "source": "csv"
            })

        except Exception:
            logger.exception("CSV live preview failed")
            return JsonResponse({"error": "CSV preview failed"}, status=500)

    # =========================================================
    # 3) SOCIAL PREVIEW (FIXED GRAPH SUPPORT)
    # =========================================================
    elif input_type == "social":
        social_text = request.POST.get("social_link", "").strip()

        if not social_text:
            return JsonResponse({"error": "No social text provided"}, status=400)

        try:
            word_count = len(social_text.split())
            uppercase_ratio = sum(1 for c in social_text if c.isupper()) / max(len(social_text), 1)
            exclamations = social_text.count("!")
            has_pump_words = any(
                word in social_text.lower()
                for word in ["pump", "moon", "100x", "buy now", "signal", "rocket", "bullish"]
            )

            rf_features = {
                "price": 0,
                "volume": 0,
                "volatility_7d": 0.5,
                "daily_return": 0,
                "price_ma7": 0,
                "price_ma30": 0,
                "post_count": max(1, word_count // 10),
                "social_score": min(1.0, 0.4 + (0.2 if has_pump_words else 0) + (uppercase_ratio * 0.4)),
                "volatility": 0.5,
                "volume_spike": min(1.0, exclamations / 10),
                "momentum": 0.3 if has_pump_words else 0.0,
                "price_change": 0.0,
            }

            result = predict_pump({
                **rf_features,
                "sequence": []
            })

            base_risk = float(result.get("risk_score", 0.0))

            # Build social trend graph based on signal intensity
            intensity = min(1.0, (
                (0.25 if has_pump_words else 0.0) +
                min(0.25, exclamations * 0.03) +
                min(0.25, uppercase_ratio * 0.8) +
                min(0.25, word_count / 100)
            ))

            history = [
                max(0.0, min(1.0, base_risk - 0.12 + intensity * 0.20)),
                max(0.0, min(1.0, base_risk - 0.08 + intensity * 0.18)),
                max(0.0, min(1.0, base_risk - 0.05 + intensity * 0.15)),
                max(0.0, min(1.0, base_risk - 0.02 + intensity * 0.12)),
                max(0.0, min(1.0, base_risk))
            ]

            return JsonResponse({
                "risk": float(result.get("risk_score", 0.0)),
                "anomaly": float(result.get("anomaly_score", 0.0)),
                "confidence": float(result.get("confidence", 0.0)),
                "history": history,
                "timestamp": time.time(),
                "source": "social"
            })

        except Exception:
            logger.exception("Social live preview failed")
            return JsonResponse({"error": "Social preview failed"}, status=500)

    return JsonResponse({"error": "Invalid preview type"}, status=400)



@login_required
def add_to_watchlist(request, asset_id):
    asset = get_object_or_404(Asset, id=asset_id)

    Watchlist.objects.get_or_create(
        user=request.user,
        asset=asset
    )

    return JsonResponse({"success": True})

@login_required
def remove_from_watchlist(request, asset_id):
    Watchlist.objects.filter(
        user=request.user,
        asset_id=asset_id
    ).delete()

    return JsonResponse({"success": True})


# ==========================================
# BACKTEST VIEW (FINAL - MATCHES NORMALIZER)
# ==========================================
def backtest_view(request):
    # ==========================================
    # GUEST FREE BACKTEST LIMIT
    # ==========================================
    free_backtests = request.session.get("free_backtests", 0)

    history = []
    if request.user.is_authenticated:
        history = BacktestRun.objects.filter(
            user=request.user
        ).order_by("-created_at")[:5]

    result = None
    chart_data = None
    metrics = None

    if request.method == "POST":

        # Require login after first guest backtest
        if not request.user.is_authenticated and free_backtests >= 1:
            messages.warning(request, "Please login to continue using backtest.")
            
            form = BacktestForm(request.POST)

            return render(request, "backtest.html", {
                "form": form,
                "history": history,
                "result": None,
                "chart_data": None,
                "metrics": None,
                "show_login_popup": True,
            })

        form = BacktestForm(request.POST)

        if form.is_valid():
            try:
                ticker = form.cleaned_data["asset"]
                start_date = form.cleaned_data["start_date"]
                end_date = form.cleaned_data["end_date"]

                # ===============================
                # SYMBOL HANDLING
                # ===============================
                if ticker.endswith(".NS"):
                    symbol = ticker
                    market_type = "stock"
                else:
                    symbol = f"{ticker}-USD"
                    market_type = "crypto"

                df = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                if df is None or df.empty or "Close" not in df.columns:
                    raise ValueError(f"No market data found for {symbol}")

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # ===============================
                # CLEAN DATA
                # ===============================
                df = df.copy()
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df[df["Close"] > 0]

                # ===============================
                # FEATURE ENGINEERING
                # ===============================
                df["daily_return"] = df["Close"].pct_change()
                df["volatility_7d"] = df["daily_return"].rolling(7).std()
                df["price_ma7"] = df["Close"].rolling(7).mean()
                df["price_ma30"] = df["Close"].rolling(30).mean()

                df = df.dropna()

                if len(df) < 20:
                    raise ValueError("Not enough historical data after cleaning")

                labels, prices, risk_scores = [], [], []
                anomalies = 0
                sequence_buffer = []

                # ===============================
                # PREDICTION LOOP
                # ===============================
                for date, row in df.iterrows():
                    try:
                        price = float(row["Close"])
                        volume = float(row.get("Volume", 0) or 0)

                        if price <= 0:
                            continue

                        features = [
                            price,
                            volume,
                            float(row["daily_return"]),
                            float(row["volatility_7d"]),
                            float(row["price_ma7"]),
                            float(row["price_ma30"]),
                        ]

                        if any(np.isnan(f) or np.isinf(f) for f in features):
                            continue

                        sequence_buffer.append(features)

                        if len(sequence_buffer) > 10:
                            sequence_buffer.pop(0)

                        pred = predict_pump({
                            "price": features[0],
                            "volume": features[1],
                            "daily_return": features[2],
                            "volatility_7d": features[3],
                            "price_ma7": features[4],
                            "price_ma30": features[5],
                            "post_count": 0,
                            "sequence": sequence_buffer.copy()
                        })

                        if not pred:
                            continue

                        score = float(pred.get("risk_score", 0)) * 100

                        labels.append(date.strftime("%Y-%m-%d"))
                        prices.append(price)
                        risk_scores.append(score)

                        if pred.get("is_manipulated"):
                            anomalies += 1

                    except Exception:
                        continue

                if not prices:
                    raise ValueError("Prediction failed")

                # ===============================
                # METRICS
                # ===============================
                prices_np = np.array(prices)

                if len(prices_np) > 1:
                    returns = np.diff(prices_np) / np.maximum(prices_np[:-1], 1e-9)
                else:
                    returns = np.array([])

                total_return = (
                    ((prices[-1] - prices[0]) / max(prices[0], 1e-9)) * 100
                    if prices else 0
                )

                volatility = (
                    np.std(returns) * np.sqrt(252) * 100
                    if len(returns) > 0 else 0
                )

                sharpe = (
                    (np.mean(returns) / np.std(returns)) * np.sqrt(252)
                    if len(returns) > 0 and np.std(returns) > 1e-9 else 0
                )

                peak = prices[0]
                max_dd = 0

                for p in prices:
                    if p > peak:
                        peak = p
                    dd = (peak - p) / max(peak, 1e-9)
                    max_dd = max(max_dd, dd)

                risk_binary = [1 if r > 70 else 0 for r in risk_scores]

                if len(returns) > 0:
                    price_move = [1 if r > 0 else 0 for r in returns]
                    price_move.append(0)

                    min_len = min(len(risk_binary), len(price_move))

                    correct = sum(
                        1 for i in range(min_len)
                        if risk_binary[i] == price_move[i]
                    )

                    accuracy = (correct / max(min_len, 1)) * 100
                else:
                    accuracy = 0

                metrics = {
                    "total_return": round(total_return, 2),
                    "volatility": round(volatility, 2),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd * 100, 2),
                    "signal_accuracy": round(accuracy, 2)
                }

                chart_data = {
                    "labels": labels,
                    "prices": [round(p, 2) for p in prices],
                    "risk_scores": [round(r, 2) for r in risk_scores]
                }

                # ===============================
                # SAVE ONLY FOR LOGGED-IN USERS
                # ===============================
                if request.user.is_authenticated:
                    asset, _ = Asset.objects.get_or_create(
                        symbol=ticker,
                        market_type=market_type,
                        defaults={
                            "name": ticker,
                            "is_active": True
                        }
                    )

                    model = MLModel.objects.filter(is_active=True).first()

                    result = BacktestRun.objects.create(
                        user=request.user,
                        asset=asset,
                        model_version=model,
                        start_date=start_date,
                        end_date=end_date,
                        simulation_results=chart_data,
                        accuracy_score=metrics["signal_accuracy"],
                        total_anomalies_detected=anomalies
                    )

                    history = BacktestRun.objects.filter(
                        user=request.user
                    ).order_by("-created_at")[:5]

                else:
                    # Guest mock result object for template display
                    class GuestResult:
                        pass

                    result = GuestResult()
                    result.asset = type("obj", (), {"symbol": ticker})
                    result.accuracy_score = metrics["signal_accuracy"]
                    result.total_anomalies_detected = anomalies

                    request.session["free_backtests"] = free_backtests + 1

                messages.success(request, f"Backtest completed for {ticker}")

                form = BacktestForm(initial={
                    "asset": ticker,
                    "start_date": start_date,
                    "end_date": end_date
                })

            except Exception as e:
                print("BACKTEST ERROR:", str(e))
                messages.error(request, str(e))

    else:
        form = BacktestForm()

    return render(request, "backtest.html", {
        "form": form,
        "history": history,
        "result": result,
        "chart_data": chart_data,
        "metrics": metrics
    })

def download_backtest_pdf(request, run_id):
    run = BacktestRun.objects.get(id=run_id, user=request.user)

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="backtest_{run.asset.symbol}.pdf"'

    doc = SimpleDocTemplate(response)
    styles = getSampleStyleSheet()
    elements = []

    # =========================
    # TITLE
    # =========================
    elements.append(Paragraph("Backtest Report (AI Strategy Analysis)", styles['Title']))
    elements.append(Spacer(1, 12))

    # =========================
    # BASIC INFO
    # =========================
    data = [
        ["Asset", run.asset.symbol],
        ["Start Date", str(run.start_date)],
        ["End Date", str(run.end_date)],
        ["Accuracy", f"{run.accuracy_score:.2f}%"],
        ["Anomalies", str(run.total_anomalies_detected)],
    ]

    table = Table(data)
    table.setStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ])

    elements.append(table)
    elements.append(Spacer(1, 20))

    # =========================
    # EXTRACT SIMULATION DATA
    # =========================
    sim = run.simulation_results or {}

    prices = sim.get("prices", [])
    risk_scores = sim.get("risk_scores", [])
    labels = sim.get("labels", [])

    entry_price = None
    trades = 0
    wins = 0

    pnl_curve = []

    equity = 1000  # starting capital
    equity_curve = [equity]

    position = 0  # 0 = no position, 1 = holding

    # =========================
    # STRATEGY SIMULATION
    # =========================
    for i in range(len(prices)):
        price = prices[i]
        risk = risk_scores[i]

        # BUY
        if risk < 30 and position == 0:
            entry_price = price
            position = 1

        # SELL
        elif risk > 70 and position == 1:
            pnl = price - entry_price
            pnl_pct = pnl / max(entry_price, 1e-9)

            equity *= (1 + pnl_pct)

            pnl_curve.append(pnl)

            trades += 1
            if pnl > 0:
                wins += 1

            position = 0

        equity_curve.append(equity)

    win_rate = (wins / trades * 100) if trades > 0 else 0
    total_pnl = sum(pnl_curve) if pnl_curve else 0

    # =========================
    # PERFORMANCE TABLE
    # =========================
    elements.append(Paragraph("Strategy Performance", styles['Heading2']))
    elements.append(Spacer(1, 8))

    perf_data = [
        ["Total Trades", str(trades)],
        ["Winning Trades", str(wins)],
        ["Win Rate", f"{win_rate:.2f}%"],
        ["Total PnL", f"{total_pnl:.2f}"],
    ]

    perf_table = Table(perf_data)
    perf_table.setStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
    ])

    elements.append(perf_table)
    elements.append(Spacer(1, 20))

    # =========================
    # PRICE + RISK CHART
    # =========================
    plt.figure(figsize=(8, 4))
    plt.plot(prices, label="Price", color="blue")
    plt.plot(risk_scores, label="Risk", color="red")
    plt.title("Backtest Simulation (Price vs Risk)")
    plt.legend()

    buffer1 = io.BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    plt.close()

    elements.append(Image(buffer1, width=400, height=200))
    elements.append(Spacer(1, 10))

    # =========================
    # EQUITY GROWTH CURVE (NEW)
    # =========================
    plt.figure(figsize=(8, 4))
    plt.plot(equity_curve, label="Equity Curve", color="green")
    plt.title("Equity Growth Curve")
    plt.legend()

    buffer2 = io.BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    plt.close()

    elements.append(Image(buffer2, width=400, height=200))
    elements.append(Spacer(1, 10))

    # =========================
    # FINAL SUMMARY
    # =========================
    summary = f"""
    Simulation Summary:
    - Total Trades: {trades}
    - Win Rate: {win_rate:.2f}%
    - Total PnL: {total_pnl:.2f}
    - Final Equity: {equity:.2f}
    - Data Points: {len(prices)}
    """

    elements.append(Paragraph("Final Summary", styles['Heading2']))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(summary.replace("\n", "<br/>"), styles['Normal']))

    # =========================
    # BUILD PDF
    # =========================
    doc.build(elements)
    return response
#-------------------------------------------------------ADMIN---------------------------------------------------

from functools import wraps
from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import get_user_model
from datetime import date
from .models import Scan
from django.views.decorators.http import require_POST
from django.core.cache import cache
from django.utils import timezone


User = get_user_model()

def admin_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated or not request.user.is_superuser:
            return redirect("/")
        return view_func(request, *args, **kwargs)
    return wrapper



@login_required
@admin_required
def admin_dashboard(request):

    cached = cache.get("admin_dashboard_data")

    if cached:
        return render(request, "admin/dashboard.html", cached)

    context = {
        "total_users": User.objects.count(),
        "total_scans": Scan.objects.count(),
        "today_scans": Scan.objects.filter(created_at__date=date.today()).count(),
        "active_users": User.objects.filter(is_active=True).count(),

        "recent_scans": Scan.objects.select_related("user", "asset")
                                    .order_by("-created_at")[:20],
    }

    cache.set("admin_dashboard_data", context, 60)  # 60 sec cache

    return render(request, "admin/dashboard.html", context)


@login_required
@admin_required
def admin_users(request):
    users = User.objects.all().order_by("-date_joined")
    return render(request, "admin/users.html", {"users": users})


@login_required
@admin_required
def admin_scans(request):
    scans = Scan.objects.select_related("user", "asset").order_by("-created_at")[:50]
    return render(request, "admin/scans.html", {"scans": scans})


@login_required
@admin_required
@require_POST
def delete_user(request, user_id):
    user = get_object_or_404(User, id=user_id)

    # Prevent deleting superuser
    if user.is_superuser:
        return redirect("admin_users")

    user.delete()
    return redirect("admin_users")

 