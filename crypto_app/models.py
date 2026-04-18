from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import BaseUserManager


# ==========================================
# USER
# ==========================================
class CustomUserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError("Email is required")

        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault("is_staff", True)
        extra_fields.setdefault("is_superuser", True)

        if extra_fields.get("is_staff") is not True:
            raise ValueError("Superuser must have is_staff=True")

        if extra_fields.get("is_superuser") is not True:
            raise ValueError("Superuser must have is_superuser=True")

        return self.create_user(email, password, **extra_fields)
    

class User(AbstractUser):
    username = None
    email = models.EmailField(unique=True, db_index=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    objects = CustomUserManager()

    def __str__(self):
        return self.email


# ==========================================
# PROFILE
# ==========================================
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")

    full_name = models.CharField(max_length=150, blank=True)
    bio = models.TextField(blank=True)
    profile_picture = models.ImageField(upload_to="profiles/", null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)


# ==========================================
# ALERT SETTINGS
# ==========================================
class AlertSetting(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="alerts")

    email_alerts = models.BooleanField(default=True)
    sms_alerts = models.BooleanField(default=False)

    risk_threshold = models.FloatField(
        default=80.0,
        validators=[MinValueValidator(0), MaxValueValidator(100)]
    )


# ==========================================
# ASSET
# ==========================================
class Asset(models.Model):
    MARKET_TYPE = (
        ('stock', 'Stock'),
        ('crypto', 'Crypto'),
    )

    name = models.CharField(max_length=100)
    symbol = models.CharField(max_length=20, db_index=True)
    market_type = models.CharField(max_length=10, choices=MARKET_TYPE)

    exchange = models.CharField(max_length=50, blank=True)
    is_active = models.BooleanField(default=True) 

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("symbol", "market_type", "exchange")

    def __str__(self):
        return f"{self.symbol} ({self.market_type})"


# ==========================================
# SCAN
# ==========================================
class Scan(models.Model):
    STATUS = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="scans",null=True,blank=True)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)

    csv_file = models.FileField(upload_to="uploads/", null=True, blank=True)
    ticker_input = models.CharField(max_length=50, null=True, blank=True)
    social_link = models.TextField(null=True, blank=True)

    status = models.CharField(max_length=20, choices=STATUS, default='pending', db_index=True)

    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True, db_index=True)


# ==========================================
# TRADE DATA
# ==========================================
class TradeData(models.Model):
    scan = models.ForeignKey(Scan, on_delete=models.CASCADE, related_name="trades")

    timestamp = models.DateTimeField(db_index=True)
    price = models.FloatField(validators=[MinValueValidator(0)])
    volume = models.FloatField(validators=[MinValueValidator(0)])


# ==========================================
# SOCIAL SIGNAL
# ==========================================
class SocialSignal(models.Model):
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)

    platform = models.CharField(max_length=20)
    content = models.TextField()

    signal_time = models.DateTimeField(db_index=True)


# ==========================================
# ML MODEL
# ==========================================
class MLModel(models.Model):
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)

    model_file = models.FileField(upload_to="ml_models/")
    created_at = models.DateTimeField(auto_now_add=True)

     # 🔥 NEW (IMPORTANT)
    is_active = models.BooleanField(default=True)
    traffic_percentage = models.FloatField(default=1.0)  # 0–1

    # 📊 OPTIONAL METRICS
    accuracy = models.FloatField(null=True, blank=True)
    total_predictions = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.name} v{self.version}"


# ==========================================
# PREDICTION
class Prediction(models.Model):
    scan = models.OneToOneField(Scan, on_delete=models.CASCADE, related_name="prediction")

    # ✅ MODEL VERSION TRACKING
    model = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True, blank=True)
    model_used = models.CharField(max_length=100, default="Hybrid GNN-RF")
    explanation_tags = models.JSONField(default=list, blank=True)

    # ✅ CORE OUTPUTS
    risk_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    confidence = models.FloatField()
    is_manipulated = models.BooleanField()
    predicted_time_window = models.CharField(max_length=50)
    anomaly_score = models.FloatField(default=0.0)

    # ✅ 🔥 NEW — INPUT SNAPSHOT (CRITICAL)
    input_snapshot = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)


# ==========================================
# CLUSTER
# ==========================================
class Cluster(models.Model):
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name="clusters")

    cluster_id = models.CharField(max_length=50)
    suspicious_score = models.FloatField()

    description = models.TextField(blank=True)


# ==========================================
# EXPLANATION
# ==========================================
class Explanation(models.Model):
    prediction = models.OneToOneField(Prediction, on_delete=models.CASCADE)

    feature_importance = models.JSONField()
    summary = models.TextField()


# ==========================================
# ALERT
# ==========================================
class Alert(models.Model):
    RISK_CHOICES = (
        ("LOW", "Low"),
        ("MEDIUM", "Medium"),
        ("HIGH", "High"),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)

    message = models.TextField()
    risk_level = models.CharField(max_length=10, choices=RISK_CHOICES)

    is_read = models.BooleanField(default=False, db_index=True)

    # 🔥 NEW (PRODUCTION)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    dedup_key = models.CharField(max_length=255, null=True, blank=True, db_index=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "-created_at"]),
        ]

# ==========================================
# SAVED SCANS
# ==========================================
class SavedScan(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE)

    saved_at = models.DateTimeField(auto_now_add=True)


class LiveSignal(models.Model):
    symbol = models.CharField(max_length=20)
    market_type = models.CharField(max_length=10)

    risk_score = models.FloatField()
    anomaly_score = models.FloatField()
    confidence = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["symbol", "created_at"])
        ]    

class Watchlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, db_index=True)
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "asset")

    def __str__(self):
        return f"{self.user} - {self.asset}"    


# ==========================================
# BACKTEST RUN
# ==========================================
class BacktestRun(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="backtests")
    asset = models.ForeignKey(Asset, on_delete=models.CASCADE)
    model_version = models.ForeignKey(MLModel, on_delete=models.SET_NULL, null=True, blank=True)
    
    start_date = models.DateField()
    end_date = models.DateField()
    
    # Stores: { "labels": [], "prices": [], "risk_scores": [] }
    simulation_results = models.JSONField(default=dict)
    
    accuracy_score = models.FloatField(default=0.0)
    total_anomalies_detected = models.IntegerField(default=0)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.asset.symbol} Analysis ({self.start_date})"