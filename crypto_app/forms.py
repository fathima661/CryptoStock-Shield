from django import forms
from django.contrib.auth import authenticate
from django.contrib.auth.forms import UserCreationForm
from .models import User, UserProfile, AlertSetting, Scan,Asset
from django.core.exceptions import ValidationError
import re



# ==========================================
# REGISTER
# ==========================================
class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ['email', 'password1', 'password2']

    def clean_email(self):
        email = self.cleaned_data.get('email').lower()

        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already registered")

        return email
    # ==========================
    # EXTRA PASSWORD RULES
    # ==========================
    def clean_password1(self):
        password = self.cleaned_data.get("password1")

        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters")

        if not re.search(r"[A-Z]", password):
            raise ValidationError("Must include at least one uppercase letter")

        if not re.search(r"[a-z]", password):
            raise ValidationError("Must include at least one lowercase letter")

        if not re.search(r"[0-9]", password):
            raise ValidationError("Must include at least one number")

        return password


# ==========================================
# LOGIN
# ==========================================
class LoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        data = super().clean()

        user = authenticate(username=data.get("email"), password=data.get("password"))

        if not user:
            raise forms.ValidationError("Invalid email or password")

        data['user'] = user
        return data


# ==========================================
# PROFILE
# ==========================================
class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ['full_name', 'bio', 'profile_picture']


# ==========================================
# ALERT SETTINGS
# ==========================================
class AlertSettingForm(forms.ModelForm):
    class Meta:
        model = AlertSetting
        fields = ['email_alerts', 'sms_alerts', 'risk_threshold']

    def clean_risk_threshold(self):
        value = self.cleaned_data["risk_threshold"]

        if value < 0 or value > 100:
            raise forms.ValidationError("Must be between 0 and 100")

        return value


# ==========================================
# SCAN FORM
# ==========================================
# forms.py (only ScanForm section)



TICKER_CHOICES = [
    ("", "Select Asset"),

    # Crypto
    ("BTC", "Bitcoin (BTC)"),
    ("ETH", "Ethereum (ETH)"),
    ("BNB", "Binance Coin (BNB)"),
    ("XRP", "Ripple (XRP)"),
    ("ADA", "Cardano (ADA)"),
    ("DOGE", "Dogecoin (DOGE)"),
    ("SOL", "Solana (SOL)"),
    ("MATIC", "Polygon (MATIC)"),
    ("DOT", "Polkadot (DOT)"),
    ("LTC", "Litecoin (LTC)"),

    # Stocks
    ("RELIANCE.NS", "Reliance"),
    ("TCS.NS", "TCS"),
    ("INFY.NS", "Infosys"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("ICICIBANK.NS", "ICICI Bank"),
]


class ScanForm(forms.ModelForm):
    # Hidden field to track which tab (Ticker/CSV/Social) is active
    input_type = forms.CharField(required=True, widget=forms.HiddenInput())

    # ChoiceField ensures only your predefined assets can be selected
    ticker_input = forms.ChoiceField(
        choices=TICKER_CHOICES,
        required=False
    )

    csv_file = forms.FileField(required=False)
    
    # Users can paste links OR social signal text
    social_link = forms.CharField(
        required=False, 
        widget=forms.Textarea(attrs={
            'placeholder': 'Paste Telegram/Reddit/X signal link or message',
            'rows': 4
        })
    )

    class Meta:
        model = Scan
        fields = ['ticker_input', 'csv_file', 'social_link', 'input_type']

    def clean(self):
        cleaned_data = super().clean()
        input_type = cleaned_data.get("input_type")

        # ==========================================
        # TICKER (DO NOT TOUCH)
        # ==========================================
        if input_type == "ticker":
            if not cleaned_data.get("ticker_input"):
                self.add_error('ticker_input', "Please select a valid asset from the list.")

        # ==========================================
        # CSV
        # ==========================================
        elif input_type == "csv":
            csv_file = cleaned_data.get("csv_file")

            if not csv_file:
                self.add_error('csv_file', "Please upload a CSV file to proceed.")
            else:
                if not csv_file.name.lower().endswith(".csv"):
                    self.add_error('csv_file', "Only CSV files are allowed.")

        # ==========================================
        # SOCIAL
        # ==========================================
        elif input_type == "social":
            social_link = cleaned_data.get("social_link")

            if not social_link or not social_link.strip():
                self.add_error('social_link', "Please provide a social link or paste a signal message.")

        # ==========================================
        # INVALID TYPE
        # ==========================================
        else:
            raise forms.ValidationError("Invalid input type selected.")

        return cleaned_data
    

# ==========================================
# BACKTEST FORM (FIXED - MATCHES SCAN)
# ==========================================

from .forms import TICKER_CHOICES  # if same file, just use directly

class BacktestForm(forms.Form):
    asset = forms.ChoiceField(
        choices=TICKER_CHOICES,   # ✅ SAME AS SCAN
        required=True,
        widget=forms.Select(attrs={
            'class': 'w-full bg-[#0B1220] border border-gray-700 rounded-lg p-3 mt-1 text-sm text-white'
        })
    )

    start_date = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'w-full bg-[#0B1220] border border-gray-700 rounded-lg p-3 text-xs text-white'
        })
    )

    end_date = forms.DateField(
        required=True,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': 'w-full bg-[#0B1220] border border-gray-700 rounded-lg p-3 text-xs text-white'
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        start = cleaned_data.get("start_date")
        end = cleaned_data.get("end_date")

        if start and end and start >= end:
            raise forms.ValidationError("End date must be after start date.")

        return cleaned_data