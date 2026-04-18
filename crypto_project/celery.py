import os
import sys
from celery import Celery

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(BASE_DIR)   # ✅ REQUIRED for your services/

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "crypto_project.settings")

app = Celery("crypto_project")

app.config_from_object("django.conf:settings", namespace="CELERY")

app.autodiscover_tasks()