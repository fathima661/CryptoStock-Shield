from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('scan/', views.scan, name='scan'),
    path('backtest/', views.backtest_view, name='backtest'),
    path("backtest/pdf/<int:run_id>/", views.download_backtest_pdf, name="backtest_pdf"),
    path('about/', views.about, name='about'),
    
    path("api/login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("api/register/", views.register_view, name="register"),

    path('user_dashboard/', views.user_dashboard, name='user_dashboard'),
    path("results/<int:scan_id>/", views.results, name="results"),
    path('download/<int:scan_id>/', views.download_report, name='download_report'),
    path("history/", views.history, name="history"),
    path("history/live/", views.history_live_data, name="history_live"),
    path("api/alerts/", views.alerts_api, name="alerts_api"),
    path("api/alerts/count/", views.alerts_count, name="alerts_count"),
    path("api/alerts/read/<int:alert_id>/", views.mark_alert_read, name="mark_alert_read"),
    path("live-preview/", views.live_preview, name="live_preview"),
    path("delete-account/", views.delete_account, name="delete_account"),


    path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("admin-users/", views.admin_users, name="admin_users"),
    path("admin-scans/", views.admin_scans, name="admin_scans"),
    path("delete-user/<int:user_id>/", views.delete_user, name="delete_user"),
    path('login/', views.home, {'auth_mode': 'login'}),
    path('register/', views.home, {'auth_mode': 'register'}),
    
]