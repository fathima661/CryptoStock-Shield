from django.core.management.base import BaseCommand
from crypto_app.models import Asset

class Command(BaseCommand):
    help = "Seed initial assets"

    def handle(self, *args, **kwargs):

        assets = [
            # Crypto
            {"name": "Bitcoin", "symbol": "BTCUSDT", "market_type": "crypto"},
            {"name": "Ethereum", "symbol": "ETHUSDT", "market_type": "crypto"},

            # Stocks (India)
            {"name": "Reliance Industries", "symbol": "RELIANCE", "market_type": "stock"},
            {"name": "TCS", "symbol": "TCS", "market_type": "stock"},
        ]

        for a in assets:
            Asset.objects.get_or_create(**a)

        self.stdout.write(self.style.SUCCESS("Assets seeded successfully"))