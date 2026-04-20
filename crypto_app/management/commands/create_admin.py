from django.core.management.base import BaseCommand
from crypto_app.models import User

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        email = "admin@gmail.com"
        password = "Admin@2001"

        if User.objects.filter(email=email).exists():
            self.stdout.write("Admin already exists")
            return

        user = User.objects.create_user(email=email, password=password)
        user.is_staff = True
        user.is_superuser = True
        user.save()

        self.stdout.write("Admin created successfully")