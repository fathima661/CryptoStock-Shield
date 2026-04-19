from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        User = get_user_model()

        email = "admin@gmail.com"
        password = "admin@2001"

        if not User.objects.filter(email=email).exists():
            User.objects.create_superuser(
                email=email,
                password=password
            )
            self.stdout.write("Admin created")
        else:
            self.stdout.write("Admin already exists")