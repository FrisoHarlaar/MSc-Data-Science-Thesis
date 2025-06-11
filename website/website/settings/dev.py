from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-aghx3bbkpk1*s+cmxx&m$rhyoy15gd#teb6-5fq+4rv)tumqx%"

# SECURITY WARNING: define the correct hosts in production!
# ALLOWED_HOSTS = ["*"]

EMAIL_BACKEND = "django.core.mail.backends.console.EmailBackend"

CSRF_TRUSTED_ORIGINS = [
    'https://*.ngrok.io',
    'https://*.ngrok-free.app',
    'http://localhost:8000',
    'http://127.0.0.1:8000',
]

try:
    from .local import *
except ImportError:
    pass
