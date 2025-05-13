# ── Dockerfile ─────────────────────────────────────────────
# Build arg lets you bump Python without touching the cache
ARG PYTHON_VERSION=3.12-slim
FROM python:${PYTHON_VERSION}

# 1. Basic hardening + faster pip
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        build-essential libpq-dev gettext && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

# 2. Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# 3. Copy project *after* deps so code changes don’t bust the cache
COPY . .

# 4. Collect static assets once at build time
RUN python manage.py collectstatic --noinput

# 5. Non-root user (avoids “don’t run as root” warnings in many clouds)
RUN adduser --disabled-password --gecos '' appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "website.wsgi:application", "--bind", "0.0.0.0:8000"]
# ───────────────────────────────────────────────────────────
