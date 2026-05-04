# Neural Router v2.0
# Build:  docker build -t neural-router .
# Run:    docker run -p 5000:5000 neural-router
# Dev:    docker run -p 5000:5000 -e LOG_LEVEL=DEBUG neural-router

FROM python:3.11-slim

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure model directory is writable
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Use gunicorn for production; app.py trains model on first boot if needed
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "300", "app:app"]
