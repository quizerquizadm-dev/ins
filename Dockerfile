FROM python:3.11-slim

# System deps for Playwright Firefox (much lighter than Chromium)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Firefox core deps
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libxt6 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    fonts-liberation \
    # aiohttp needs this
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Firefox — the perchance library uses pw.firefox, NOT chromium
RUN playwright install firefox --with-deps

COPY main.py .

EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
