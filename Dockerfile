# ── Stage 1: base image with all system deps for Playwright ───────────────────
FROM python:3.11-slim

# Playwright needs these system libs for Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium core
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Font rendering
    fonts-liberation \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright's bundled Chromium
# (this is what perchance library uses under the hood)
RUN playwright install chromium --with-deps

# Copy app source
COPY main.py .

# Expose port (Render uses $PORT env var)
EXPOSE 10000

# Start server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
