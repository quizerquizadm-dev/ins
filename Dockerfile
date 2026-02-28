FROM python:3.11-slim

# System deps for Playwright full Chromium + Xvfb virtual display
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
    # Virtual display — makes Chromium think it has a real screen
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install full Chromium (NOT headless-shell) so fingerprinting passes
RUN playwright install chromium --with-deps

# Copy app source
COPY main.py .

# Expose port
EXPOSE 10000

# Start Xvfb virtual display on :99, then launch the server
# DISPLAY=:99 makes Playwright use the virtual display → Chromium is non-headless
CMD ["sh", "-c", "Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset & sleep 1 && DISPLAY=:99 uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
