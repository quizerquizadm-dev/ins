# ── Base image ────────────────────────────────────────────────────────────────
# python:3.11-slim uses Debian Bookworm where libasound2 was renamed to
# libasound2t64 — using the correct package name here.
FROM python:3.11-slim

# ── System deps for Chromium ──────────────────────────────────────────────────
# This is the complete list needed by Playwright's Chromium on Debian Bookworm.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Networking / crypto
    libssl3 \
    ca-certificates \
    # X11 / display (needed even in headless mode)
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libxkbcommon0 \
    # Graphics
    libgbm1 \
    libdrm2 \
    libgl1 \
    # GTK / Pango / Cairo (UI toolkit deps, required by Chromium)
    libgtk-3-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libcairo-gobject2 \
    # ATK / AT-SPI (accessibility layer, required by Chromium)
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libatspi2.0-0 \
    # Cups / DBus
    libcups2 \
    libdbus-1-3 \
    # NSS / NSPR
    libnss3 \
    libnspr4 \
    # Audio (renamed in Bookworm)
    libasound2t64 \
    # Fonts
    fonts-liberation \
    fonts-noto-color-emoji \
    # Misc tools
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Install Playwright's Chromium ─────────────────────────────────────────────
# We install only chromium (not firefox/webkit) to keep image size down.
# --with-deps is skipped because we already installed deps above — this avoids
# a second apt run and keeps layer caching efficient.
RUN playwright install chromium

# Verify the binary exists (fails the build early if something is wrong)
RUN ls /root/.cache/ms-playwright/chromium*/chrome-linux/chrome

# ── Application code ──────────────────────────────────────────────────────────
COPY main.py .

# ── Runtime ───────────────────────────────────────────────────────────────────
EXPOSE 10000

# PORT env var is set by Render automatically; default to 10000 for local runs.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
