"""
Perchance Image Generation API
--------------------------------
FastAPI server wrapping the eeemoon/perchance library.

Key facts from reading the actual source (imagegen.py):
  - Library uses pw.firefox.launch(headless=True) — Firefox, NOT Chromium
  - ImageGenerator is a plain class: gen = perchance.ImageGenerator()
  - Must call await gen.refresh() to pre-fetch the user key
  - gen.image() returns an ImageResponse used as:
      async with await gen.image(prompt) as result:
          binary = await result.download()
  - Valid shapes: 'portrait' | 'square' | 'landscape' ONLY
  - ImageResponse.download() returns io.BytesIO

Endpoints:
  GET  /health   → health check
  POST /generate → generate image(s), returns list of base64 data URLs
"""

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
API_SECRET      = os.getenv("API_SECRET", "change-me-in-production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGES      = int(os.getenv("MAX_IMAGES", "4"))

# ── Single shared generator ────────────────────────────────────────────────────
# ImageGenerator is a plain class (not an async context manager).
# We create one instance and keep it alive for the process lifetime.
# refresh() is called at startup to pre-fetch the Firefox/user key,
# and again automatically by the library whenever the key expires.
_generator = None


async def get_generator():
    global _generator
    if _generator is None:
        import perchance
        log.info("Creating ImageGenerator...")
        _generator = perchance.ImageGenerator()
        log.info("Pre-fetching user key via Firefox (this takes ~15s)...")
        await _generator.refresh()
        log.info("User key acquired ✓")
    return _generator


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up at startup so first request isn't slow
    try:
        await get_generator()
    except Exception as exc:
        # Don't crash startup — will retry on first request
        log.warning("Startup warm-up failed (will retry on first request): %s", exc)
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Perchance Image API", version="4.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # must be False when allow_origins=["*"]
    allow_methods=["*"],       # includes OPTIONS preflight
    allow_headers=["*"],
    max_age=3600,
)

# ── Auth ───────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Shape mapping ──────────────────────────────────────────────────────────────
# The library ONLY supports these three shapes — from reading imagegen.py source
SHAPE_MAP = {
    "square":    "square",     # 512x512
    "portrait":  "portrait",   # 512x768
    "landscape": "landscape",  # 768x512
    # Map our UI-only shapes to nearest supported
    "tall":      "portrait",   # closest to 9:16
    "wide":      "landscape",  # closest to 16:9
}


# ── Models ─────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(
        default="blurry, low quality, watermark, text, deformed, ugly, bad anatomy",
        max_length=1000,
    )
    shape: Optional[str] = Field(default="portrait")
    num_images: Optional[int] = Field(default=1, ge=1, le=4)
    seed: Optional[int] = Field(default=None)


class ImageResult(BaseModel):
    index: int
    data_url: str        # "data:image/png;base64,…" — ready for <img src>
    seed: Optional[int]
    shape: str
    width: Optional[int]
    height: Optional[int]


class GenerateResponse(BaseModel):
    images: list[ImageResult]
    prompt: str
    elapsed_seconds: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "generator_ready": _generator is not None,
        "has_key": _generator is not None and _generator._key is not None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    t0    = time.perf_counter()
    shape = SHAPE_MAP.get(req.shape or "portrait", "portrait")
    num   = min(req.num_images or 1, MAX_IMAGES)
    seed  = req.seed if req.seed is not None else -1  # -1 = random in the lib

    log.info("generate | prompt=%r  shape=%s  n=%d", req.prompt[:60], shape, num)

    gen = await get_generator()
    results: list[ImageResult] = []

    for i in range(num):
        attempt = 0
        while attempt < 3:
            try:
                log.info("  image %d/%d attempt %d…", i + 1, num, attempt + 1)

                # ── Correct usage per the actual source code ───────────────
                # gen.image() returns ImageResponse which IS an async CM.
                # Pattern: async with await gen.image(prompt) as result:
                # ──────────────────────────────────────────────────────────
                async with await gen.image(
                    req.prompt,
                    negative_prompt=req.negative_prompt,
                    shape=shape,
                    seed=seed,
                ) as result:
                    binary   = await result.download()   # returns io.BytesIO
                    b64      = base64.b64encode(binary.read()).decode()
                    data_url = f"data:image/png;base64,{b64}"

                    results.append(ImageResult(
                        index=i,
                        data_url=data_url,
                        seed=result.seed,
                        shape=shape,
                        width=result.width,
                        height=result.height,
                    ))
                    log.info("  image %d/%d done ✓ (%dx%d)",
                             i + 1, num, result.width, result.height)
                    break  # success

            except Exception as exc:
                attempt += 1
                log.warning("  image %d attempt %d failed: %s", i + 1, attempt, exc)

                # If the key expired mid-session, refresh and retry
                if "key" in str(exc).lower() or "auth" in str(exc).lower():
                    log.info("  Key issue detected — refreshing...")
                    try:
                        await gen.refresh()
                    except Exception as ref_exc:
                        log.error("  refresh failed: %s", ref_exc)

                if attempt >= 3:
                    log.error("  giving up on image %d", i + 1)
                    if i == 0 and num == 1:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Generation failed after 3 attempts: {exc}",
                        )
                    break

                await asyncio.sleep(5 * attempt)

        # Pause between images to respect rate limits
        if i < num - 1:
            await asyncio.sleep(3)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("complete in %.1fs — %d image(s) returned", elapsed, len(results))

    return GenerateResponse(
        images=results,
        prompt=req.prompt,
        elapsed_seconds=elapsed,
    )
