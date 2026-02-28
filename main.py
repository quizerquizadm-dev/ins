"""
Perchance Image Generation API
--------------------------------
FastAPI server that wraps the eeemoon/perchance library.
Deploy on Render.com free tier.

Endpoints:
  GET  /health         → health check
  POST /generate       → generate image(s), returns list of base64 URLs
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

# ── Config from environment ────────────────────────────────────────────────────
# Set these in Render dashboard → Environment tab
API_SECRET = os.getenv("API_SECRET", "change-me-in-production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "4"))  # cap per request

# ── Perchance generator singleton ─────────────────────────────────────────────
# We keep ONE ImageGenerator alive for the whole process lifetime.
# The library handles its own key refresh internally.
_generator = None
_generator_lock = asyncio.Lock()


async def get_generator():
    """Return the shared ImageGenerator, creating it on first call."""
    global _generator
    if _generator is None:
        async with _generator_lock:
            if _generator is None:
                log.info("Initialising Perchance ImageGenerator…")
                try:
                    import perchance
                    _generator = perchance.ImageGenerator()
                    log.info("ImageGenerator ready ✓")
                except Exception as exc:
                    log.error("Failed to create ImageGenerator: %s", exc)
                    raise
    return _generator


# ── Lifespan: warm up the generator at startup ────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_generator()
    except Exception as exc:
        log.warning("Warm-up failed (will retry on first request): %s", exc)
    yield
    # Cleanup on shutdown
    global _generator
    if _generator is not None:
        try:
            await _generator.__aexit__(None, None, None)
        except Exception:
            pass


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Perchance Image API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Request / Response models ──────────────────────────────────────────────────

# Perchance supported shapes (maps to resolution internally in the library)
VALID_SHAPES = {
    "square":    "square",      # 512×512
    "portrait":  "portrait",    # 512×768
    "landscape": "landscape",   # 768×512
    "tall":      "tall",        # 384×680  (9:16)
    "wide":      "wide",        # 680×384  (16:9)
}


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
    data_url: str        # "data:image/png;base64,…"  ready to use in <img src>
    seed: Optional[int]
    shape: str


class GenerateResponse(BaseModel):
    images: list[ImageResult]
    prompt: str
    elapsed_seconds: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "generator_ready": _generator is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    t0 = time.perf_counter()

    shape = VALID_SHAPES.get(req.shape, "portrait")
    num   = min(req.num_images or 1, MAX_IMAGES)

    log.info("generate | prompt=%r  shape=%s  n=%d", req.prompt[:60], shape, num)

    gen = await get_generator()

    results: list[ImageResult] = []

    # Generate images sequentially — Perchance free tier is rate-limited
    for i in range(num):
        attempt = 0
        while attempt < 3:
            try:
                async with await gen.image(
                    req.prompt,
                    negative_prompt=req.negative_prompt,
                    shape=shape,
                    seed=req.seed,
                ) as result:
                    binary = await result.download()
                    b64 = base64.b64encode(binary.read()).decode()
                    data_url = f"data:image/png;base64,{b64}"

                    results.append(ImageResult(
                        index=i,
                        data_url=data_url,
                        seed=getattr(result, "seed", req.seed),
                        shape=shape,
                    ))
                    log.info("  image %d/%d done", i + 1, num)
                    break  # success — exit retry loop

            except Exception as exc:
                attempt += 1
                log.warning("  image %d attempt %d failed: %s", i + 1, attempt, exc)
                if attempt >= 3:
                    # Don't crash entire request — include a failure placeholder
                    log.error("  giving up on image %d", i + 1)
                    # Re-raise only if ALL images failed
                    if i == 0 and num == 1:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Perchance generation failed after 3 attempts: {exc}",
                        )
                else:
                    await asyncio.sleep(5 * attempt)  # back-off: 5s, 10s

        # Brief pause between images to respect rate limits
        if i < num - 1:
            await asyncio.sleep(3)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("request complete in %.1fs", elapsed)

    return GenerateResponse(
        images=results,
        prompt=req.prompt,
        elapsed_seconds=elapsed,
    )
