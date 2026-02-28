"""
Perchance Image Generation API
--------------------------------
FastAPI server that wraps the eeemoon/perchance library.
Uses Xvfb virtual display so Chromium runs non-headless,
bypassing Perchance's bot/fingerprint detection.

Endpoints:
  GET  /health         → health check (shows DISPLAY value)
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
API_SECRET      = os.getenv("API_SECRET", "change-me-in-production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGES      = int(os.getenv("MAX_IMAGES", "4"))

# ── Ensure DISPLAY is set for Xvfb so Chromium runs non-headless ──────────────
# Set by Dockerfile CMD before uvicorn starts. We defensively set it
# here too in case of manual/local runs.
if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":99"
    log.warning("DISPLAY was not set — defaulting to :99")


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — DISPLAY=%s", os.environ.get("DISPLAY", "NOT SET"))
    try:
        import perchance  # noqa: F401
        log.info("perchance import OK ✓")
    except Exception as exc:
        log.error("perchance import FAILED: %s", exc)
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Perchance Image API",
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],      # includes OPTIONS preflight
    allow_headers=["*"],
    max_age=3600,
)

# ── Auth ───────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Valid shapes ───────────────────────────────────────────────────────────────
VALID_SHAPES = {"square", "portrait", "landscape", "tall", "wide"}


# ── Request / Response models ──────────────────────────────────────────────────
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
    data_url: str        # "data:image/png;base64,…" ready to use in <img src>
    seed: Optional[int]
    shape: str


class GenerateResponse(BaseModel):
    images: list[ImageResult]
    prompt: str
    elapsed_seconds: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — no auth required. Shows DISPLAY so you can verify Xvfb."""
    return {"status": "ok", "display": os.environ.get("DISPLAY", "NOT SET")}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    import perchance

    t0    = time.perf_counter()
    shape = req.shape if req.shape in VALID_SHAPES else "portrait"
    num   = min(req.num_images or 1, MAX_IMAGES)

    log.info("generate | prompt=%r  shape=%s  n=%d  display=%s",
             req.prompt[:60], shape, num, os.environ.get("DISPLAY"))

    results: list[ImageResult] = []

    # ── Correct usage per eeemoon/perchance README ─────────────────────────
    # ImageGenerator must be used as an async context manager.
    # DISPLAY=:99 makes Playwright launch full Chromium via Xvfb virtual
    # display, which passes Perchance's browser fingerprinting check.
    # ──────────────────────────────────────────────────────────────────────
    async with perchance.ImageGenerator() as gen:
        for i in range(num):
            attempt = 0
            while attempt < 3:
                try:
                    log.info("  generating image %d/%d (attempt %d)…",
                             i + 1, num, attempt + 1)

                    result = await gen.image(
                        req.prompt,
                        negative_prompt=req.negative_prompt,
                        shape=shape,
                        seed=req.seed,
                    )

                    binary   = await result.download()
                    b64      = base64.b64encode(binary.read()).decode()
                    data_url = f"data:image/png;base64,{b64}"

                    results.append(ImageResult(
                        index=i,
                        data_url=data_url,
                        seed=getattr(result, "seed", req.seed),
                        shape=shape,
                    ))
                    log.info("  image %d/%d done ✓", i + 1, num)
                    break

                except Exception as exc:
                    attempt += 1
                    log.warning("  image %d attempt %d failed: %s",
                                i + 1, attempt, exc)
                    if attempt >= 3:
                        log.error("  giving up on image %d after 3 attempts", i + 1)
                        if i == 0 and num == 1:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Perchance generation failed after 3 attempts: {exc}",
                            )
                        break
                    await asyncio.sleep(5 * attempt)

            if i < num - 1:
                await asyncio.sleep(3)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("request complete in %.1fs — %d image(s) returned",
             elapsed, len(results))

    return GenerateResponse(
        images=results,
        prompt=req.prompt,
        elapsed_seconds=elapsed,
    )
