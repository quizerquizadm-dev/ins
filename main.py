"""
Perchance Image Generation API
--------------------------------
FastAPI server using Pollinations.ai — free, no auth, no Playwright needed.

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
from urllib.parse import quote

import httpx
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

# Pollinations base URL
POLLINATIONS_URL = "https://image.pollinations.ai/prompt"


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — Pollinations.ai backend, no Playwright needed ✓")
    yield
    log.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Perchance Image API",
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# ── Auth ───────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


# ── Shape → width/height map ───────────────────────────────────────────────────
SHAPE_DIMENSIONS = {
    "square":    (512, 512),
    "portrait":  (512, 768),
    "landscape": (768, 512),
    "tall":      (384, 680),
    "wide":      (680, 384),
}


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
    data_url: str        # "data:image/jpeg;base64,…" ready for <img src>
    seed: Optional[int]
    shape: str


class GenerateResponse(BaseModel):
    images: list[ImageResult]
    prompt: str
    elapsed_seconds: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "backend": "pollinations.ai"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    t0    = time.perf_counter()
    shape = req.shape if req.shape in SHAPE_DIMENSIONS else "portrait"
    num   = min(req.num_images or 1, MAX_IMAGES)
    w, h  = SHAPE_DIMENSIONS[shape]

    # Combine prompt + negative prompt hint for Pollinations
    full_prompt = req.prompt
    if req.negative_prompt:
        full_prompt += f" | avoid: {req.negative_prompt}"

    log.info("generate | prompt=%r  shape=%s  size=%dx%d  n=%d",
             req.prompt[:60], shape, w, h, num)

    results: list[ImageResult] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i in range(num):
            # Each image gets a unique seed (or increments from user seed)
            seed = (req.seed + i) if req.seed is not None else (int(time.time()) + i)

            encoded_prompt = quote(full_prompt)
            url = (
                f"{POLLINATIONS_URL}/{encoded_prompt}"
                f"?width={w}&height={h}&seed={seed}&nologo=true&enhance=false"
            )

            attempt = 0
            while attempt < 3:
                try:
                    log.info("  fetching image %d/%d (attempt %d) seed=%d…",
                             i + 1, num, attempt + 1, seed)

                    response = await client.get(url)

                    if response.status_code == 200:
                        b64      = base64.b64encode(response.content).decode()
                        data_url = f"data:image/jpeg;base64,{b64}"
                        results.append(ImageResult(
                            index=i,
                            data_url=data_url,
                            seed=seed,
                            shape=shape,
                        ))
                        log.info("  image %d/%d done ✓  (%d bytes)",
                                 i + 1, num, len(response.content))
                        break
                    else:
                        raise Exception(
                            f"Pollinations returned HTTP {response.status_code}"
                        )

                except Exception as exc:
                    attempt += 1
                    log.warning("  image %d attempt %d failed: %s",
                                i + 1, attempt, exc)
                    if attempt >= 3:
                        log.error("  giving up on image %d", i + 1)
                        if i == 0 and num == 1:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Image generation failed: {exc}",
                            )
                        break
                    await asyncio.sleep(3 * attempt)

            # Small pause between images
            if i < num - 1:
                await asyncio.sleep(1)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("request complete in %.1fs — %d image(s) returned",
             elapsed, len(results))

    return GenerateResponse(
        images=results,
        prompt=req.prompt,
        elapsed_seconds=elapsed,
    )
