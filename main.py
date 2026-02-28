"""
Perchance Image Generation API
--------------------------------
FastAPI server using Pollinations.ai — free, no auth, no Playwright needed.
Uses gen.pollinations.ai (new unified endpoint) with fallback to pollinations.ai/p/

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

# Two endpoints to try — new unified one first, legacy as fallback
ENDPOINTS = [
    "https://image.pollinations.ai/prompt",   # primary
    "https://pollinations.ai/p",              # fallback
]


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting up — Pollinations.ai backend ✓")
    yield


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Perchance Image API", version="3.1.0", lifespan=lifespan)

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

# ── Shape → width/height ───────────────────────────────────────────────────────
SHAPE_DIMENSIONS = {
    "square":    (512, 512),
    "portrait":  (512, 768),
    "landscape": (768, 512),
    "tall":      (384, 680),
    "wide":      (680, 384),
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
    data_url: str
    seed: Optional[int]
    shape: str

class GenerateResponse(BaseModel):
    images: list[ImageResult]
    prompt: str
    elapsed_seconds: float


# ── Helper: fetch one image, trying both endpoints ────────────────────────────
async def fetch_image(client: httpx.AsyncClient, prompt: str, w: int, h: int, seed: int) -> bytes:
    encoded = quote(prompt)
    last_exc = None

    for base_url in ENDPOINTS:
        url = f"{base_url}/{encoded}?width={w}&height={h}&seed={seed}&nologo=true&enhance=false"
        log.info("    trying %s", base_url)
        try:
            resp = await client.get(url, follow_redirects=True, timeout=90.0)
            if resp.status_code == 200 and len(resp.content) > 1000:
                log.info("    ✓ got %d bytes from %s", len(resp.content), base_url)
                return resp.content
            else:
                last_exc = Exception(f"HTTP {resp.status_code} from {base_url}")
                log.warning("    ✗ %s", last_exc)
        except Exception as exc:
            last_exc = exc
            log.warning("    ✗ %s error: %s", base_url, exc)

    raise last_exc or Exception("All endpoints failed")


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

    # Weave negative prompt into the positive prompt
    full_prompt = req.prompt
    if req.negative_prompt:
        full_prompt += f" | avoid: {req.negative_prompt}"

    log.info("generate | prompt=%r  shape=%s  size=%dx%d  n=%d",
             req.prompt[:60], shape, w, h, num)

    results: list[ImageResult] = []

    async with httpx.AsyncClient() as client:
        for i in range(num):
            seed = (req.seed + i) if req.seed is not None else (int(time.time() * 1000 + i) % 99999999)

            attempt = 0
            while attempt < 3:
                try:
                    log.info("  image %d/%d attempt %d  seed=%d",
                             i + 1, num, attempt + 1, seed)

                    content  = await fetch_image(client, full_prompt, w, h, seed)
                    b64      = base64.b64encode(content).decode()
                    data_url = f"data:image/jpeg;base64,{b64}"

                    results.append(ImageResult(
                        index=i, data_url=data_url, seed=seed, shape=shape
                    ))
                    log.info("  image %d/%d done ✓", i + 1, num)
                    break

                except Exception as exc:
                    attempt += 1
                    log.warning("  image %d attempt %d failed: %s", i + 1, attempt, exc)
                    if attempt >= 3:
                        if i == 0 and num == 1:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Image generation failed: {exc}",
                            )
                        break
                    await asyncio.sleep(4 * attempt)

            if i < num - 1:
                await asyncio.sleep(1)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("complete in %.1fs — %d image(s)", elapsed, len(results))

    return GenerateResponse(images=results, prompt=req.prompt, elapsed_seconds=elapsed)
