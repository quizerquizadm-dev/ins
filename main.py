"""
Perchance Image Generation API
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

API_SECRET      = os.getenv("API_SECRET", "change-me-in-production")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
MAX_IMAGES      = int(os.getenv("MAX_IMAGES", "4"))

_generator     = None
_startup_error = None


async def get_generator():
    global _generator
    if _generator is None:
        import perchance
        log.info("Creating ImageGenerator…")
        # Per the README: just instantiate — no refresh() call needed
        _generator = perchance.ImageGenerator()
        log.info("ImageGenerator ready ✓")
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _startup_error
    try:
        await get_generator()
    except Exception as exc:
        _startup_error = str(exc)
        log.error("Startup failed: %s", exc)
        # Don't raise — let server start so /health can report the error
    yield


app = FastAPI(title="Perchance Image API", version="4.2.0", lifespan=lifespan)

# CORS must be registered before all routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# Only the 3 shapes the library actually supports
SHAPE_MAP = {
    "square":    "square",
    "portrait":  "portrait",
    "landscape": "landscape",
    "tall":      "portrait",   # fallback
    "wide":      "landscape",  # fallback
}

class GenerateRequest(BaseModel):
    prompt:          str           = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(
        default="blurry, low quality, watermark, text, deformed, ugly, bad anatomy")
    shape:           Optional[str] = Field(default="portrait")
    num_images:      Optional[int] = Field(default=1, ge=1, le=4)
    seed:            Optional[int] = Field(default=None)

class ImageResult(BaseModel):
    index:    int
    data_url: str
    seed:     Optional[int]
    shape:    str
    width:    Optional[int]
    height:   Optional[int]

class GenerateResponse(BaseModel):
    images:          list[ImageResult]
    prompt:          str
    elapsed_seconds: float


@app.get("/health")
async def health():
    # Always returns 200 — never crashes — so CORS headers are always present
    return {
        "status":          "ok" if _startup_error is None else "degraded",
        "generator_ready": _generator is not None,
        "startup_error":   _startup_error,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    t0    = time.perf_counter()
    shape = SHAPE_MAP.get(req.shape or "portrait", "portrait")
    num   = min(req.num_images or 1, MAX_IMAGES)
    seed  = req.seed if req.seed is not None else -1

    log.info("generate | prompt=%r  shape=%s  n=%d", req.prompt[:60], shape, num)

    gen = await get_generator()
    results: list[ImageResult] = []

    for i in range(num):
        attempt = 0
        while attempt < 3:
            try:
                log.info("  image %d/%d attempt %d…", i + 1, num, attempt + 1)

                # Correct usage per README + actual source:
                # gen.image() returns ImageResponse which is an async context manager
                async with await gen.image(
                    req.prompt,
                    negative_prompt=req.negative_prompt,
                    shape=shape,
                    seed=seed,
                ) as result:
                    binary   = await result.download()
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
                    break

            except Exception as exc:
                attempt += 1
                log.warning("  image %d attempt %d failed: %s", i + 1, attempt, exc)
                if attempt >= 3:
                    if i == 0 and num == 1:
                        raise HTTPException(
                            status_code=502,
                            detail=f"Generation failed after 3 attempts: {exc}",
                        )
                    break
                await asyncio.sleep(5 * attempt)

        if i < num - 1:
            await asyncio.sleep(3)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("complete in %.1fs — %d image(s)", elapsed, len(results))

    return GenerateResponse(
        images=results,
        prompt=req.prompt,
        elapsed_seconds=elapsed,
    )
