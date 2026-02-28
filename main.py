"""
Perchance Image Generation API

Based on reading the ACTUAL GitHub source:
  generator.py  → uses self._pw.chromium.launch(headless=True)
  imagegen.py   → ImageGenerator extends Generator, uses async with
  ImageResult   → NOT an async context manager, just await result.download()

Correct usage per README:
    async with ImageGenerator() as gen:
        result = await gen.image(prompt, shape='landscape')
        binary = await result.download()
        image = Image.open(binary)
"""

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "*",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Server starting up ✓")
    yield
    log.info("Server shutting down.")


app = FastAPI(title="Perchance Image API", version="5.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Inject CORS headers on ALL error responses — fixes "CORS error" masking real errors
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=CORS_HEADERS,
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers=CORS_HEADERS,
    )

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_key(key: Optional[str] = Security(api_key_header)):
    if key != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key

# Only shapes the library supports — from imagegen.py source
SHAPE_MAP = {
    "square":    "square",
    "portrait":  "portrait",
    "landscape": "landscape",
    "tall":      "portrait",    # fallback
    "wide":      "landscape",   # fallback
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
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    from perchance import ImageGenerator

    t0    = time.perf_counter()
    shape = SHAPE_MAP.get(req.shape or "portrait", "portrait")
    num   = min(req.num_images or 1, MAX_IMAGES)
    seed  = req.seed if req.seed is not None else -1

    log.info("generate | prompt=%r  shape=%s  n=%d", req.prompt[:60], shape, num)

    results: list[ImageResult] = []

    # ── Correct usage confirmed from README + generator.py source ─────────
    # ImageGenerator IS an async context manager (has __aenter__/__aexit__)
    # gen.image() returns ImageResult directly — NOT a context manager
    # Just: result = await gen.image(...) then binary = await result.download()
    # ──────────────────────────────────────────────────────────────────────
    async with ImageGenerator() as gen:
        for i in range(num):
            attempt = 0
            while attempt < 3:
                try:
                    log.info("  image %d/%d attempt %d…", i + 1, num, attempt + 1)

                    result = await gen.image(
                        req.prompt,
                        negative_prompt=req.negative_prompt,
                        shape=shape,
                        seed=seed,
                    )

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
                    log.warning("  image %d attempt %d failed: %s",
                                i + 1, attempt, exc)
                    if attempt >= 3:
                        if i == 0 and num == 1:
                            raise HTTPException(
                                status_code=502,
                                detail=f"Generation failed: {exc}",
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
