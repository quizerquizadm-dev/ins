"""
Perchance Image Generation API - Render deployment (v9)

The challenge: Perchance uses Cloudflare which serves a JS challenge to
datacenter IPs. ctx.request (plain HTTP) gets blocked with 403.

The fix: navigate a real browser page to the verifyUser endpoint and intercept
the response BEFORE Cloudflare can redirect - using page.route() to capture
the raw JSON response body at the network layer while the full JS engine
handles the Cloudflare challenge automatically.

Flow:
  1. Launch Chromium once; reuse across requests
  2. Open a page, navigate to perchance.org first (establishes CF cookies/tokens)
  3. Use page.route() to intercept the verifyUser response and grab the body
  4. Cache userKey 4 min, refresh lazily
  5. POST /generate via aiohttp with CF cookies from the browser context
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from playwright.async_api import async_playwright, Browser, BrowserContext, Playwright, Route, Request as PWRequest
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

API_SECRET = os.getenv("API_SECRET", "change-me-in-production")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "4"))
BASE_URL   = "https://image-generation.perchance.org/api"
KEY_TTL    = 240  # seconds

CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "*",
}

_pw:       Optional[Playwright]     = None
_browser:  Optional[Browser]        = None
_context:  Optional[BrowserContext] = None
_user_key: Optional[str]            = None
_key_ts:   float                    = 0.0
_http:     Optional[aiohttp.ClientSession] = None
_lock      = asyncio.Lock()


async def _get_context() -> BrowserContext:
    global _pw, _browser, _context
    if _context is not None:
        return _context
    log.info("Launching Chromium...")
    _pw = await async_playwright().start()
    _browser = await _pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--single-process",
        ],
    )
    _context = await _browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    log.info("Chromium ready")
    return _context


async def _get_user_key() -> str:
    """
    Get userKey by:
    1. Loading perchance.org in a real browser page (solves CF challenge, gets cookies)
    2. Using page.route() to intercept the verifyUser network response
    3. Extracting userKey from the intercepted JSON body

    This works because Playwright's full browser engine executes Cloudflare's
    JS challenge, earns the cf_clearance cookie, and subsequent requests from
    that browser context pass through normally.
    """
    global _user_key, _key_ts

    async with _lock:
        age = time.monotonic() - _key_ts
        if _user_key and age < KEY_TTL:
            return _user_key

        log.info("Refreshing userKey (age=%.0fs)...", age)
        ctx = await _get_context()

        intercepted_body: dict = {}
        verify_done = asyncio.Event()

        async def handle_route(route: Route, request: PWRequest):
            # Let the request proceed, then capture the response body
            response = await route.fetch()
            body = await response.text()
            log.info("intercepted verifyUser status=%d body(200)=%r",
                     response.status, body[:200])
            intercepted_body["text"] = body
            intercepted_body["status"] = response.status
            verify_done.set()
            # Fulfill with the real response so the page doesn't break
            await route.fulfill(response=response, body=body)

        async with await ctx.new_page() as page:
            # Intercept the verifyUser call at network level
            await page.route("**/verifyUser**", handle_route)

            # Navigate to the actual generator page - this:
            # a) Triggers Cloudflare JS challenge (browser solves it automatically)
            # b) Earns cf_clearance cookie for future requests
            # c) The page itself calls verifyUser, which we intercept
            log.info("Navigating to perchance generator page...")
            try:
                await page.goto(
                    "https://perchance.org/ai-text-to-image-generator",
                    wait_until="domcontentloaded",
                    timeout=60_000,
                )
            except Exception as e:
                log.warning("page.goto raised (may be ok if route fired): %s", e)

            # Wait for the intercepted verifyUser response (up to 30s)
            try:
                await asyncio.wait_for(verify_done.wait(), timeout=30.0)
            except asyncio.TimeoutError:
                # Page loaded but didn't call verifyUser automatically
                # Trigger it manually via page.evaluate
                log.info("verifyUser not auto-called, triggering manually...")
                result = await page.evaluate("""
                    async () => {
                        try {
                            const r = await fetch(
                                'https://image-generation.perchance.org/api/verifyUser'
                                + '?thread=0&__cacheBust=' + Math.random()
                            );
                            return await r.text();
                        } catch(e) { return 'ERR:' + e.message; }
                    }
                """)
                intercepted_body["text"] = result
                intercepted_body["status"] = 200
                log.info("manual verifyUser result(200): %r", result[:200])

        content = intercepted_body.get("text", "")
        status  = intercepted_body.get("status", 0)

        marker = '"userKey":"'
        start  = content.find(marker)
        if start == -1:
            if "too_many_requests" in content:
                raise HTTPException(502, "Perchance rate limit - try again later")
            raise HTTPException(
                502,
                f"userKey not found (status={status}). Body: {content[:300]}"
            )

        start += len(marker)
        end    = content.find('"', start)
        key    = content[start:end] if end != -1 else ""
        if not key:
            raise HTTPException(502, f"Empty/malformed userKey. Body: {content[:200]}")

        _user_key = key
        _key_ts   = time.monotonic()
        log.info("userKey refreshed: %.12s...", key)
        return key


async def _get_http() -> aiohttp.ClientSession:
    global _http
    if _http is None or _http.closed:
        # Pull CF cookies from the browser context to use in aiohttp requests
        cookies = {}
        if _context:
            try:
                cf_cookies = await _context.cookies(
                    ["https://perchance.org", "https://image-generation.perchance.org"]
                )
                cookies = {c["name"]: c["value"] for c in cf_cookies}
                log.info("Loaded %d cookies from browser context", len(cookies))
            except Exception as e:
                log.warning("Could not load cookies: %s", e)
        _http = aiohttp.ClientSession(cookies=cookies)
    return _http


async def _generate_image(
    prompt: str,
    negative_prompt: str,
    shape: str,
    seed: int,
) -> tuple[io.BytesIO, int, int, int]:

    resolution = {
        "portrait":  "512x768",
        "landscape": "768x512",
        "square":    "768x768",
    }.get(shape, "512x768")

    key  = await _get_user_key()
    http = await _get_http()

    url = (
        f"{BASE_URL}/generate"
        f"?userKey={key}"
        f"&requestId=aiImageCompletion{random.randint(0, 2**30)}"
        f"&__cacheBust={random.random()}"
    )
    body = {
        "generatorName":  "ai-image-generator",
        "channel":        "ai-text-to-image-generator",
        "subChannel":     "public",
        "prompt":         prompt,
        "negativePrompt": negative_prompt,
        "seed":           seed,
        "resolution":     resolution,
        "guidanceScale":  7,
    }
    req_headers = {
        "Referer": "https://perchance.org/ai-text-to-image-generator",
        "Origin":  "https://perchance.org",
    }

    image_id = width = height = out_seed = None

    for attempt in range(60):
        try:
            async with http.post(
                url, json=body, headers=req_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json(content_type=None)
        except Exception as e:
            log.warning("generate HTTP error attempt %d: %s", attempt, e)
            await asyncio.sleep(5)
            continue

        status = data.get("status", "")
        log.info("generate status=%r attempt=%d", status, attempt)

        if status == "invalid_key":
            global _user_key, _http
            _user_key = None
            _http = None  # also reset http so cookies are refreshed
            key = await _get_user_key()
            http = await _get_http()
            url = (
                f"{BASE_URL}/generate"
                f"?userKey={key}"
                f"&requestId=aiImageCompletion{random.randint(0, 2**30)}"
                f"&__cacheBust={random.random()}"
            )
        elif status == "too_many_requests":
            raise HTTPException(429, "Perchance rate limit")
        elif status == "invalid_data":
            raise HTTPException(400, f"Perchance rejected request: {data}")
        elif status == "success":
            image_id = data["imageId"]
            width    = data["width"]
            height   = data["height"]
            out_seed = data.get("seed", seed)
            break
        else:
            await asyncio.sleep(4)
    else:
        raise HTTPException(504, "Timed out waiting for image generation")

    dl_url = f"{BASE_URL}/downloadTemporaryImage?imageId={image_id}"
    async with http.get(
        dl_url, headers=req_headers,
        timeout=aiohttp.ClientTimeout(total=30),
    ) as resp:
        if resp.status != 200:
            raise HTTPException(502, f"Image download failed: HTTP {resp.status}")
        raw = await resp.read()

    buf = io.BytesIO(raw)
    buf.seek(0)
    return buf, out_seed, width, height


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await _get_user_key()
        log.info("Pre-warm complete")
    except Exception as e:
        log.warning("Pre-warm failed (will retry on first request): %s", e)
    yield
    global _pw, _browser, _context, _http
    for obj, method in [(_context, "close"), (_browser, "close"), (_pw, "stop")]:
        if obj:
            try:
                await getattr(obj, method)()
            except Exception:
                pass
    if _http and not _http.closed:
        await _http.close()
    log.info("Shutdown complete")


app = FastAPI(title="Perchance Image API", version="9.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=CORS_HEADERS,
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)}, headers=CORS_HEADERS)


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(key: Optional[str] = Security(api_key_header)):
    if API_SECRET != "change-me-in-production" and key != API_SECRET:
        raise HTTPException(401, "Invalid or missing API key")
    return key


SHAPE_MAP = {
    "square":    "square",
    "portrait":  "portrait",
    "landscape": "landscape",
    "tall":      "portrait",
    "wide":      "landscape",
}


class GenerateRequest(BaseModel):
    prompt:          str           = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(
        default="blurry, low quality, watermark, text, deformed, ugly, bad anatomy"
    )
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
    return {"status": "ok", "browser_ready": _browser is not None, "has_key": bool(_user_key)}


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    _key: str = Security(verify_key),
):
    t0    = time.perf_counter()
    shape = SHAPE_MAP.get((req.shape or "portrait").lower(), "portrait")
    num   = min(req.num_images or 1, MAX_IMAGES)
    seed  = req.seed if req.seed is not None else random.randint(0, 99_999_999)

    log.info("generate prompt=%r shape=%s n=%d seed=%d", req.prompt[:60], shape, num, seed)
    results: list[ImageResult] = []

    for i in range(num):
        for attempt in range(3):
            try:
                log.info("  image %d/%d attempt %d", i + 1, num, attempt + 1)
                buf, out_seed, width, height = await _generate_image(
                    prompt=req.prompt, negative_prompt=req.negative_prompt or "",
                    shape=shape, seed=seed + i,
                )
                b64 = base64.b64encode(buf.read()).decode()
                results.append(ImageResult(
                    index=i, data_url=f"data:image/png;base64,{b64}",
                    seed=out_seed, shape=shape, width=width, height=height,
                ))
                log.info("  image %d/%d done (%dx%d)", i + 1, num, width, height)
                break
            except HTTPException:
                raise
            except Exception as exc:
                log.warning("  image %d attempt %d failed: %s", i + 1, attempt + 1, exc)
                if attempt == 2:
                    if i == 0 and num == 1:
                        raise HTTPException(502, f"Generation failed: {exc}")
                    break
                await asyncio.sleep(4 * (attempt + 1))

        if i < num - 1:
            await asyncio.sleep(2)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info("complete %.1fs %d image(s)", elapsed, len(results))
    return GenerateResponse(images=results, prompt=req.prompt, elapsed_seconds=elapsed)
