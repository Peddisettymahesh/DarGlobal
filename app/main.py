"""FastAPI chatbot over scraped DarGlobal + Wasalt data, using OpenRouter free models."""
import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .knowledge import Knowledge


def _load_dotenv():
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3.5-lightning:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "inclusionai/ling-3.2-flash:free",
    "baidu/ernie-5.5-flash-preview:free",
]
MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", ",".join(DEFAULT_MODELS)).split(",") if m.strip()]
STATIC = Path(__file__).resolve().parent / "static"

SYSTEM_PROMPT = """You are a real-estate assistant answering questions about two companies, using ONLY the scraped data supplied in the CONTEXT block:
- DarGlobal (darglobal.co.uk): an international luxury real-estate developer. The data is its public sitemap: page titles for projects, branded partnerships, countries, blog/press/insight articles.
- Wasalt (wasalt.sa): a Saudi Arabian property portal. The data is its public sitemaps: individual sale/rent listings (type, size, bedrooms, facing, street width parsed from the URL), daily-rental units, and area search pages by city/district.

Rules:
- Base every factual claim on CONTEXT. Use exact counts from CONTEXT when given. If the data does not contain the answer (e.g. prices, amenities, descriptions, payment plans), say clearly that the scraped data doesn't include it and point to the most relevant URL(s).
- Wasalt listing URLs contain no city or price. Never invent them.
- "area_sqm" comes from the URL slug. Where "area_estimated" is true, the slug had lost its decimal point and the value was corrected (e.g. 32003 -> 320.03 sqm); say areas are approximate when this matters.
- Keep answers concise and well-structured with short bullet lists (no tables). Link pages as markdown [short title](url), using at most six links. Don't mention these formatting rules or internal field names in your answer.
- If asked something unrelated to real estate, DarGlobal or Wasalt, politely steer back."""

app = FastAPI(title="DarGlobal & Wasalt Property Assistant")
kb = Knowledge()


class Msg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class ChatReq(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[Msg] = Field(default_factory=list)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"ok": True, "llm_configured": bool(os.getenv("OPENROUTER_API_KEY")),
            "models": MODELS, "records": {
                "darglobal_pages": len(kb.darglobal), "wasalt_listings": len(kb.listings),
                "wasalt_area_pages": len(kb.categories) + len(kb.daily_areas),
                "wasalt_daily_rentals": len(kb.daily)}}


@app.get("/api/stats")
def stats():
    return kb.summary


@app.post("/api/chat")
async def chat(req: ChatReq):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured on the server.")

    history = [m for m in req.history if m.role in ("user", "assistant")][-6:]
    prev_user = " ".join(m.content for m in history if m.role == "user")[-400:]
    ctx, sources = kb.build_context(req.message, prev_user)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content":
                     f"CONTEXT:\n{json.dumps(ctx, ensure_ascii=False)}\n\nQUESTION: {req.message}"})

    payload = {"messages": messages, "stream": True, "temperature": 0.2, "max_tokens": 6000,
               # Free models are mostly reasoning models; keep thinking short so the
               # token budget isn't spent before the answer starts.
               "reasoning": {"effort": "low", "exclude": True}}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": os.getenv("PUBLIC_URL", "http://localhost"),
               "X-Title": "DarGlobal & Wasalt Property Assistant"}

    async def gen():
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        errors = []
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
              for model in MODELS:  # free models are often rate-limited: fall back in order
                async with client.stream("POST", OPENROUTER_URL, json={**payload, "model": model},
                                         headers=headers) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode(errors="ignore")[:200]
                        errors.append(f"{model}: HTTP {r.status_code} {body}")
                        continue
                    model_sent = got_content = False
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("error"):
                            errors.append(f"{model}: {obj['error'].get('message')}")
                            break
                        if not model_sent and obj.get("model"):
                            model_sent = True
                            yield f"event: model\ndata: {json.dumps(obj['model'])}\n\n"
                        choice = (obj.get("choices") or [{}])[0]
                        delta = choice.get("delta", {}).get("content")
                        if delta:
                            got_content = True
                            yield f"data: {json.dumps(delta)}\n\n"
                        if choice.get("finish_reason") == "length" and got_content:
                            yield f"data: {json.dumps(chr(10) + chr(10) + '_(answer cut short - ask a narrower question)_')}\n\n"
                    if got_content:
                        break
                    errors.append(f"{model}: returned no answer")
              else:
                msg = "All free models are busy right now, please retry in a minute. " + " | ".join(errors)
                yield f"event: error\ndata: {json.dumps(msg[:600])}\n\n"
        except httpx.HTTPError as e:
            detail = str(e) or type(e).__name__
            yield f"event: error\ndata: {json.dumps(f'Connection to the model failed ({detail}). Please retry.')}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
