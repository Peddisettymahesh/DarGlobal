---
title: DarGlobal & Wasalt Property Assistant
emoji: 🏙️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: AI chatbot over public DarGlobal and Wasalt data
---

# DarGlobal & Wasalt Property Assistant

An AI chatbot that answers questions about **DarGlobal** (luxury developer) and **Wasalt** (Saudi property portal) using publicly available data, powered by free LLMs on OpenRouter.

## How it works

1. **Scraper** (`scraper/scrape.py`): both sites block page bodies behind bot protection (DarGlobal uses Incapsula, Wasalt uses Cloudflare). The scraper therefore collects what each site publishes openly for crawlers: the XML sitemaps listed in their `robots.txt`. URL slugs are parsed into structured records.

   | Dataset | Records | Fields |
   |---|---|---|
   | DarGlobal pages | 479 | section (project, partner, blog, press, insight), title, country, property type |
   | Wasalt listings | 60,000 | sale/rent, type, area (sqm), bedrooms, facing, street width, URL |
   | Wasalt area pages | 8,634 | city, district, category, bedrooms, price band |
   | Wasalt daily rentals | 2,684 units, plus 5,054 area pages | type, city, district |

2. **Retrieval** (`app/knowledge.py`):
   - Filters are pulled from the question (purpose, type, bedrooms, sqm, facing, city).
   - Those filters run against the structured data to get exact counts and examples.
   - A TF-IDF index finds related DarGlobal and Wasalt pages.
   - A dataset summary is always included.
3. **LLM** (`app/main.py`): the context goes to OpenRouter with free models and automatic fallback. The default list is `google/gemma-4-31b-it:free`, then `nvidia/nemotron-3.5-lightning:free`, then `google/gemma-4-26b-a4b-it:free`. Responses stream to a web chat UI that shows source links.

**Limitations:** the data does not include prices, descriptions or amenities, because those exist only in the protected page bodies. Wasalt listing URLs carry no city, so city questions are answered from the area pages.

## Run locally

```bash
docker build -t property-bot .                 # add --build-arg RESCRAPE=1 to refresh the data
docker run -p 8000:8000 -e OPENROUTER_API_KEY=sk-or-... property-bot
# open http://localhost:8000
```

Without Docker:

```bash
pip install -r requirements.txt
python scraper/scrape.py
OPENROUTER_API_KEY=sk-or-... uvicorn app.main:app --port 8000
```

## Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | free key from https://openrouter.ai/keys |
| `OPENROUTER_MODELS` | no | comma-separated model list; the first is the primary, the rest are fallbacks |
| `PORT` | no | defaults to 8000 |

## Endpoints

- `GET /`: chat UI
- `POST /api/chat`: body is `{message, history}`; returns a server-sent events stream
- `GET /api/stats`: dataset summary
- `GET /health`: health check

## Deploy (Docker)

Any container host works. The image needs about 512 MB of RAM.

- **Render:** create a new Web Service from this repo with the Docker runtime and set `OPENROUTER_API_KEY`.
- **Fly.io:** run `fly launch --no-deploy`, then `fly secrets set OPENROUTER_API_KEY=...`, then `fly deploy`.
- **Hugging Face Spaces:** create a Docker Space, set `PORT=7860` as a variable, and add the key as a secret.
