# India Property Pricing Agent

A conversational AI agent that estimates residential property prices for **any city and locality across India**. Powered by LangGraph, GPT-4o, and live MagicBricks data via Apify MCP.

## How it works

```
User query → GPT-4o → MagicBricks scraper (live listings)
                    → property_adjustments (BHK, age, floor, amenities)
                    → price_estimator
                    → plain-English answer with ₹ range + PSF
```

The agent asks **at most one clarifying question** (only if city/locality/BHK/type are missing), then calls tools and returns a concrete price — never refuses.

## Features

- Works for any Indian city: Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Pune, Bhopal, and more
- Live data from MagicBricks via Apify MCP (`krazee_kaushik/magicbricks-search-results-scraper`)
- Conversational — handles follow-up questions in multi-turn chat
- Tier-based fallback PSF when listings are unavailable for a locality
- Structured adjustment factors: BHK size, property age, floor, parking, gym, pool, gated community, property type

## Prerequisites

- Python 3.11+
- [OpenAI API key](https://platform.openai.com/api-keys)
- [Apify API token](https://console.apify.com/account/integrations) (free plan works)

## Setup

```bash
git clone https://github.com/raghu640/ollive_assignment.git
cd ollive_assignment

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and APIFY_API_TOKEN
```

## Usage

**Interactive chat:**
```bash
python agent.py
```
```
You: 3BHK apartment in Bandra Mumbai
Agent: A 3BHK apartment in Bandra would cost ₹285–₹718 lakh (~₹45,000/sqft).
       Bandra commands premium pricing due to its coastal location and connectivity.
       Based on live MagicBricks listings.

You: what about 2BHK in Gachibowli Hyderabad
Agent: A 2BHK apartment in Gachibowli would cost ₹72–₹190 lakh (~₹10,500/sqft)...
```

**As a library:**
```python
from agent import price_property

print(price_property("2BHK apartment in Koramangala Bangalore"))
```

**With LangSmith tracing:**
```bash
LANGCHAIN_TRACING_V2=true LANGCHAIN_PROJECT=property-agent python agent.py
```

## Project structure

```
agent.py                  # LangGraph graph, system prompt, entry point
tools/
  __init__.py             # LOCAL_TOOLS list
  property_adjustments.py # Multiplier from BHK/age/floor/amenities
  price_estimator.py      # Final ₹ range from PSF + multiplier + area
data/
  adjustments.json        # Adjustment factor tables
tests/
  test_agent.py
  test_tools.py
  conftest.py
```

MCP tools (MagicBricks scraper) are loaded at startup from Apify's MCP server and bound directly to GPT-4o alongside the local tools.

## Running tests

```bash
pytest tests/ -v
pytest tests/ --cov=. --cov-report=term-missing   # with coverage
```

## Extending

**Add a new local tool:**
1. Create `tools/my_tool.py` with a single `@tool`-decorated function and a Google-style docstring
2. Import and append to `LOCAL_TOOLS` in `tools/__init__.py`

**Add LangSmith tracing:** set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env` — no code changes needed.

**Wrap as a FastAPI endpoint:** create `api.py` with `POST /estimate` calling `price_property(query)`, add `fastapi` and `uvicorn` to `requirements.txt`.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | GPT-4o inference |
| `APIFY_API_TOKEN` | Yes | MagicBricks scraper via Apify MCP |
| `LANGCHAIN_TRACING_V2` | No | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | No | LangSmith API key |
| `LANGCHAIN_PROJECT` | No | LangSmith project name |
# real_estate_agent
