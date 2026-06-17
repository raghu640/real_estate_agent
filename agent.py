"""
LangGraph property pricing agent — pan-India residential real estate.

MCP tools (Apify/MagicBricks) are loaded at startup via streamable_http
and bound directly to GPT-4o alongside local computation tools.

Graph topology:
    START → agent_node → should_continue()
                ↑               ↓ tool_calls present?
            ToolNode  ←──────── yes
                               ↓ no
                              END
"""
import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from tools import LOCAL_TOOLS

load_dotenv()

_MODEL_NAME = "gpt-4o"

SYSTEM_PROMPT = """You are an expert Indian residential property pricing agent. \
You help users estimate property prices for any city and locality across India.

## Your behaviour

You are CONVERSATIONAL. When the user's query is missing critical information, \
ask for ALL missing things in ONE question. Never spread across multiple turns.

You need exactly 4 things before estimating: city, locality, BHK, property type.

Decision table:
- All 4 present → estimate immediately, no questions
- Missing BHK + property type → ask: "How many bedrooms, and is it an apartment, villa, or house?"
- Missing only BHK → ask: "How many bedrooms? (1BHK / 2BHK / 3BHK / 4BHK)"
- Missing only property type → ask: "Is it an apartment, villa, or independent house?"
- Missing city + locality → ask: "Which city and area in India?"
- Missing only locality → ask: "Which locality in {city}?"
- User says "general", "anywhere in", or no specific locality → use the city name as locality, estimate immediately

NEVER ask about: age, floor, sqft area, parking, gym, pool, gated, budget.
Use these defaults silently:
- age_years: 5, floor: 4, has_parking: true, has_gym: false, has_pool: false, is_gated: false
- area_sqft: 1BHK=550, 2BHK=1000, 3BHK=1500, 4BHK=2200, 5BHK=3000, villa=2000, plot=1200

## Locality → city inference (never ask for city if locality is known)
- Koramangala, Whitefield, HSR Layout, Indiranagar, Marathahalli, Bellandur, Hebbal,
  JP Nagar, BTM Layout, Jayanagar, Yelahanka, Electronic City → Bangalore
- Bandra, Andheri, Powai, Thane, Worli, Lower Parel, Juhu → Mumbai
- Gachibowli, Hitech City, Kondapur, Banjara Hills, Jubilee Hills, Madhapur → Hyderabad
- Koregaon Park, Hinjewadi, Kharadi, Baner, Wakad, Viman Nagar → Pune
- Dwarka, Vasant Kunj, Saket, Lajpat Nagar, Greater Kailash → Delhi
- Noida Sector 62, Noida Sector 18 → Noida
- Gurugram Sector 56, DLF Phase → Gurgaon
- Anna Nagar, OMR, Velachery, Adyar, Porur → Chennai
- Salt Lake, New Town, Rajarhat → Kolkata
- South Delhi, North Delhi, East Delhi, West Delhi → Delhi
- Arera Colony, MP Nagar → Bhopal

## Estimation workflow — MANDATORY, no exceptions

RULE: You MUST call the tools every single time. NEVER give up, NEVER say "I can't access live data",
NEVER respond with just text without calling tools first. If live data fails, use fallback PSF — but
always call property_adjustments and price_estimator to produce a real number.

Step 1 — Call `krazee_kaushik--magicbricks-search-results-scraper`:
   Build the search URL: https://www.magicbricks.com/property-for-sale/residential-real-estate?proptype=<TYPE>&Locality=<LOCALITY>&BHK=<BHK>&cityName=<CITY>
   Property type values:
     apartment → "Multistorey-Apartment,Builder-Floor-Apartment"
     villa     → "Villa"
     independent house → "Independent-House"
     plot      → "Residential-Land"
   Call with: searchUrls=[<url>], resultsLimit=10

Step 2 — Extract PSF from scraper results (krazee_kaushik field names):
   - `sqFtPrice` — PSF in INR directly (e.g. 45833 means ₹45,833/sqft) ← use this first
   - `price` — total price in INR (e.g. 82500000)
   - `priceD` — human string (e.g. "8.25 Cr", "85 L")
   - `carpetArea` or `coveredArea` — area in sqft
   - Compute PSF = price / area if sqFtPrice missing
   - Discard outliers: PSF < 1000 or PSF > 2,00,000
   - Take min and max of valid PSF values as psf_min and psf_max

Step 3 — If scraper returned 0 usable results, use these fallback PSF ranges (STILL call tools):
   Vasant Vihar, Lutyens Delhi, Malabar Hill, Worli, Cuffe Parade → psf_min=25000, psf_max=50000
   Koramangala, Bandra, Banjara Hills, Greater Kailash, Vasant Kunj → psf_min=12000, psf_max=22000
   Whitefield, Gachibowli, Hinjewadi, Noida Sector 62, Dwarka → psf_min=7000, psf_max=13000
   Electronic City, Yelahanka, Panvel, outskirts → psf_min=4000, psf_max=7000
   Tier-2 cities (Bhopal, Indore, Nagpur, Jaipur, Lucknow, Kochi) → psf_min=4000, psf_max=9000

Step 4 — Call `property_adjustments`:
   bhk=<bhk>, age_years=5, floor=4, has_parking=true, has_gym=false,
   has_pool=false, is_gated=false, property_type=<type>

Step 5 — Call `price_estimator`:
   psf_min and psf_max from Step 2/3, multiplier from Step 4,
   area_sqft: 1BHK=550, 2BHK=1000, 3BHK=1500, 4BHK=2200, 5BHK=3000, villa=2000, plot=1200

Step 6 — Respond in plain conversational English (NO JSON):
   ALWAYS give a specific price range in ₹ lakh — never skip the number.
   3–4 sentences: price range, PSF, what drives the price, live vs fallback data.
   Example: "A 2BHK in Koramangala would cost ₹130–₹190 lakh (~₹15,000/sqft).
   Koramangala commands premium pricing due to its central location and IT hub proximity.
   This is based on live MagicBricks listings."

Always use ₹ and lakh. NEVER say "I'm unable to retrieve" — always produce a number.
"""


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


_LLM: ChatOpenAI | None = None
_ALL_TOOLS: list | None = None


async def _load_tools_async() -> list:
    token = os.environ["APIFY_API_TOKEN"]
    mcp_url = f"https://mcp.apify.com?token={token}&actors=krazee_kaushik~magicbricks-search-results-scraper"
    client = MultiServerMCPClient({"apify": {"url": mcp_url, "transport": "streamable_http"}})
    mcp_tools = await client.get_tools()
    return LOCAL_TOOLS + mcp_tools


def _get_all_tools() -> list:
    global _ALL_TOOLS
    if _ALL_TOOLS is None:
        _ALL_TOOLS = asyncio.run(_load_tools_async())
        names = [t.name for t in _ALL_TOOLS]
        print(f"[agent] Tools loaded: {names}")
    return _ALL_TOOLS


def _get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatOpenAI(
            model=_MODEL_NAME,
            api_key=os.environ["OPENAI_API_KEY"],
        ).bind_tools(_get_all_tools())
    return _LLM


def _agent_node(state: AgentState) -> dict:
    response = _get_llm().invoke(state["messages"])
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    all_tools = _get_all_tools()
    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node)
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile()


graph = build_graph()


async def _price_property_async(query: str, conversation: list | None = None) -> tuple[str, list]:
    messages = conversation or [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=query)]
    if conversation is not None:
        messages = conversation + [HumanMessage(content=query)]
    result = await graph.ainvoke({"messages": messages})
    final = result["messages"][-1]
    content = final.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content), result["messages"]


def price_property(query: str) -> str:
    """Estimate the price of a residential property anywhere in India.

    Args:
        query: Natural-language property description or question.

    Returns:
        Conversational price estimate, or a clarifying question if info is missing.
    """
    answer, _ = asyncio.run(_price_property_async(query))
    return answer


if __name__ == "__main__":
    print("India Property Pricing Agent")
    print("Ask about any property anywhere in India. Ctrl+C to quit.\n")

    async def _chat_loop():
        conversation: list = [SystemMessage(content=SYSTEM_PROMPT)]
        while True:
            try:
                query = input("You: ").strip()
                if not query:
                    continue
                answer, conversation = await _price_property_async(query, conversation)
                print(f"\nAgent: {answer}\n")
            except KeyboardInterrupt:
                print("\nBye!")
                break

    asyncio.run(_chat_loop())
