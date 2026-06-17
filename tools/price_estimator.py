"""Local tool — no external calls required.

Combines PSF benchmarks and adjustment multiplier into a final
price range with midpoint PSF and confidence level.
"""
from langchain_core.tools import tool


def _confidence(psf_min: int, psf_max: int) -> str:
    spread_pct = (psf_max - psf_min) / psf_min * 100
    if spread_pct <= 20:
        return "high"
    if spread_pct <= 40:
        return "medium"
    return "low"


@tool
def price_estimator(
    psf_min: int,
    psf_max: int,
    area_sqft: float,
    multiplier: float,
) -> dict:
    """Compute the final price range for a property.

    Args:
        psf_min: Minimum locality PSF benchmark (INR per sq ft).
        psf_max: Maximum locality PSF benchmark (INR per sq ft).
        area_sqft: Built-up area of the property in square feet.
        multiplier: Composite adjustment multiplier from property_adjustments tool.

    Returns:
        dict with keys:
            price_min_lakh (float): lower bound in INR lakhs (1 lakh = 100,000)
            price_max_lakh (float): upper bound in INR lakhs
            midpoint_psf (int): adjusted midpoint PSF used for estimation
            confidence (str): "high" | "medium" | "low" — width of the range
    """
    try:
        adj_min = psf_min * multiplier
        adj_max = psf_max * multiplier
        price_min = adj_min * area_sqft / 100_000
        price_max = adj_max * area_sqft / 100_000
        midpoint_psf = int((adj_min + adj_max) / 2)

        return {
            "price_min_lakh": round(price_min, 2),
            "price_max_lakh": round(price_max, 2),
            "midpoint_psf": midpoint_psf,
            "confidence": _confidence(int(adj_min), int(adj_max)),
        }
    except Exception as exc:
        return {"error": str(exc), "data": []}
