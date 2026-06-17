"""Local tool — no external calls required.

Computes a composite price adjustment multiplier based on property
characteristics: BHK type, building age, floor level, and amenities.
"""
import json
import pathlib

from langchain_core.tools import tool

_DATA_PATH = pathlib.Path(__file__).parent.parent / "data" / "adjustments.json"
_ADJ: dict = json.loads(_DATA_PATH.read_text(encoding="utf-8"))


def _age_multiplier(age_years: int) -> float:
    table = _ADJ["age_years"]
    thresholds = sorted(int(k) for k in table)
    closest = min(thresholds, key=lambda t: abs(t - age_years))
    return table[str(closest)]


def _floor_multiplier(floor: int) -> float:
    table = _ADJ["floor"]
    if floor == 0:
        return table["ground"]
    if 1 <= floor <= 3:
        return table["1_3"]
    if 4 <= floor <= 7:
        return table["4_7"]
    if 8 <= floor <= 15:
        return table["8_15"]
    return table["16_plus"]


@tool
def property_adjustments(
    bhk: int,
    age_years: int,
    floor: int,
    has_parking: bool,
    has_gym: bool,
    has_pool: bool,
    is_gated: bool,
    property_type: str = "apartment",
) -> dict:
    """Compute a price adjustment multiplier for a property's characteristics.

    Args:
        bhk: Number of bedrooms (1–5). For plots/villas use 0 if not applicable.
        age_years: Age of the building in years (0 = under construction / new).
        floor: Floor number (0 = ground floor). Use 0 for independent houses/villas/plots.
        has_parking: Whether covered parking is included.
        has_gym: Whether a gym/fitness centre is available.
        has_pool: Whether a swimming pool is available.
        is_gated: Whether it is a gated community.
        property_type: Type of property — "apartment", "villa", "independent house",
                       "plot", "penthouse", or "studio". Default is "apartment".

    Returns:
        dict with keys:
            multiplier (float): composite adjustment multiplier to apply to base PSF
            breakdown (dict): per-factor contributions for transparency
    """
    try:
        prop_table = _ADJ["property_type"]
        prop_factor = prop_table.get(property_type.lower().strip(), 1.0)

        bhk_table = _ADJ["bhk"]
        bhk_key = str(min(max(bhk, 1), 5))
        bhk_factor = bhk_table.get(bhk_key, 1.0)

        age_factor = _age_multiplier(age_years)
        floor_factor = _floor_multiplier(floor)

        amenity_table = _ADJ["amenities"]
        amenity_bonus = 0.0
        amenity_detail: dict = {}
        if has_parking:
            amenity_bonus += amenity_table["parking"]
            amenity_detail["parking"] = amenity_table["parking"]
        if has_gym:
            amenity_bonus += amenity_table["gym"]
            amenity_detail["gym"] = amenity_table["gym"]
        if has_pool:
            amenity_bonus += amenity_table["pool"]
            amenity_detail["pool"] = amenity_table["pool"]
        if is_gated:
            amenity_bonus += amenity_table["gated"]
            amenity_detail["gated"] = amenity_table["gated"]

        multiplier = round(
            max(0.5, min(2.0, prop_factor * bhk_factor * age_factor * floor_factor * (1.0 + amenity_bonus))),
            4,
        )

        return {
            "multiplier": multiplier,
            "breakdown": {
                "property_type_factor": prop_factor,
                "bhk_factor": bhk_factor,
                "age_factor": age_factor,
                "floor_factor": floor_factor,
                "amenity_bonus": round(amenity_bonus, 4),
                "amenity_detail": amenity_detail,
            },
        }
    except Exception as exc:
        return {"error": str(exc), "data": []}
