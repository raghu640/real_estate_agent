from tools.property_adjustments import property_adjustments
from tools.price_estimator import price_estimator, _confidence


def test_adjustments_returns_multiplier():
    result = property_adjustments.invoke({
        "bhk": 2, "age_years": 5, "floor": 3,
        "has_parking": True, "has_gym": False, "has_pool": False, "is_gated": True,
    })
    assert "error" not in result
    assert "multiplier" in result
    assert 0.5 < result["multiplier"] < 2.0


def test_new_construction_commands_premium():
    result_new = property_adjustments.invoke({
        "bhk": 2, "age_years": 0, "floor": 3,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    result_old = property_adjustments.invoke({
        "bhk": 2, "age_years": 25, "floor": 3,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    assert result_new["multiplier"] > result_old["multiplier"]


def test_amenities_increase_multiplier():
    result_bare = property_adjustments.invoke({
        "bhk": 2, "age_years": 5, "floor": 3,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    result_full = property_adjustments.invoke({
        "bhk": 2, "age_years": 5, "floor": 3,
        "has_parking": True, "has_gym": True, "has_pool": True, "is_gated": True,
    })
    assert result_full["multiplier"] > result_bare["multiplier"]


def test_adjustments_returns_breakdown():
    result = property_adjustments.invoke({
        "bhk": 3, "age_years": 10, "floor": 5,
        "has_parking": True, "has_gym": False, "has_pool": False, "is_gated": True,
    })
    assert "breakdown" in result
    assert isinstance(result["breakdown"], dict)


def test_multiplier_clamped_to_minimum():
    result = property_adjustments.invoke({
        "bhk": 1, "age_years": 40, "floor": 0,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    assert "error" not in result
    assert result["multiplier"] >= 0.5
    assert result["multiplier"] <= 2.0


def test_adjustments_high_floor_8_to_15():
    result = property_adjustments.invoke({
        "bhk": 2, "age_years": 5, "floor": 10,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    assert "error" not in result
    assert "multiplier" in result


def test_adjustments_high_floor_16_plus():
    result = property_adjustments.invoke({
        "bhk": 2, "age_years": 5, "floor": 20,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    assert "error" not in result
    assert "multiplier" in result


def test_adjustments_very_old_building():
    result = property_adjustments.invoke({
        "bhk": 2, "age_years": 60, "floor": 3,
        "has_parking": False, "has_gym": False, "has_pool": False, "is_gated": False,
    })
    assert "error" not in result
    assert result["multiplier"] >= 0.5


def test_price_estimator_returns_range():
    result = price_estimator.invoke({
        "psf_min": 9000, "psf_max": 14000, "area_sqft": 1200, "multiplier": 1.05,
    })
    assert "error" not in result
    assert "price_min_lakh" in result
    assert "price_max_lakh" in result
    assert result["price_max_lakh"] > result["price_min_lakh"]


def test_price_estimator_returns_midpoint_psf():
    result = price_estimator.invoke({
        "psf_min": 10000, "psf_max": 10000, "area_sqft": 1000, "multiplier": 1.0,
    })
    assert result["midpoint_psf"] == 10000
    assert abs(result["price_min_lakh"] - 100.0) < 0.1


def test_price_estimator_returns_confidence():
    result = price_estimator.invoke({
        "psf_min": 9000, "psf_max": 14000, "area_sqft": 1200, "multiplier": 1.05,
    })
    assert "confidence" in result
    assert result["confidence"] in ("low", "medium", "high")


def test_price_estimator_high_confidence():
    result = price_estimator.invoke({
        "psf_min": 12000, "psf_max": 12000, "area_sqft": 800, "multiplier": 1.0,
    })
    assert result["confidence"] == "high"
    assert result["price_min_lakh"] == result["price_max_lakh"]


def test_confidence_high():
    assert _confidence(10000, 11500) == "high"


def test_confidence_medium():
    assert _confidence(10000, 13000) == "medium"


def test_confidence_low():
    assert _confidence(10000, 16000) == "low"
