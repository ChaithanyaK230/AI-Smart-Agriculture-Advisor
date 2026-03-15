"""Rule-based fertilizer recommendation based on soil nutrient levels."""


def recommend_fertilizer(nitrogen, phosphorus, potassium):
    """
    Recommend fertilizer based on N, P, K levels.

    Thresholds are simplified for learning purposes.
    Returns a dict with status of each nutrient and a recommendation.
    """
    recommendations = []
    nutrient_status = {}

    # Nitrogen assessment
    if nitrogen < 50:
        nutrient_status["nitrogen"] = "low"
        recommendations.append("Apply Urea or Ammonium Sulphate to boost nitrogen.")
    elif nitrogen > 100:
        nutrient_status["nitrogen"] = "high"
        recommendations.append("Reduce nitrogen-rich fertilizers. Consider crop rotation with legumes.")
    else:
        nutrient_status["nitrogen"] = "adequate"

    # Phosphorus assessment
    if phosphorus < 30:
        nutrient_status["phosphorus"] = "low"
        recommendations.append("Apply Single Super Phosphate (SSP) or DAP for phosphorus.")
    elif phosphorus > 80:
        nutrient_status["phosphorus"] = "high"
        recommendations.append("Reduce phosphorus input. Excess can harm water systems.")
    else:
        nutrient_status["phosphorus"] = "adequate"

    # Potassium assessment
    if potassium < 30:
        nutrient_status["potassium"] = "low"
        recommendations.append("Apply Muriate of Potash (MOP) to increase potassium.")
    elif potassium > 80:
        nutrient_status["potassium"] = "high"
        recommendations.append("Potassium levels are high. No additional potash needed.")
    else:
        nutrient_status["potassium"] = "adequate"

    if not recommendations:
        recommendations.append("Soil nutrient levels are balanced. Use a general NPK fertilizer for maintenance.")

    return {
        "nutrient_status": nutrient_status,
        "recommendations": recommendations,
    }
