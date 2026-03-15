"""
Fertilizer Recommendation Module
==================================
This module analyzes soil nutrient levels (N, P, K) and recommends
specific fertilizers to improve soil health.

HOW THE LOGIC WORKS:
--------------------
1. GENERAL ANALYSIS (recommend_fertilizer):
   - Compares your soil's N, P, K values against universal thresholds
   - Classifies each nutrient as "low", "adequate", or "high"
   - Suggests specific fertilizer products for any deficiency

2. CROP-SPECIFIC ANALYSIS (recommend_for_crop):
   - Each crop has an ideal N, P, K range (stored in CROP_NUTRIENT_REQUIREMENTS)
   - Compares YOUR soil values against THAT CROP's optimal range
   - Calculates the exact deficit or surplus for each nutrient
   - Recommends fertilizers and dosage direction to match the crop's needs

WHY THIS MATTERS:
-----------------
- Too little of a nutrient --> poor crop growth and low yield
- Too much of a nutrient  --> wasted money, soil damage, water pollution
- The goal is to match soil nutrients to what the specific crop needs
"""


# ─── Crop Nutrient Requirements ──────────────────────────────────────────────────
# Each crop has an optimal range for N, P, K derived from agricultural data.
# Format: "crop_name": {"N": (min, max), "P": (min, max), "K": (min, max)}
#
# These ranges represent the ideal soil nutrient levels (in kg/ha) for each crop.
# Values outside these ranges indicate a deficiency or excess.

CROP_NUTRIENT_REQUIREMENTS = {
    "rice":        {"N": (60, 100),  "P": (35, 60),  "K": (35, 50)},
    "maize":       {"N": (60, 100),  "P": (35, 60),  "K": (15, 30)},
    "chickpea":    {"N": (15, 45),   "P": (55, 80),  "K": (70, 85)},
    "kidneybeans": {"N": (15, 30),   "P": (55, 75),  "K": (15, 25)},
    "pigeonpeas":  {"N": (15, 30),   "P": (55, 75),  "K": (15, 25)},
    "mothbeans":   {"N": (15, 35),   "P": (40, 65),  "K": (15, 25)},
    "mungbean":    {"N": (15, 35),   "P": (40, 65),  "K": (15, 25)},
    "blackgram":   {"N": (25, 50),   "P": (55, 75),  "K": (15, 25)},
    "lentil":      {"N": (15, 30),   "P": (55, 75),  "K": (15, 25)},
    "pomegranate": {"N": (15, 35),   "P": (5, 20),   "K": (30, 45)},
    "banana":      {"N": (80, 120),  "P": (70, 85),  "K": (45, 55)},
    "mango":       {"N": (15, 35),   "P": (15, 35),  "K": (25, 40)},
    "grapes":      {"N": (15, 35),   "P": (120, 145),"K": (195, 205)},
    "watermelon":  {"N": (80, 110),  "P": (15, 30),  "K": (45, 55)},
    "muskmelon":   {"N": (80, 110),  "P": (15, 25),  "K": (45, 55)},
    "apple":       {"N": (15, 35),   "P": (120, 140),"K": (195, 205)},
    "orange":      {"N": (15, 30),   "P": (5, 20),   "K": (5, 15)},
    "papaya":      {"N": (35, 65),   "P": (45, 65),  "K": (45, 55)},
    "coconut":     {"N": (15, 30),   "P": (5, 15),   "K": (25, 40)},
    "cotton":      {"N": (100, 140), "P": (40, 55),  "K": (15, 25)},
    "jute":        {"N": (60, 100),  "P": (35, 55),  "K": (35, 45)},
    "coffee":      {"N": (80, 120),  "P": (15, 30),  "K": (25, 40)},
}


# ─── Fertilizer Database ────────────────────────────────────────────────────────
# Maps each nutrient deficiency to specific fertilizer products.
# Each fertilizer has:
#   name        -- commercial/common name
#   nutrient    -- which nutrient it primarily provides
#   description -- what it does and when to use it

FERTILIZER_DATABASE = {
    "low_N": [
        {
            "name": "Urea",
            "nutrient": "Nitrogen (46% N)",
            "description": "Most concentrated solid nitrogen fertilizer. "
                           "Apply in split doses -- half at sowing, half after 3-4 weeks.",
        },
        {
            "name": "Ammonium Sulphate",
            "nutrient": "Nitrogen (21% N) + Sulphur (24% S)",
            "description": "Good for crops that also need sulphur (e.g. mustard, pulses). "
                           "Works well in alkaline soils.",
        },
        {
            "name": "DAP (Di-Ammonium Phosphate)",
            "nutrient": "Nitrogen (18% N) + Phosphorus (46% P2O5)",
            "description": "Provides both N and P. Best applied at sowing time.",
        },
    ],
    "low_P": [
        {
            "name": "Single Super Phosphate (SSP)",
            "nutrient": "Phosphorus (16% P2O5) + Sulphur (12% S)",
            "description": "Affordable phosphorus source. Also provides sulphur and calcium.",
        },
        {
            "name": "DAP (Di-Ammonium Phosphate)",
            "nutrient": "Phosphorus (46% P2O5) + Nitrogen (18% N)",
            "description": "High phosphorus content. Apply at sowing for root development.",
        },
        {
            "name": "Rock Phosphate",
            "nutrient": "Phosphorus (20-30% P2O5)",
            "description": "Slow-release, organic option. Works best in acidic soils (pH < 5.5).",
        },
    ],
    "low_K": [
        {
            "name": "Muriate of Potash (MOP)",
            "nutrient": "Potassium (60% K2O)",
            "description": "Most common potash fertilizer. Avoid for salt-sensitive crops like tobacco.",
        },
        {
            "name": "Sulphate of Potash (SOP)",
            "nutrient": "Potassium (50% K2O) + Sulphur (18% S)",
            "description": "Premium potash source. Ideal for fruits, vegetables, and salt-sensitive crops.",
        },
    ],
    "high_N": [
        {
            "name": "Crop Rotation with Legumes",
            "nutrient": "N/A",
            "description": "Legumes (beans, peas, lentils) fix nitrogen from air. "
                           "Growing them naturally reduces excess soil nitrogen.",
        },
    ],
    "high_P": [
        {
            "name": "Reduce Phosphorus Input",
            "nutrient": "N/A",
            "description": "Stop phosphorus fertilizers. Excess P can run off into water "
                           "bodies causing algal blooms. Grow P-hungry crops to draw it down.",
        },
    ],
    "high_K": [
        {
            "name": "Reduce Potash Input",
            "nutrient": "N/A",
            "description": "Stop potassium fertilizers. Excess K can block magnesium and "
                           "calcium absorption by plants.",
        },
    ],
}


# ─── General Fertilizer Recommendation ──────────────────────────────────────────

def recommend_fertilizer(nitrogen, phosphorus, potassium):
    """
    Analyze soil N, P, K and recommend fertilizers using universal thresholds.

    Logic:
      - Compare each nutrient against fixed thresholds:
          N:  low < 50,  adequate 50-100,  high > 100
          P:  low < 30,  adequate 30-80,   high > 80
          K:  low < 30,  adequate 30-80,   high > 80
      - For any nutrient that is low or high, suggest matching fertilizers
        from the FERTILIZER_DATABASE.

    Parameters:
        nitrogen    (float): Soil nitrogen level in kg/ha
        phosphorus  (float): Soil phosphorus level in kg/ha
        potassium   (float): Soil potassium level in kg/ha

    Returns:
        dict with nutrient_status, recommendations, and suggested_fertilizers
    """
    nutrient_status = {}
    recommendations = []
    suggested_fertilizers = []

    # ── Nitrogen check ──
    # Nitrogen drives leaf growth. Too little = stunted plants, yellow leaves.
    # Too much = excessive foliage but weak stems and fewer fruits.
    if nitrogen < 50:
        nutrient_status["nitrogen"] = "low"
        recommendations.append("Nitrogen is low. Apply nitrogen-rich fertilizers to support leaf growth.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["low_N"])
    elif nitrogen > 100:
        nutrient_status["nitrogen"] = "high"
        recommendations.append("Nitrogen is high. Reduce nitrogen input to avoid weak stems and excess foliage.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["high_N"])
    else:
        nutrient_status["nitrogen"] = "adequate"

    # ── Phosphorus check ──
    # Phosphorus supports root development and flowering.
    # Too little = poor roots, delayed maturity. Too much = water pollution risk.
    if phosphorus < 30:
        nutrient_status["phosphorus"] = "low"
        recommendations.append("Phosphorus is low. Apply phosphorus fertilizers for stronger roots and flowering.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["low_P"])
    elif phosphorus > 80:
        nutrient_status["phosphorus"] = "high"
        recommendations.append("Phosphorus is high. Stop P fertilizers to prevent environmental damage.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["high_P"])
    else:
        nutrient_status["phosphorus"] = "adequate"

    # ── Potassium check ──
    # Potassium regulates water use and disease resistance.
    # Too little = weak plants, poor drought tolerance. Too much = nutrient lockout.
    if potassium < 30:
        nutrient_status["potassium"] = "low"
        recommendations.append("Potassium is low. Apply potash fertilizers for better water regulation and disease resistance.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["low_K"])
    elif potassium > 80:
        nutrient_status["potassium"] = "high"
        recommendations.append("Potassium is high. Stop potash input to avoid blocking other nutrients.")
        suggested_fertilizers.extend(FERTILIZER_DATABASE["high_K"])
    else:
        nutrient_status["potassium"] = "adequate"

    # If all nutrients are adequate, suggest a balanced maintenance fertilizer
    if not recommendations:
        recommendations.append(
            "All nutrient levels are balanced. Use a general NPK (10-10-10) fertilizer for maintenance."
        )

    return {
        "nutrient_status": nutrient_status,
        "recommendations": recommendations,
        "suggested_fertilizers": suggested_fertilizers,
    }


# ─── Crop-Specific Fertilizer Recommendation ────────────────────────────────────

def recommend_for_crop(nitrogen, phosphorus, potassium, crop_name):
    """
    Compare soil nutrients against the optimal range for a specific crop.

    Logic:
      1. Look up the crop's ideal N, P, K ranges from CROP_NUTRIENT_REQUIREMENTS
      2. For each nutrient, check if your soil value falls:
         - BELOW the crop's minimum --> deficit (need to add fertilizer)
         - WITHIN the range         --> optimal (no action needed)
         - ABOVE the crop's maximum --> surplus (reduce or skip that nutrient)
      3. Calculate the exact gap (how much below/above the ideal range)
      4. Suggest fertilizers to fill any deficits

    Parameters:
        nitrogen    (float): Soil nitrogen level in kg/ha
        phosphorus  (float): Soil phosphorus level in kg/ha
        potassium   (float): Soil potassium level in kg/ha
        crop_name   (str):   Name of the crop (must match a key in CROP_NUTRIENT_REQUIREMENTS)

    Returns:
        dict with crop, nutrient_analysis (per-nutrient breakdown), and fertilizer_plan
    """
    crop_name = crop_name.lower().strip()

    # Check if we have data for this crop
    if crop_name not in CROP_NUTRIENT_REQUIREMENTS:
        available = sorted(CROP_NUTRIENT_REQUIREMENTS.keys())
        return {
            "error": f"Crop '{crop_name}' not found.",
            "available_crops": available,
        }

    # Get the ideal ranges for this crop
    requirements = CROP_NUTRIENT_REQUIREMENTS[crop_name]

    # Analyze each nutrient against the crop's specific range
    #   soil_value:  what the farmer's soil currently has
    #   ideal_range: what this crop needs (min, max)
    #   status:      "deficit" / "optimal" / "surplus"
    #   gap:         how far off from the ideal range (0 if optimal)

    nutrients = {
        "N": {"value": nitrogen,   "ideal": requirements["N"]},
        "P": {"value": phosphorus, "ideal": requirements["P"]},
        "K": {"value": potassium,  "ideal": requirements["K"]},
    }

    analysis = {}
    fertilizer_plan = []

    for nutrient_name, info in nutrients.items():
        soil_value = info["value"]
        ideal_min, ideal_max = info["ideal"]

        if soil_value < ideal_min:
            # Deficit: soil has less than the crop needs
            gap = ideal_min - soil_value
            status = "deficit"
            # Look up fertilizers for this deficiency
            fert_key = f"low_{nutrient_name}"
            fertilizers = FERTILIZER_DATABASE.get(fert_key, [])
            fertilizer_plan.append({
                "nutrient": nutrient_name,
                "action": f"Increase {nutrient_name} by ~{gap:.0f} kg/ha",
                "fertilizers": fertilizers,
            })
        elif soil_value > ideal_max:
            # Surplus: soil has more than the crop needs
            gap = soil_value - ideal_max
            status = "surplus"
            fert_key = f"high_{nutrient_name}"
            fertilizers = FERTILIZER_DATABASE.get(fert_key, [])
            fertilizer_plan.append({
                "nutrient": nutrient_name,
                "action": f"Reduce {nutrient_name} (surplus of ~{gap:.0f} kg/ha)",
                "fertilizers": fertilizers,
            })
        else:
            # Optimal: soil nutrient is within the crop's ideal range
            gap = 0
            status = "optimal"

        analysis[nutrient_name] = {
            "soil_value": soil_value,
            "ideal_range": f"{ideal_min}-{ideal_max}",
            "status": status,
            "gap": gap,
        }

    return {
        "crop": crop_name,
        "nutrient_analysis": analysis,
        "fertilizer_plan": fertilizer_plan,
    }
