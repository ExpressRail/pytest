"""
Loader and query helpers for the personal baseline meal dataset.
Used by the AI agent to retrieve reference values for differential analysis.
"""

import json
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "personal_meals.json"

with open(_DATA_PATH) as f:
    _DB = json.load(f)

MEALS = _DB["meals"]


def get_meal(key: str) -> dict:
    """Return a single baseline meal by key.

    Keys: 'my_classic_breakfast', 'my_classic_coffee', 'my_classic_dinner'
    """
    if key not in MEALS:
        raise KeyError(f"Unknown meal key '{key}'. Available: {list(MEALS.keys())}")
    return MEALS[key]


def get_all_meals() -> dict:
    """Return the full meals dictionary."""
    return MEALS


def meal_summary_text(key: str) -> str:
    """Return a compact human-readable summary string for prompt injection."""
    meal = get_meal(key)
    lines = [
        f"=== {meal['label']} ===",
        f"Description: {meal['description']}",
        "",
        "Items:",
    ]
    for item in meal["items"]:
        lines.append(
            f"  - {item['name']} ({item['quantity']} {item['unit']}): "
            f"{item['calories']} kcal | "
            f"P:{item['protein_g']}g  C:{item['carbs_g']}g  F:{item['fat_g']}g"
            + (f"  [{item['notes']}]" if item.get("notes") else "")
        )
    t = meal["totals"]
    lines += [
        "",
        f"TOTAL: {t['calories']} kcal  |  Protein: {t['protein_g']}g  "
        f"Carbs: {t['carbs_g']}g  Fat: {t['fat_g']}g",
    ]
    return "\n".join(lines)


def all_meals_summary_text() -> str:
    """Return summaries for all baseline meals concatenated."""
    return "\n\n".join(meal_summary_text(k) for k in MEALS)


def compute_differential(baseline_key: str, analyzed: dict) -> dict:
    """
    Compare an AI-analyzed meal dict against a named baseline meal.

    Parameters
    ----------
    baseline_key : str
        One of the MEALS keys (e.g. 'my_classic_breakfast').
    analyzed : dict
        A dict with at minimum a 'total_calories' int and optionally
        'protein_g', 'carbs_g', 'fat_g' totals — matching the structure
        returned by calorie_checker.analyze_food_image().

    Returns
    -------
    dict with 'baseline', 'analyzed', and 'delta' sub-dicts.
    """
    baseline_totals = get_meal(baseline_key)["totals"]

    def _delta(field):
        a = analyzed.get(field, 0) or 0
        b = baseline_totals.get(field, 0) or 0
        return round(a - b, 1)

    analyzed_calories = analyzed.get("total_calories", 0) or 0

    return {
        "baseline_label": get_meal(baseline_key)["label"],
        "baseline": baseline_totals,
        "analyzed": {
            "calories": analyzed_calories,
            "protein_g": analyzed.get("protein_g", None),
            "carbs_g": analyzed.get("carbs_g", None),
            "fat_g": analyzed.get("fat_g", None),
        },
        "delta": {
            "calories": round(analyzed_calories - baseline_totals["calories"], 1),
            "protein_g": _delta("protein_g"),
            "carbs_g": _delta("carbs_g"),
            "fat_g": _delta("fat_g"),
        },
        "summary": (
            f"The analyzed meal has "
            f"{abs(analyzed_calories - baseline_totals['calories'])} kcal "
            f"{'more' if analyzed_calories > baseline_totals['calories'] else 'fewer'} "
            f"than your classic baseline ({baseline_totals['calories']} kcal)."
        ),
    }


if __name__ == "__main__":
    # Quick smoke-test: print all meal summaries
    print(all_meals_summary_text())
