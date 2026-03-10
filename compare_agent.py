"""
Compare agent: agentic loop that analyzes a food photo and compares it against
one of the user's personal baseline meals from data/personal_meals.json.

Claude has two tools:
  - list_available_baselines()  — discovery / auto-select
  - lookup_meal_baseline(key)   — fetch detailed data for a specific meal

The runner loops until Claude produces a final text block containing JSON.
"""

import json
import os

import anthropic
from anthropic import beta_tool

from meal_data import MEALS, meal_summary_text

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

@beta_tool
def list_available_baselines() -> str:
    """List all personal homemade baseline meals available for comparison.
    Call this first to discover what baselines exist and pick the best match."""
    lines = ["Personal baseline meals available for comparison:"]
    for key, meal in MEALS.items():
        t = meal["totals"]
        lines.append(
            f"  key={key!r}  label={meal['label']!r}  "
            f"total={t['calories']} kcal  "
            f"(P:{t['protein_g']}g  C:{t['carbs_g']}g  F:{t['fat_g']}g)"
        )
    return "\n".join(lines)


@beta_tool
def lookup_meal_baseline(meal_key: str) -> str:
    """Fetch the full nutritional breakdown for one personal baseline meal.

    Args:
        meal_key: Exact key from list_available_baselines().
                  One of: 'my_classic_breakfast', 'my_classic_coffee',
                  'my_classic_dinner'.
    """
    if meal_key not in MEALS:
        return (
            f"Error: unknown key {meal_key!r}. "
            f"Valid keys: {list(MEALS.keys())}"
        )
    return meal_summary_text(meal_key)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a personal nutrition analyst helping a user understand how restaurant \
or prepared meals compare to their own homemade classics.

Your workflow for every request:
1. Call list_available_baselines() to see what personal meals exist.
2. Analyze the uploaded food photo to identify every visible item and \
   estimate calories.
3. If the user specified a baseline hint, use that key. Otherwise choose \
   the best matching baseline based on what you see in the photo.
4. Call lookup_meal_baseline(key) to retrieve the full nutritional profile.
5. Compute the calorie delta and macro differences.
6. Output ONLY a single valid JSON object — no prose, no markdown fences.

The final JSON must follow this exact schema:
{
  "analyzed_foods": [
    {
      "name": "string",
      "portion_estimate": "string",
      "calories": <integer>,
      "confidence": "high|medium|low"
    }
  ],
  "analyzed_total_calories": <integer>,
  "baseline_key": "string",
  "baseline_label": "string",
  "baseline_total_calories": <integer>,
  "calorie_delta": <integer, positive = analyzed has MORE calories>,
  "delta_direction": "more|fewer|same",
  "delta_percent": <integer, rounded>,
  "macro_deltas": {
    "protein_g": <number>,
    "carbs_g": <number>,
    "fat_g": <number>
  },
  "item_comparisons": [
    "One sentence comparing a specific item or portion difference."
  ],
  "narrative": "2-3 sentence plain-English summary of the comparison.",
  "verdict": "higher|lower|similar",
  "disclaimer": "string"
}

Rules:
- All calorie values are integers.
- delta_percent = round(abs(calorie_delta) / baseline_total_calories * 100).
- verdict is 'similar' if abs(delta_percent) <= 10, else 'higher' or 'lower'.
- item_comparisons should list 2-4 specific observations.
- Never include baseline_key values not returned by list_available_baselines().
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compare_to_baseline(
    image_b64: str,
    media_type: str,
    baseline_hint: str | None = None,
) -> dict:
    """
    Run the agentic comparison loop.

    Parameters
    ----------
    image_b64 : str   Base64-encoded image bytes.
    media_type : str  MIME type, e.g. 'image/jpeg'.
    baseline_hint : str | None
        If provided (e.g. 'my_classic_breakfast'), Claude is told to use
        that baseline. If None, Claude auto-selects based on the photo.

    Returns
    -------
    dict  Parsed JSON from the agent's final response, or {'error': ...}.
    """
    hint_text = (
        f"\n\nIMPORTANT: The user wants to compare against their "
        f"'{baseline_hint}' baseline specifically. Use that key."
        if baseline_hint
        else "\n\nAuto-select the most appropriate baseline for this meal."
    )

    user_text = (
        "Please analyze this food photo and compare it to my personal "
        "homemade baseline." + hint_text
    )

    try:
        runner = client.beta.messages.tool_runner(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=[list_available_baselines, lookup_meal_baseline],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
        )

        # Collect the final assistant message (tool runner loops until done)
        final_message = None
        for message in runner:
            final_message = message

        if final_message is None:
            return {"error": "Agent produced no response."}

        # Extract the last text block
        raw = ""
        for block in reversed(final_message.content):
            if hasattr(block, "text"):
                raw = block.text.strip()
                break

        if not raw:
            return {"error": "Agent returned no text content."}

        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]).strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"Could not parse agent response as JSON: {e}", "raw": raw}
    except anthropic.AuthenticationError:
        return {"error": "Invalid or missing ANTHROPIC_API_KEY."}
    except anthropic.APIError as e:
        return {"error": f"API error: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {e}"}
