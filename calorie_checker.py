import os
import json
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a professional nutritionist and dietitian with deep expertise in food \
identification and calorie estimation. When shown a food image, you:
1. Identify each distinct food item visible in the image
2. Estimate the portion size based on visual cues (plate size, utensils, hands for scale, \
   standard container sizes, etc.)
3. Provide calorie estimates based on typical nutritional values; use typical single-serving \
   assumptions when visual cues are ambiguous
4. Always respond with valid JSON only — no prose, no markdown code fences, no extra text"""

USER_PROMPT = """Analyze this food image and return a JSON object with this exact structure:

{
  "foods": [
    {
      "name": "food item name",
      "portion_estimate": "e.g. 1 cup, 200g, 1 medium piece",
      "calories": 250,
      "confidence": "high",
      "notes": "any relevant notes about the estimate"
    }
  ],
  "total_calories": 250,
  "meal_type_guess": "breakfast",
  "overall_confidence": "high",
  "disclaimer": "brief note that these are estimates based on typical values"
}

Rules:
- "calories" must be an integer (whole number, never a range like "200-300")
- "total_calories" must equal the sum of all individual food item calories
- "confidence" and "overall_confidence" must be one of: "high", "medium", or "low"
- "meal_type_guess" must be one of: "breakfast", "lunch", "dinner", "snack", or "unknown"
- If you cannot identify any food at all, return an empty "foods" array and set "total_calories" to 0
- Estimate as if this is a typical single serving unless visual evidence suggests otherwise"""


def analyze_food_image(image_b64: str, media_type: str) -> dict:
    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
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
                        {
                            "type": "text",
                            "text": USER_PROMPT,
                        },
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()

        # Strip markdown fences if Claude adds them despite instructions
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first line (```json or ```) and last line (```)
            raw = "\n".join(lines[1:-1]).strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        return {"error": f"Could not parse AI response: {str(e)}", "raw": raw}
    except anthropic.AuthenticationError:
        return {"error": "Invalid or missing ANTHROPIC_API_KEY. Please check your .env file."}
    except anthropic.APIError as e:
        return {"error": f"API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
