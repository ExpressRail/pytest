import base64
import io
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

from calorie_checker import analyze_food_image
from compare_agent import compare_to_baseline
from meal_data import MEALS

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "webp"}

PILLOW_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _decode_upload(file):
    """Validate and decode an uploaded image file.

    Returns (image_bytes, media_type) on success, or (None, error_message) on failure.
    """
    if not file or not file.filename:
        return None, "No file selected"
    if not allowed_file(file.filename):
        return None, "Unsupported file type. Please upload a JPG, PNG, GIF, or WEBP image."

    image_bytes = file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        return None, "Could not read the image file. It may be corrupt or not a valid image."

    # Re-open after verify() closes the stream
    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or "JPEG"
    media_type = PILLOW_FORMAT_TO_MIME.get(fmt.upper(), "image/jpeg")
    return image_bytes, media_type


@app.route("/")
def index():
    baselines = {key: meal["label"] for key, meal in MEALS.items()}
    return render_template("index.html", baselines=baselines)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes, result = _decode_upload(request.files["image"])
    if image_bytes is None:
        return jsonify({"error": result}), 400

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data = analyze_food_image(image_b64, result)

    if "error" in data:
        return jsonify(data), 500
    return jsonify(data)


@app.route("/compare", methods=["POST"])
def compare():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes, result = _decode_upload(request.files["image"])
    if image_bytes is None:
        return jsonify({"error": result}), 400

    # Optional baseline hint; "auto" means let the agent decide
    baseline_hint = request.form.get("baseline", "auto")
    if baseline_hint == "auto":
        baseline_hint = None
    elif baseline_hint not in MEALS:
        return jsonify({"error": f"Unknown baseline '{baseline_hint}'."}), 400

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    data = compare_to_baseline(image_b64, result, baseline_hint)

    if "error" in data:
        return jsonify(data), 500
    return jsonify(data)


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    app.run(debug=True, port=5000)
