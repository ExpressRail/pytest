import base64
import io
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from PIL import Image

from calorie_checker import analyze_food_image

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if not file or not file.filename:
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG, PNG, GIF, or WEBP image."}), 400

    image_bytes = file.read()

    # Validate image integrity with Pillow
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Raises if corrupt
    except Exception:
        return jsonify({"error": "Could not read the image file. It may be corrupt or not a valid image."}), 400

    # Re-open after verify() (verify closes the internal stream)
    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or "JPEG"
    media_type = PILLOW_FORMAT_TO_MIME.get(fmt.upper(), "image/jpeg")

    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    result = analyze_food_image(image_b64, media_type)

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)


if __name__ == "__main__":
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    app.run(debug=True, port=5000)
