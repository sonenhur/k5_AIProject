import logging
import os
import re

import cv2
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/received_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MODEL_STORAGE_DIRECTORY = "C:/workspace_project/AIproject/workspace/user_network_dir"
BLOCKED_CHARACTERS = "<>|]\:[+=`@#$%^&*;OD'"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_STORAGE_DIRECTORY, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def extract_text_from_image(image_path, languages=["ko"]):
    try:
        reader = easyocr.Reader(lang_list=languages, gpu=True)
        preprocessed_image = preprocess_image(image_path)
        results = reader.readtext(
            preprocessed_image,
            blocklist=BLOCKED_CHARACTERS,
            width_ths=15,
            ycenter_ths=0.5,
        )
        extracted_data = [
            {"text": result[1], "confidence": result[2]} for result in results
        ]
        return extracted_data
    except Exception as e:
        logging.error(f"Error occurred while extracting text: {e}")
        return [], None  # 오류 발생 시 빈 리스트와 None 반환


def parse_items(texts):
    try:
        item_pattern = re.compile(r"([\w가-힣]+/[\w가-힣]+/[A-Z]+)")
        price_pattern = re.compile(r"(\d{1,3}(?:,\d{3})*)")
        items = []

        for text in texts:
            item_match = item_pattern.search(text)
            price_matches = price_pattern.findall(text)
            if item_match and price_matches and len(price_matches) >= 3:
                item = item_match.group()
                unit_price = price_matches[0]
                quantity = price_matches[1]
                total_price = price_matches[2]
                items.append(
                    {
                        "item": item,
                        "unitPrice": int(unit_price.replace(",", "")),
                        "quantity": int(quantity),
                        "price": int(total_price.replace(",", "")),
                    }
                )
        return items
    except Exception as e:
        logging.error(f"Error occurred while parsing items: {e}")
        return []


def find_date_in_text(texts):
    try:
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for text in texts:
            date_match = date_pattern.search(text)
            if date_match:
                return date_match.group()
        return None
    except Exception as e:
        logging.error(f"Error occurred while finding date: {e}")
        return None


@app.route("/image/upload", methods=["POST"])
def get_photo_input():
    try:
        file = request.files.get("image")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            extracted_data = extract_text_from_image(save_path)
            if not extracted_data:
                return jsonify({"error": "No text found in image"}), 400

            date = find_date_in_text([entry["text"] for entry in extracted_data])
            items = parse_items([entry["text"] for entry in extracted_data])
            total_sum = sum(item["price"] for item in items)

            image_url = url_for("get_image", filename=filename, _external=True)
            return jsonify(
                {"tradeAt": date, "sum": total_sum, "items": items, "image": image_url}
            )

        return jsonify({"error": "Invalid file"}), 400
    except Exception as e:
        logging.error(f"Error occurred while processing image: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/images/<filename>")
def get_image(filename):
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        logging.error(f"Error occurred while retrieving image: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
