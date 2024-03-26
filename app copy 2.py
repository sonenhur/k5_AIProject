import os
import re

import cv2
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/recieved_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def extract_text_from_image(image_path, languages=["en", "ko"]):
    reader = easyocr.Reader(languages)
    preprocessed_image = preprocess_image(image_path)
    result = reader.readtext(
        preprocessed_image, width_ths=10, ycenter_ths=0.5
    )  # width_ths 및 ycenter_ths 값 조정 가능

    print("OCR Result:", result)  # OCR 결과 출력

    items = []
    if result:  # 텍스트 추출 결과가 있는 경우
        flag = False
        for bbox, text, prob in result:
            print("Extracted Text:", text)  # 추출된 텍스트 출력
            if "금액" in text or "수량" in text or "단가" in text:
                flag = True
            if flag:
                items.append(text)

    return items


def parse_items(extracted_texts):
    parsed_items = []
    item_pattern = re.compile(r"([\w가-힣]+/ [\w가-힣]+/[A-Z])")
    price_pattern = re.compile(r"(\d{1,3},\d{3})")

    for text in extracted_texts:
        item_match = item_pattern.search(text)
        price_matches = price_pattern.findall(text)

        if item_match and price_matches:
            item = item_match.group()
            prices = price_matches
            if len(prices) >= 2:
                # 가정: 첫 번째가 단가, 마지막이 총액, 중간 값이 수량일 경우
                parsed_items.append(
                    {
                        "item": item,
                        "unitPrice": prices[0],
                        "quantity": (
                            "1" if len(prices) == 2 else prices[1]
                        ),  # 수량이 명시되지 않은 경우 1로 가정
                        "price": prices[-1],
                    }
                )

    return parsed_items


def find_by_date(image_path):
    reader = easyocr.Reader(["en", "ko"])
    result = reader.readtext(image_path, width_ths=0.5, ycenter_ths=0.5)

    pattern = r"(\d{4}-\d{2}-\d{2})"
    for bbox, text, prob in result:
        match = re.search(pattern, text)
        if match:
            return match.group()

    return None


@app.route("/image/upload", methods=["POST"])
def get_photo_input():
    file = request.files.get("image")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        items_text = extract_text_from_image(save_path)
        items = parse_items(items_text)
        date = find_by_date(save_path)

        image_url = url_for("get_image", filename=filename, _external=True)
        return jsonify({"date": date, "items": items, "image": image_url})

    return jsonify({"error": "Invalid file"}), 400


@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
