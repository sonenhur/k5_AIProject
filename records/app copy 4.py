import os
import re
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/recieved_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MODEL_STORAGE_DIRECTORY = "C:/workspace_project/AIproject/workspace/user_network_dir"
BLOCKED_CHARACTERS = "<>|]\-:[+=`@#$%^&*;'칭"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_STORAGE_DIRECTORY, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def rotate_image(image_path, angle=0):
    # 이미지를 그레이스케일로 읽기
    image = cv2.imread(image_path)

    # 이미지의 중심 좌표를 구함
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # 회전을 위한 변환 행렬을 구함
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 변환 행렬을 사용하여 이미지 회전
    rotated_image = cv2.warpAffine(image, M, (w, h))

    return rotated_image


def preprocess_image(image_path):
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Increase contrast if needed (can be uncommented if required)
    # image = cv2.equalizeHist(image)

    # Noise reduction with Gaussian blur
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 이미지 회전 적용
    image = rotate_image(image_path, angle=0)  # 우측으로 1도 회전을 위해 -1 사용

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def extract_text_from_image(
    image_path, filename, use_custom_model=False, languages=["ko"]
):
    model_storage_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)
    user_network_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)

    reader = (
        easyocr.Reader(
            lang_list=languages,
            gpu=True,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory,
            recog_network="custom",
        )
        if use_custom_model
        else easyocr.Reader(lang_list=languages, gpu=True)
    )

    # Apply the preprocessing to the image
    preprocessed_image = preprocess_image(image_path)

    # OCR on the preprocessed image
    result = reader.readtext(
        preprocessed_image,
        blocklist=BLOCKED_CHARACTERS,
        width_ths=0.7,
        ycenter_ths=0.7,
    )

    extracted_data = []
    loaded_image = cv2.imread(image_path)
    for bbox, text, confidence in result:
        extracted_data.append({"text": text, "confidence": confidence})
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(loaded_image, top_left, bottom_right, (255, 0, 0), 2)

    # Save the preprocessed image if you want to see it or for debugging
    preprocessed_image_path = image_path.replace(".jpg", "_preprocessed.jpg")
    cv2.imwrite(preprocessed_image_path, preprocessed_image)

    result_image_filename = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_image_filename, loaded_image)

    return extracted_data, result_image_filename


def filter_texts(texts):
    # 텍스트 필터링 로직
    start_index = 0
    end_index = len(texts)
    for i, text in enumerate(texts):
        if (
            "금액" in text
            or "가격" in text
            or "얼마" in text
            or "총얼마" in text
            or "PRICE" in text
            or "AMOUNT" in text
        ):
            start_index = i + 1
            break
    for i, text in enumerate(texts):
        if (
            "판매" in text
            or "판매소계" in text
            or "신한" in text
            or "국민" in text
            or "수협" in text
            or "하나" in text
            or "농협" in text
            or "기업" in text
            or "은행" in text
            or "신한은행" in text
            or "국민은행" in text
            or "기업은행" in text
            or "하나은행" in text
        ):
            end_index = i
            break
    return texts[start_index:end_index]


def find_date_in_text(texts):
    try:
        date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
        for text in texts:
            date_match = date_pattern.search(text)
            if date_match:
                return date_match.group()
        return "0000-00-00"
    except Exception as e:
        logging.error(f"Error occurred while finding date: {e}")
        return None


def group_texts(texts, n=4):
    """n개씩 그룹화된 텍스트 리스트를 반환합니다."""
    return [texts[i : i + n] for i in range(0, len(texts), n)]


@app.route("/image/upload", methods=["POST"])
def get_photo_input():
    file = request.files.get("image")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        extracted_data, result_image_filename = extract_text_from_image(
            save_path, filename
        )
        if not extracted_data:
            return jsonify({"error": "No text found in image"}), 400

        texts = [entry["text"] for entry in extracted_data]

        # 날짜 추출
        date = find_date_in_text([entry["text"] for entry in extracted_data])

        # 필터링과 그룹화
        filtered_texts = filter_texts(texts)
        grouped_texts = group_texts(filtered_texts, 4)  # 4개씩 그룹화

        image_url = url_for("get_image", filename=filename, _external=True)
        return jsonify({"tradeAt": date, "items": grouped_texts, "image": image_url})

    return jsonify({"error": "Invalid file"}), 400


@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Server running on IP: {ip_address}")
    app.run(host="0.0.0.0", port=5000, debug=True)
