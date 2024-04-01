import os
import re
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

# 설정 변수들
UPLOAD_FOLDER = "/workspace_project/AIproject/workspace/received_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MODEL_STORAGE_DIRECTORY = "/workspace_project/AIproject/workspace/user_network_dir"
BLOCKED_CHARACTERS = "<>|]\[+=`@#$%^&;'?"

# 폴더가 없는 경우 생성
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_STORAGE_DIRECTORY, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# 허용되는 파일 형식인지 확인
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 이미지를 그레이스케일로 전처리
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# 이미지에서 텍스트 추출
def extract_text_from_image(
    image_path, filename, use_custom_model=False, languages=["ko"]
):
    model_storage_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)
    user_network_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)

    if use_custom_model:
        reader = easyocr.Reader(
            lang_list=languages,
            gpu=True,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory,
            recog_network="custom",
        )
    else:
        reader = easyocr.Reader(lang_list=["ko", "en"], gpu=True)

    preprocessed_image = preprocess_image(image_path)

    result = reader.readtext(
        preprocessed_image, blocklist=BLOCKED_CHARACTERS, width_ths=15, ycenter_ths=0.5
    )

    extracted_data = []
    loaded_image = cv2.imread(image_path)
    for bbox, text, confidence in result:
        extracted_data.append({"text": text, "confidence": confidence})
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(loaded_image, top_left, bottom_right, (255, 0, 0), 2)

    result_image_filename = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_image_filename, loaded_image)

    return extracted_data, result_image_filename


# 날짜 추출
def extract_date(texts):
    for text in texts:
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        if match:
            return match.group()
    return None


# OCR 실수 교정
def correct_ocr_mistakes(text):
    corrections = {
        "I": "1",
        "l": "1",
        "O": "0",
        "o": "0",
        " ": "",
        ",": "",
        ".": ".",
        "_": "000",
    }
    corrected_text = "".join(corrections.get(char, char) for char in text)
    return corrected_text


# 숫자만 추출
def extract_numbers(text):
    corrected_text = correct_ocr_mistakes(text)
    matches = re.findall(r"[\d]+(?:\.\d+)?", corrected_text)
    return "".join(matches)


# 아이템 라인 처리
def process_item_line(line):
    # 여러 공백을 하나의 공백으로 처리하기 위해 정규식 사용
    parts = re.split(r"\s+", line.strip())

    item_data = {"item": "", "unitPrice": "", "quantity": "", "price"}

    if len(parts) >= 4:
        # 상품명, 단가, 수량, 금액 정보가 한 줄에 모두 있을 경우
        item_data["item"] = parts[0]
        item_data["unitPrice"] = extract_numbers(parts[1])
        item_data["quantity"] = extract_numbers(parts[2])
        item_data["price"] = extract_numbers(parts[3])
    elif len(parts) > 1:
        # 정보가 여러 줄에 걸쳐 있을 경우
        # 예를 들어, 상품명만 있는 줄 다음에 가격 정보가 오는 경우
        # 이 경우 더 복잡한 로직을 사용해야 할 수 있으며,
        # 여기서는 단순화를 위해 첫 번째 항목을 상품명으로 간주합니다.
        item_data["item"] = parts[0]
        # 나머지 정보는 이후의 행에서 처리해야 할 수 있습니다.

    return item_data


# 아이템 추출
def extract_items(texts):
    items_section = []
    capture = False

    # 시작 및 종료 키워드를 확인합니다.
    start_keywords = [
        "청구",
        "온라인",
        "수량",
        "금액",
        "단가금액",
        "ITEM",
        "Item",
        "PCS",
        "Pcs",
        "PRICE",
        "Price",
        "AMOUNT",
        "quantity",
        "price",
        "얼마",
        "총얼마",
        "더하기",
    ]

    end_keywords = [
        "판매",
        "판매소계",
        "반품소계",
        "부가세",
        "국민",
        "국민은행",
        "신한",
        "신한은행",
        "우리",
        "우리은행",
        "수협",
        "국민은행",
        "TOTAL",
    ]

    for text in texts:
        # 시작 키워드가 있는 라인을 찾으면, 다음 라인부터 capture을 시작합니다.
        if any(keyword in text for keyword in start_keywords):
            capture = True
            continue  # 키워드가 포함된 라인은 건너뜁니다.

        # 종료 키워드가 있는 라인을 찾으면, capture을 종료합니다.
        if any(keyword in text for keyword in end_keywords):
            break

        # capture 상태일 때, 항목을 추출합니다.
        if capture:
            items_section.append(process_item_line(text))
    return items_section


# 회사 이름 추출
def extract_company_name(texts):
    for text in texts:
        match = re.search(r"(.*?)\s+(청평화|전화|HP|TEL|)", text)
        if match:
            return match.group(1).strip()
    return "(가게 이름)"  # 회사 이름을 찾을 수 없을 경우 반환될 기본값


# OCR 실행 라우트
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
            return jsonify({"error": "이미지에서 텍스트를 찾을 수 없습니다."}), 400

        texts = [entry["text"] for entry in extracted_data]
        date = extract_date(texts)
        company_name = extract_company_name(texts)  # 회사 이름 추출
        items = extract_items(texts)

        image_url = url_for("get_image", filename=filename, _external=True)
        return jsonify(
            {
                "text": texts,
                "image": image_url,
                "tradeAt": date,
                "company": company_name,
                "items": items,
            }
        )
    return jsonify({"error": "유효하지 않은 파일입니다."}), 400


# 업로드된 이미지를 제공하는 라우트
@app.route("/image/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# 메인 실행
if __name__ == "__main__":
    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"서버가 다음 IP에서 실행 중입니다: {ip_address}")
    app.run(host="0.0.0.0", port=5000)
