import logging
import os
import re
import socket

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

# 환경 변수 로드
load_dotenv()

# 환경 변수 사용:
UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER", "/workspace_project/AIproject/workspace/received_images"
)
MODEL_STORAGE_DIRECTORY = os.getenv(
    "MODEL_STORAGE_DIRECTORY", "/workspace_project/AIproject/workspace/user_network_dir"
)
# 로깅 추가:
logging.basicConfig(level=logging.INFO)

# 설정 변수들
UPLOAD_FOLDER = "/workspace_project/AIproject/workspace/received_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MODEL_STORAGE_DIRECTORY = "/workspace_project/AIproject/workspace/user_network_dir"
BLOCKED_CHARACTERS = "<\>][+=|`@#$%^&;'}{"

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
    image_path, filename, use_custom_model=True, languages=["ko"]
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
        preprocessed_image, blocklist=BLOCKED_CHARACTERS, width_ths=1.5, ycenter_ths=0.5
    )

    extracted_data = []
    loaded_image = cv2.imread(image_path)
    for bbox, text, confidence in result:
        extracted_data.append({"text": text, "confidence": confidence})
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(loaded_image, top_left, bottom_right, (251, 84, 20), 2)

    result_image_filename = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_image_filename, loaded_image)

    return extracted_data, result_image_filename


# 날짜 추출
def extract_date(texts):
    for text in texts:
        # yyyy-mm-dd 형태 찾기
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        match1 = re.search(r"\d{2}-\d{2}-\d{2}", text)
        if match:
            return match.group()
        elif match1:
            return match1.group()

        # yyyymmdd 형태 찾기
        match = re.search(r"\d{4}\d{2}\d{2}", text)
        if match:
            # 찾은 날짜를 yyyy-mm-dd 형태로 변환
            return f"{match.group(0)[:4]}-{match.group(0)[4:6]}-{match.group(0)[6:8]}"

    return None


# OCR 실수 교정
def correct_ocr_mistakes(text):
    corrections = {
        "I": "1",
        "l": "1",
        "O": "0",
        "o": "0",
        "D": "0",
        " ": "",
        ",": "",
        ".": "",
        "-": "000",
        "_": "000",
    }
    corrected_text = "".join(corrections.get(char, char) for char in text)
    return corrected_text


# 숫자만 추출
def extract_numbers(text):
    corrected_text = correct_ocr_mistakes(text)
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", corrected_text)
    if matches:
        return int(matches[0]) if "." not in matches[0] else float(matches[0])
    return 0  # 숫자가 없는 경우 0을 반환


# 아이템 라인 처리
def process_item_line(line):
    parts = re.split(r"\s+", line.strip())
    item_data = {"item": "", "unitPrice": 0, "quantity": 0, "price": 0}

    if parts:
        item_data["item"] = parts[0]  # 첫 부분을 상품명으로 가정
        # 숫자가 포함된 부분을 가격, 수량, 단가로 가정
        numbers = [
            extract_numbers(part) for part in parts[1:]
        ]  # 첫 부분을 제외한 나머지에서 숫자 추출

        if len(numbers) >= 3:  # 가격, 수량, 단가 순으로 가정
            item_data["price"], item_data["quantity"], item_data["unitPrice"] = numbers[
                :3
            ]

    return item_data


# 숫자인지 확인 및 변환 로직
def get_numeric_value(text):
    corrected_text = correct_ocr_mistakes(text)
    # OCR 실수 교정 함수를 통해 텍스트를 수정
    corrected_text = correct_ocr_mistakes(text)
    # 쉼표를 제거하여 숫자만 있는 문자열 생성
    corrected_text = corrected_text.replace(",", "")

    numeric_value = re.sub(r"[^\d.]", "", corrected_text)  # 숫자와 소수점만 남김

    # 숫자 변환 시도
    try:
        # 소수점이 포함되어 있는지 확인하고 적절히 변환
        return float(corrected_text) if "." in corrected_text else int(corrected_text)
    except ValueError:
        # 변환 실패 시 0 반환
        return 0


# 아이템 추출
def extract_items(texts):
    items = []
    temp_items = []
    capture = False

    start_keywords = os.getenv("START_KEYWORDS").split(",")
    end_keywords = os.getenv("END_KEYWORDS").split(",")

    for text in texts:
        if any(keyword in text for keyword in start_keywords):
            capture = True
            continue

        if any(keyword in text for keyword in end_keywords):
            capture = False
            continue

        if capture and text.strip():
            temp_items.append(text.strip())

    # 아이템 데이터 처리
    item_set = {"item": "", "unitPrice": 0, "quantity": 0, "price": 0}
    item_index = 0
    for text in temp_items:
        if item_index == 0:  # 아이템 이름
            item_set["item"] = text
            item_index += 1
        else:
            numeric_value = get_numeric_value(text)
            if isinstance(numeric_value, (int, float)):
                if item_index == 1:
                    item_set["unitPrice"] = numeric_value
                elif item_index == 2:
                    item_set["quantity"] = numeric_value
                elif item_index == 3:
                    item_set["price"] = numeric_value
                item_index += 1
            else:
                # 숫자가 아닌 값은 다음 세트로 이동
                items.append(item_set)
                item_set = {"item": text, "unitPrice": 0, "quantity": 0, "price": 0}
                item_index = 1

        if item_index > 3:
            items.append(item_set)
            item_set = {"item": "", "unitPrice": 0, "quantity": 0, "price": 0}
            item_index = 0

    # 마지막 아이템 처리
    if item_index > 0:
        items.append(item_set)

    return items


# 회사 이름 추출
company_identifiers = os.getenv("COMPANY_IDENTIFIERS").split(",")


def extract_company_name(texts):
    priority_keywords = ["HP", "청평화"]  # 우선 순위가 높은 순으로 정렬
    company_name = None  # 초기 회사 이름은 없음으로 설정

    for text in texts:
        # 가게 이름이 될 수 있는 텍스트를 찾기
        if any(company_id in text for company_id in company_identifiers):
            if company_name:
                return company_name
            else:
                # 가게 이름을 찾을 수 없는 경우 현재 텍스트를 반환
                return text
        else:
            # 가게 이름 후보를 업데이트
            company_name = text

    # 회사 이름이 발견되면 필터링 규칙 적용
    if company_name:
        # 필터링 규칙 정의
        filters = {
            # 추가 필터링 규칙은 여기에 정의
        }

        # 규칙에 따라 텍스트 변환
        for key, value in filters.items():
            company_name = company_name.replace(key, value)

    return company_name or "(가게 이름)"  # 회사 이름을 찾을 수 없을 경우 기본값 반환


# 필터링 함수 정의
def apply_text_filters(text_list):
    filters = {
        "광바패년 올": "광희패션몰",
        "뜨월": "프릴",
        "g6PLACE": "96PLACE",
        "아 N 길": "아 사 렐",
        "얼로무미": "얼로우미",
        "그럴t루": "그랑블루",
        "CP.u": "C.P.U",
        "CP.U": "C.P.U",
        "CPu": "C.P.U",
        "safl": "Sam",
        "( 신스)": "(샘스)",
        "오렌지붕": "오렌지붐",
        "오렌지봄": "오렌지붐",
        "Galdstar": "Goldstar",
        "청평하가": "청평화상가",
        "덕분입다다": "덕분입니다",
        "mrghmallo": "marshmallow",
        "(marahmallo)": "(marshmallow)",
        "(marghmmallo)": "(marshmallow)",
        "marsmmallo": "marshmallow",
        "야트 프라자 ! 층": "아트 프라자 1층",
        "만매소계": "판매소계",
        "만 매소계": "판매소계",
        "민매소계": "판매소계",
        "딴대 소계": "판매소계",
        "딴매소계": "판매소계",
        "당입합계": "당일합계",
        "거대처명": "거래처명",
        "신 업": "신협",
        "신환은행": "신한은행",
        "0OO": "000",
        "O0O": "000",
        "00O": "000",
        "O00": "000",
        "0O0": "000",
        "0oo": "000",
        "o0o": "000",
        "00o": "000",
        "o00": "000",
        "0o0": "000",
        "O0o": "000",
        "OOO": "000",
        ",0OU": ",000",
        ",OuO": ",000",
        ",rOO": ",000",
        ",0DO": ",000",
        ",OOO": ",000",
        "OO0": ",000",
        ",OOO": ",000",
        # 추가 필터링 규칙은 여기에 정의
    }
    filtered_texts = []
    for text in text_list:
        for key, value in filters.items():
            text = text.replace(key, value)
        filtered_texts.append(text)
    return filtered_texts


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

        # 날짜 추출
        date = extract_date(texts)  # 필터링 적용 전에 날짜 추출

        # 필터링 적용
        filtered_texts = apply_text_filters(texts)

        confidences = [entry["confidence"] for entry in extracted_data]
        if confidences:
            average_confidence = sum(confidences) / len(confidences)
            average_confidence = round(
                average_confidence, 4
            )  # 소수점 네 자리에서 반올림
        else:
            average_confidence = 0

        date = extract_date(filtered_texts)
        company_name = extract_company_name(filtered_texts)  # 회사 이름 추출
        items = extract_items(filtered_texts)

        image_url = url_for("get_image", filename=filename, _external=True)
        return jsonify(
            {
                "text": filtered_texts,
                "image": image_url,
                "tradeAt": date,
                "company": company_name,
                "items": items,
                "confidence": average_confidence,
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
    app.run(host="0.0.0.0", port=5000, debug=True)
