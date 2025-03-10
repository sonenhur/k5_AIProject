# 필요한 라이브러리 임포트
import logging  # 로깅을 위한 모듈
import os  # 파일 및 디렉토리 작업을 위한 모듈
import re  # 정규 표현식을 위한 모듈
import socket  # 네트워크 관련 작업을 위한 모듈

import cv2  # OpenCV, 이미지 처리를 위한 라이브러리
from dotenv import load_dotenv  # 환경 변수 로드를 위한 모듈
from flask import Flask, jsonify, request, send_from_directory, url_for  # Flask 웹 프레임워크 모듈
from werkzeug.utils import secure_filename  # 파일 이름을 안전하게 처리하는 유틸리티

import easyocr  # OCR(광학 문자 인식)을 위한 라이브러리

# Flask 애플리케이션 생성
app = Flask(__name__)

# 환경 변수 로드: .env 파일에서 설정값을 가져옴
load_dotenv()

# 환경 변수 사용: 이미지 업로드와 모델 저장 경로를 설정
UPLOAD_FOLDER = os.getenv(
    "UPLOAD_FOLDER", "/workspace_project/AIproject/workspace/received_images"
)  # 업로드된 이미지를 저장할 폴더 경로
MODEL_STORAGE_DIRECTORY = os.getenv(
    "MODEL_STORAGE_DIRECTORY", "/workspace_project/AIproject/workspace/user_network_dir"
)  # OCR 모델 저장 경로

# 로깅 설정: 코드 실행 중 정보를 기록하도록 설정
logging.basicConfig(level=logging.INFO)

# 설정 변수들: 코드에서 사용할 주요 변수 정의
UPLOAD_FOLDER = "/workspace_project/AIproject/workspace/received_images"  # 이미지 업로드 폴더
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}  # 허용되는 이미지 파일 확장자
MODEL_STORAGE_DIRECTORY = "/workspace_project/AIproject/workspace/user_network_dir"  # 모델 저장 폴더
BLOCKED_CHARACTERS = "<\>][+=|`@#$%^&;'}{\""  # OCR에서 제외할 특수 문자

# 폴더가 없는 경우 생성: 지정된 경로에 폴더가 없으면 새로 만듦
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_STORAGE_DIRECTORY, exist_ok=True)

# Flask 설정에 업로드 폴더 경로 추가
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# 허용되는 파일 형식인지 확인하는 함수
def allowed_file(filename):
    """파일 이름에 '.'이 있고, 확장자가 허용된 목록에 있는지 확인"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# 이미지를 그레이스케일로 전처리하는 함수
def preprocess_image(image_path):
    """이미지를 읽고 그레이스케일로 변환하여 OCR 성능을 높임"""
    image = cv2.imread(image_path)  # 이미지 파일 읽기
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 컬러 이미지를 그레이스케일로 변환
    return image

# 이미지에서 텍스트를 추출하는 함수
def extract_text_from_image(image_path, filename, use_custom_model=False, languages=["ko"]):
    """이미지에서 텍스트를 읽고, 결과를 이미지에 표시"""
    model_storage_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)  # 절대 경로로 변환
    user_network_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)

    # OCR 리더 설정: 커스텀 모델 사용 여부에 따라 다르게 초기화
    if use_custom_model:
        reader = easyocr.Reader(
            lang_list=languages,  # 인식할 언어 목록
            gpu=True,  # GPU 사용 여부
            model_storage_directory=model_storage_directory,  # 모델 저장 경로
            user_network_directory=user_network_directory,  # 사용자 네트워크 경로
            recog_network="custom",  # 커스텀 인식 네트워크 사용
        )
    else:
        reader = easyocr.Reader(lang_list=["ko", "en"], gpu=True)  # 기본 한국어+영어 설정

    preprocessed_image = preprocess_image(image_path)  # 이미지 전처리

    # OCR 실행: 텍스트와 위치, 신뢰도 추출
    result = reader.readtext(
        preprocessed_image,
        blocklist=BLOCKED_CHARACTERS,  # 제외할 문자 목록
        width_ths=1.5,  # 텍스트 너비 기준
        ycenter_ths=0.5  # 텍스트 높이 기준
    )

    extracted_data = []  # 추출된 데이터를 저장할 리스트
    loaded_image = cv2.imread(image_path)  # 원본 이미지 로드
    for bbox, text, confidence in result:
        extracted_data.append({"text": text, "confidence": confidence})  # 텍스트와 신뢰도 저장
        top_left = tuple(map(int, bbox[0]))  # 텍스트 영역의 좌상단 좌표
        bottom_right = tuple(map(int, bbox[2]))  # 텍스트 영역의 우하단 좌표
        cv2.rectangle(loaded_image, top_left, bottom_right, (251, 84, 20), 2)  # 텍스트 영역에 사각형 그리기

    # 결과 이미지를 저장
    result_image_filename = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_image_filename, loaded_image)

    return extracted_data, result_image_filename  # 추출된 데이터와 저장된 이미지 경로 반환

# 날짜를 추출하는 함수
def extract_date(texts):
    """텍스트에서 날짜 형식(yyyy-mm-dd 또는 yyyymmdd)을 찾아 반환"""
    for text in texts:
        # yyyy-mm-dd 또는 yy-mm-dd 형식 찾기
        match = re.search(r"\d{4}-\d{2}-\d{2}", text)
        match1 = re.search(r"\d{2}-\d{2}-\d{2}", text)
        if match:
            return match.group()
        elif match1:
            return match1.group()

        # yyyymmdd 형식 찾기
        match = re.search(r"\d{4}\d{2}\d{2}", text)
        if match:
            # yyyy-mm-dd 형식으로 변환
            return f"{match.group(0)[:4]}-{match.group(0)[4:6]}-{match.group(0)[6:8]}"
    
    return None  # 날짜가 없으면 None 반환

# OCR 실수를 교정하는 함수
def correct_ocr_mistakes(text):
    """OCR에서 흔히 발생하는 문자 인식 오류를 교정"""
    corrections = {
        "I": "1", "l": "1", "O": "0", "o": "0", "D": "0",  # 비슷한 모양의 문자 교정
        " ": "", ",": "", ".": "", "-": "000", "_": "000"  # 공백 및 특수 문자 제거
    }
    corrected_text = "".join(corrections.get(char, char) for char in text)  # 교정 적용
    return corrected_text

# 텍스트에서 숫자만 추출하는 함수
def extract_numbers(text):
    """텍스트에서 숫자를 추출하고 정수 또는 실수로 변환"""
    corrected_text = correct_ocr_mistakes(text)  # OCR 오류 교정
    matches = re.findall(r"\b\d+(?:\.\d+)?\b", corrected_text)  # 숫자 패턴 찾기
    if matches:
        return int(matches[0]) if "." not in matches[0] else float(matches[0])  # 정수/실수 변환
    return 0  # 숫자가 없으면 0 반환

# 아이템 라인을 처리하는 함수
def process_item_line(line):
    """한 줄의 텍스트를 상품명, 수량, 단가, 가격으로 분리"""
    parts = re.split(r"\s+", line.strip())  # 공백으로 분리
    item_data = {"item": "", "quantity": 0, "unitPrice": 0, "price": 0}  # 기본 데이터 구조

    if parts:
        item_data["item"] = parts[0]  # 첫 부분을 상품명으로 설정
        numbers = [extract_numbers(part) for part in parts[1:]]  # 나머지에서 숫자 추출
        if len(numbers) >= 3:
            item_data["quantity"], item_data["unitPrice"], item_data["price"] = numbers[:3]  # 수량, 단가, 가격 설정
    
    return item_data

# 숫자인지 확인하고 변환하는 함수
def get_numeric_value(text):
    """텍스트를 숫자로 변환, 실패 시 0 반환"""
    corrected_text = correct_ocr_mistakes(text)  # OCR 오류 교정
    corrected_text = corrected_text.replace(",", "")  # 쉼표 제거
    numeric_value = re.sub(r"[^\d.]", "", corrected_text)  # 숫자와 소수점만 남김
    
    try:
        return float(corrected_text) if "." in corrected_text else int(corrected_text)  # 숫자 변환
    except ValueError:
        return 0  # 변환 실패 시 0 반환

# 아이템을 추출하는 함수
def extract_items(texts):
    """텍스트에서 아이템 목록을 추출"""
    items = []
    temp_items = []
    capture = False  # 아이템 캡처 여부

    start_keywords = os.getenv("START_KEYWORDS").split(",")  # 아이템 시작 키워드
    end_keywords = os.getenv("END_KEYWORDS").split(",")  # 아이템 종료 키워드

    # 텍스트에서 아이템 영역 찾기
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
    item_set = {"item": "", "quantity": 0, "unitPrice": 0, "price": 0}
    item_index = 0
    for text in temp_items:
        if item_index == 0:  # 상품명
            item_set["item"] = text
            item_index += 1
        else:
            numeric_value = get_numeric_value(text)
            if isinstance(numeric_value, (int, float)):
                if item_index == 1:
                    item_set["quantity"] = numeric_value
                elif item_index == 2:
                    item_set["unitPrice"] = numeric_value
                elif item_index == 3:
                    item_set["price"] = numeric_value
                item_index += 1
            else:
                items.append(item_set)
                item_set = {"item": text, "quantity": 0, "unitPrice": 0, "price": 0}
                item_index = 1
        
        if item_index > 3:
            items.append(item_set)
            item_set = {"item": "", "quantity": 0, "unitPrice": 0, "price": 0}
            item_index = 0

    if item_index > 0:  # 마지막 아이템 추가
        items.append(item_set)

    return items

# 회사 이름을 추출하는 함수
company_identifiers = os.getenv("COMPANY_IDENTIFIERS").split(",")  # 회사 식별 키워드

def extract_company_name(texts):
    """텍스트에서 회사 이름을 추출"""
    priority_keywords = ["HP", "청평화"]  # 우선순위 키워드
    company_name = None

    for text in texts:
        if any(company_id in text for company_id in company_identifiers):
            if company_name:
                return company_name
            else:
                return text
        else:
            company_name = text

    if company_name:
        filters = {}  # 필터링 규칙 (필요 시 추가)
        for key, value in filters.items():
            company_name = company_name.replace(key, value)

    return company_name or "(가게 이름)"  # 기본값 반환

# 필터 규칙을 환경 변수에서 불러오는 함수
def load_filters():
    """환경 변수에서 필터 규칙 로드"""
    filters = {}
    for key in os.environ:
        if key.startswith("FILTER_RULE_"):
            rule = os.environ[key]
            source, target = rule.split("=")
            filters[source] = target
    return filters

# 텍스트 필터링 함수
def apply_text_filters(text_list):
    """텍스트에 필터 규칙 적용"""
    filters = load_filters()
    filtered_texts = []
    for text in text_list:
        for key, value in filters.items():
            text = text.replace(key, value)
        filtered_texts.append(text)
    return filtered_texts

# OCR 실행 라우트: 이미지 업로드 및 텍스트 추출
@app.route("/image/upload", methods=["POST"])
def get_photo_input():
    """이미지를 업로드받아 OCR 처리 후 결과 반환"""
    file = request.files.get("image")  # 업로드된 파일 가져오기
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)  # 안전한 파일 이름 생성
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)  # 저장 경로
        file.save(save_path)  # 파일 저장

        # 텍스트 추출 및 이미지 처리
        extracted_data, result_image_filename = extract_text_from_image(save_path, filename)
        if not extracted_data:
            return jsonify({"error": "이미지에서 텍스트를 찾을 수 없습니다."}), 400

        texts = [entry["text"] for entry in extracted_data]  # 추출된 텍스트 리스트

        date = extract_date(texts)  # 날짜 추출
        filtered_texts = apply_text_filters(texts)  # 텍스트 필터링

        # 신뢰도 평균 계산
        confidences = [entry["confidence"] for entry in extracted_data]
        average_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0

        date = extract_date(filtered_texts)  # 필터링된 텍스트에서 날짜 재추출
        company_name = extract_company_name(filtered_texts)  # 회사 이름 추출
        items = extract_items(filtered_texts)  # 아이템 목록 추출

        image_url = url_for("get_image", filename=filename, _external=True)  # 이미지 URL 생성
        return jsonify(
            {
                "text": filtered_texts,  # 필터링된 텍스트
                "image": image_url,  # 결과 이미지 URL
                "tradeAt": date,  # 거래 날짜
                "company": company_name,  # 회사 이름
                "items": items,  # 아이템 목록
                "confidence": average_confidence  # 평균 신뢰도
            }
        )
    return jsonify({"error": "유효하지 않은 파일입니다."}), 400

# 업로드된 이미지를 제공하는 라우트
@app.route("/image/<filename>")
def get_image(filename):
    """업로드된 이미지를 클라이언트에 제공"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# 메인 실행: 서버 시작
if __name__ == "__main__":
    ip_address = socket.gethostbyname(socket.gethostname())  # 현재 IP 주소 가져오기
    print(f"서버가 다음 IP에서 실행 중입니다: {ip_address}")
    app.run(host="0.0.0.0", port=5000, debug=True)  # Flask 서버 실행
