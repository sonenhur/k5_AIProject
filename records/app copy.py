import os
import socket

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

# Configuration variables
UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/recieved_images"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MODEL_STORAGE_DIRECTORY = "C:/workspace_project/AIproject/workspace/user_network_dir"
# BLOCKED_CHARACTERS = "<{}>|]\-:[_+=`!@~#$%^&*)(?;'"
BLOCKED_CHARACTERS = "<>|]\-:[_+=`@#$%^&*;'"
# Ensure upload and model directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_STORAGE_DIRECTORY, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_path):
    # 이미지 읽기
    image = cv2.imread(image_path)

    # 노이즈 제거를 위한 Gaussian blur 적용
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # 중간값 필터(Median Filter)를 사용하여 소금-후추 노이즈를 제거합니다.
    # image = cv2.medianBlur(image, 5)

    # 그레이스케일 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 향상을 위한 히스토그램 평활화
    # image = cv2.equalizeHist(image)

    # 적응 임계값 적용
    # image = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )

    return image


def extract_text_from_image(
    image_path, filename, use_custom_model=False, languages=["ko"]
):
    # 경로 설정
    model_storage_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)
    user_network_directory = os.path.abspath(MODEL_STORAGE_DIRECTORY)

    # 모델 타입에 따른 EasyOCR reader 초기화
    if use_custom_model:
        reader = easyocr.Reader(
            lang_list=languages,
            gpu=True,
            model_storage_directory=model_storage_directory,
            user_network_directory=user_network_directory,
            recog_network="custom",
        )
    else:
        reader = easyocr.Reader(lang_list=languages, gpu=True)

    # 이미지 전처리
    preprocessed_image = preprocess_image(image_path)

    # OCR 실행
    result = reader.readtext(
        preprocessed_image, blocklist=BLOCKED_CHARACTERS, width_ths=15, ycenter_ths=0.5
    )

    # 결과 추출 및 이미지에 박스 그리기
    extracted_data = []
    loaded_image = cv2.imread(image_path)
    for bbox, text, confidence in result:
        extracted_data.append({"text": text, "confidence": confidence})
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(loaded_image, top_left, bottom_right, (255, 0, 0), 2)

        # 신뢰도 표시를 원하지 않으므로 아래 코드를 주석 처리하거나 삭제합니다.
        # text_position = (top_left[0], top_left[1] - 10)
        # cv2.putText(
        #     loaded_image,
        #     f"{confidence*100:.2f}%",
        #     text_position,
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.8,
        #     (255, 0, 0),
        #     2,
        # )

    # 결과 이미지 저장
    result_image_filename = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(result_image_filename, loaded_image)

    return extracted_data, result_image_filename


# @app.route("/upload", methods=["POST"])
# def get_photo_input():
#     file = request.files.get("image")
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#         file.save(save_path)

#         extracted_data, result_image_filename = extract_text_from_image(
#             save_path, filename
#         )
#         if not extracted_data:
#             return jsonify({"error": "No text found in image"}), 400

#         texts = [entry["text"] for entry in extracted_data]
#         image = url_for("get_image", filename=filename, _external=True)
#         return jsonify({"text": texts, "image": image})

#     return jsonify({"error": "Invalid file"}), 400


# 하드코딩 테스트
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

        image_url = url_for("get_image", filename=filename, _external=True)
        texts = [entry["text"] for entry in extracted_data]
        # return jsonify({"text": texts, "image": image_url})

        # # 아래는 예시 데이터입니다. 실제 데이터로 대체해야 합니다.
        # return jsonif                                                                                                                               y(
        #     {
        #         "items": [
        #             "비비포켓배기/그레이/L",
        #             20000,
        #             6,
        #             120000,
        #             "비비포켓배기/그레이/S",
        #             20000,
        #             3,
        #             60000,
        #             "비비포켓배기/그레이/M",
        #             20000,
        #             6,
        #             120000,
        #         ],
        #         "image": image_url,
        #     }
        # )

        # # 더미 데이터
        return jsonify(
            {
                "company": "디디",
                "tradeAt": "2022-04-25",
                "sum": 500000,
                "items": [
                    {
                        "item": "비비포켓배기/그레이/L",
                        "unitPrice": 20000,
                        "quantity": 6,
                        "price": 120000,
                    },
                    {
                        "item": "비비포켓배기/그레이/S",
                        "unitPrice": 20000,
                        "quantity": 3,
                        "price": 60000,
                    },
                    {
                        "item": "비비포켓배기/그레이/M",
                        "unitPrice": 20000,
                        "quantity": 6,
                        "price": 120000,
                    },
                    {
                        "item": "비비포켓배기/베이지/S",
                        "unitPrice": 20000,
                        "quantity": 4,
                        "price": 80000,
                    },
                    {
                        "item": "비비포켓배기/그레이/S",
                        "unitPrice": 20000,
                        "quantity": 3,
                        "price": 60000,
                    },
                    {
                        "item": "비비포켓배기/그레이/M",
                        "unitPrice": 20000,
                        "quantity": 6,
                        "price": 120000,
                    },
                ],
                "image": image_url,
            }
        )

    return jsonify({"error": "Invalid file"}), 400


@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    ip_address = socket.gethostbyname(socket.gethostname())
    print(f"Server running on IP: {ip_address}")
    app.run(host="0.0.0.0", port=5000, debug=True)
