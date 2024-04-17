import os
import socket

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from PIL import Image
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

print(f"Torch CUDA is_available ?: {torch.cuda.is_available()}")

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/demo_images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["POST"])
def getPhotoInput():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # 이미지를 열어 numpy 배열로 변환합니다.
        src = cv2.imread(save_path, 1)

        # 이미지가 올바르게 읽혔는지 확인
        if src is None:
            print(f"Could not read image {save_path}.")
            return jsonify({"error": "Could not read image"}), 400

        # 이미지가 RGB 또는 RGBA 채널을 가지고 있는지 확인
        if len(src.shape) < 3 or src.shape[2] < 3:
            print(f"Image {save_path} is not in RGB or RGBA format.")
            return jsonify({"error": "Image is not in RGB or RGBA format"}), 400

        # 이미지 향상
        src = cv2.convertScaleAbs(
            src, alpha=1.5, beta=0
        )  # 명암, 채도, 대비 등을 조정합니다.

        # 노이즈 제거
        denoised_src = cv2.fastNlMeansDenoisingColored(src, None, 1, 1, 9, 27)

        # 그레이스케일 변환
        gray = cv2.cvtColor(denoised_src, cv2.COLOR_BGR2GRAY)

        # 적응 임계값 적용
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # OCR을 수행합니다.
        reader = easyocr.Reader(["ko", "en"])
        ocr_result = reader.readtext(binary, paragraph=True)

        # OCR 결과에서 텍스트만 추출합니다.
        texts = [res[1] for res in ocr_result]

        # OCR 결과에 대한 박스를 그립니다.
        for res in ocr_result:
            top_left = tuple(res[0][0])
            bottom_right = tuple(res[0][2])
            color = (255, 0, 0)  # BGR 색상을 지정합니다. 이 경우는 빨간색입니다.
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)

        # 박스가 그려진 이미지를 저장합니다.
        result_image_filename = "ocr_result.png"
        result_image_path = os.path.join(UPLOAD_FOLDER, result_image_filename)
        cv2.imwrite(result_image_path, image)

        # # 임시로 저장된 이미지 파일을 삭제합니다.
        # os.remove(save_path)

        # OCR 결과 텍스트와 이미지 파일 경로를 JSON 응답으로 반환합니다.
        return jsonify(
            {
                "text": texts,
                "image_url": url_for(
                    "get_image", filename=result_image_filename, _external=True
                ),
            }
        )

    else:
        return jsonify({"error": "File type not allowed"}), 400


hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print(f"Server running on IP: {ip_address}")

# Flask 앱을 실행합니다.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
