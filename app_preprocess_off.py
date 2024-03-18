import os
import socket

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory, url_for
from PIL import Image
from werkzeug.utils import secure_filename

import easyocr

app = Flask(__name__)

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/uploaded_images"
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
        image = np.array(Image.open(save_path))

        # OCR을 수행합니다.
        reader = easyocr.Reader(["ko", "en"])
        ocr_result = reader.readtext(image, paragraph=True)

        # OCR 결과에서 텍스트를 추출합니다.
        texts = [res[1] for res in ocr_result]

        # OCR 결과에 대한 박스를 그립니다.
        for res in ocr_result:
            top_left = tuple(res[0][0])
            bottom_right = tuple(res[0][2])
            color = (255, 0, 0)  # BGR 색상을 지정합니다. 이 경우는 파란색입니다.
            image = cv2.rectangle(image, top_left, bottom_right, color, 2)

        # 박스가 그려진 이미지를 저장합니다.
        name, ext = os.path.splitext(filename)
        result_image_filename = f"{name}_result{ext}"
        result_image_path = os.path.join(UPLOAD_FOLDER, result_image_filename)
        cv2.imwrite(result_image_path, image)

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


@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print(f"Server running on IP: {ip_address}")

# Flask 앱을 실행합니다.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
