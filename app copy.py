import os
import re
import socket

import cv2
from flask import Flask, jsonify, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

import easyocr
from easyocr import Reader

app = Flask(__name__)

UPLOAD_FOLDER = "C:/workspace_project/AIproject/workspace/demo_images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_image(image_path, filename, languages=["ko", "en"]):
    blocked_characters = "COUBJuIi<{}>|]\-:[_+=`!@~#$%^&*)(?;\"'"
    model_storage_directory = (
        "C:/workspace_project/AIproject/workspace/user_network_dir"
    )
    user_network_directory = "C:/workspace_project/AIproject/workspace/user_network_dir"

    # Ensure the directories exist and contain the expected files
    assert os.path.exists(
        user_network_directory
    ), f"Directory not found: {user_network_directory}"

    reader = Reader(
        lang_list=languages,
        gpu=True,
        # model_storage_directory=model_storage_directory,
        # user_network_directory=user_network_directory,
        # recog_network="custom",  # Make sure the 'custom' model exists in the specified directory
    )
    loaded_image = cv2.imread(
        image_path
    )  # Renamed variable to 'loaded_image' to avoid conflict
    try:
        result = reader.readtext(
            loaded_image,
            detail=1,
            blocklist=blocked_characters,
        )
        extracted_data = []
        for bbox, text, confidence in result:
            extracted_data.append({"text": text, "confidence": confidence})

            # Draw bounding boxes on the loaded image
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])
            loaded_image = cv2.rectangle(
                loaded_image, top_left, bottom_right, (0, 255, 0), 2
            )

        # Save the image with drawn boxes
        name, ext = os.path.splitext(filename)
        result_image_filename = f"{name}{ext}"
        result_image_path = os.path.join(UPLOAD_FOLDER, result_image_filename)
        cv2.imwrite(result_image_path, loaded_image)

        return extracted_data, result_image_filename
    except Exception as e:
        print(f"Error during OCR: {e}")
        return [], None


def find_by_date(image, languages=["en", "ko"]):
    reader = easyocr.Reader(languages)
    blocked_characters = ";이터,"

    result = reader.readtext(
        image,
        width_ths=0.5,
        ycenter_ths=0.5,
        blocklist=blocked_characters,
        paragraph=True,
    )
    pattern = r"(\d{4}-\d{2}-\d{2})"

    for bbox, text, prob in result:
        match = re.search(pattern, text)
        if match:
            return match.group(1) + "T00:00:00"

    return ""


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

        extracted_data, result_image_filename = extract_text_from_image(
            save_path, filename
        )
        if not extracted_data:
            return jsonify({"error": "No text found in image"}), 400

        texts = [entry["text"] for entry in extracted_data]

        return jsonify(
            {
                "text": texts,
                "image_url": url_for(
                    "get_image", filename=result_image_filename, _external=True
                ),
            }
        )


# Ensure the route for get_image is correctly defined
@app.route("/images/<filename>")
def get_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Server running on IP: {ip_address}")
    app.run(host="0.0.0.0", port=5000, debug=True)
