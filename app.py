import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify
from model.svm_model import load_model
from preprocessing.preprocess import preprocess_image
from feature_extraction.features import extract_features, extract_color_features  # ← tambah import

os.chdir(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = load_model()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "File kosong"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        img_bgr = cv2.imread(filepath)
        if img_bgr is None:
            return jsonify({"error": "Gambar tidak valid"}), 400

        # Preprocessing → grayscale
        img_gray = preprocess_image(img_bgr)

        # Gabung fitur gray + warna (sama seperti main.py)
        feat_gray  = extract_features(img_gray)
        feat_color = extract_color_features(img_bgr)       # ← tambah fitur warna
        features   = np.concatenate([feat_gray, feat_color]).reshape(1, -1)

        prediction = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        classes    = model.classes_

        # Confidence per kelas
        confidence_dict = {
            cls: f"{prob * 100:.2f}%"
            for cls, prob in zip(classes, proba)
        }
        confidence_max = f"{proba.max() * 100:.2f}%"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return jsonify({
        "result":     prediction,       # "Sehat" / "Berlubang" / "Kering"
        "confidence": confidence_max,
        "detail":     confidence_dict
    })

if __name__ == "__main__":
    app.run(debug=True)