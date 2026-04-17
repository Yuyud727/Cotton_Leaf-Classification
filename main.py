import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.svm_model import train_model, save_model
from preprocessing.preprocess import preprocess_image
from feature_extraction.features import extract_features

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Paksa working directory ke lokasi file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load Dataset ──────────────────────────────────────────
dataset_path = "dataset"
categories = ["crack", "good"]
data = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        print(f"[ERROR] Folder tidak ditemukan: {folder}")
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            data.append((img, category))
        else:
            print(f"[WARNING] Gagal baca: {img_path}")

print(f"Total data: {len(data)}")

# ── Preprocessing ─────────────────────────────────────────
processed_data = []

for img, label in data:
    try:
        img_processed = preprocess_image(img)
        processed_data.append((img_processed, label))
    except Exception as e:
        print(f"[WARNING] Preprocessing gagal: {e}")

print(f"Data setelah preprocessing: {len(processed_data)}")

# ── Feature Extraction ────────────────────────────────────
X, y = [], []

for img, label in processed_data:
    try:
        features = extract_features(img)
        X.append(features)
        y.append(label)
    except Exception as e:
        print(f"[WARNING] Feature extraction gagal: {e}")

X = np.array(X)
y = np.array(y)

print(f"Shape X: {X.shape}")
print(f"Distribusi kelas: { {c: list(y).count(c) for c in categories} }")

# ── Split Data ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # jaga distribusi kelas
)

# ── Training ──────────────────────────────────────────────
print("\nTraining model...")
model = train_model(X_train, y_train)
print("Training selesai!")

# ── Simpan Model ──────────────────────────────────────────
save_model(model)
print("Model berhasil disimpan!")

# ── Evaluasi ──────────────────────────────────────────────
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=categories)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=categories,
    yticklabels=categories
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Akurasi: {accuracy * 100:.2f}%)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix disimpan ke confusion_matrix.png")