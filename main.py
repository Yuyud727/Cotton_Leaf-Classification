import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model.svm_model import train_model, train_model_tuned, save_model
from preprocessing.preprocess import preprocess_image
from feature_extraction.features import extract_features, extract_color_features

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Load Dataset ──────────────────────────────────────────
dataset_path = "dataset"
categories   = ["Sehat", "Berlubang", "Kering"]  # ← 3 kelas
data         = []

for category in categories:
    folder = os.path.join(dataset_path, category)
    if not os.path.exists(folder):
        print(f"[ERROR] Folder tidak ditemukan: {folder}")
        continue

    count = 0
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img_bgr  = cv2.imread(img_path)
        if img_bgr is not None:
            data.append((img_bgr, category))
            count += 1
        else:
            print(f"[WARNING] Gagal baca: {img_path}")
    print(f"  {category}: {count} gambar")

print(f"\nTotal data: {len(data)}")

# ── Preprocessing + Feature Extraction ───────────────────
X, y = [], []

for img_bgr, label in data:
    try:
        img_gray     = preprocess_image(img_bgr)
        feat_gray    = extract_features(img_gray)
        feat_color   = extract_color_features(img_bgr)
        features     = np.concatenate([feat_gray, feat_color])
        X.append(features)
        y.append(label)
    except Exception as e:
        print(f"[WARNING] Gagal proses: {e}")

X = np.array(X)
y = np.array(y)

print(f"Shape X    : {X.shape}")
print(f"Distribusi : { {c: list(y).count(c) for c in categories} }")

# ── Split Data ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.6,
    random_state=42,
    stratify=y
)

# ── Training ──────────────────────────────────────────────
USE_TUNING = False

print("\nTraining model...")
model = train_model_tuned(X_train, y_train) if USE_TUNING else train_model(X_train, y_train)
print("Training selesai!")

# ── Cross Validation ──────────────────────────────────────
print("\nCross Validation (5-fold)...")
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
print(f"CV F1 Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ── Simpan Model ──────────────────────────────────────────
save_model(model)
print("Model berhasil disimpan!")

# ── Evaluasi ──────────────────────────────────────────────
y_pred   = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAkurasi: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Confusion Matrix ──────────────────────────────────────
cm = confusion_matrix(y_test, y_pred, labels=categories)
plt.figure(figsize=(7, 6))
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