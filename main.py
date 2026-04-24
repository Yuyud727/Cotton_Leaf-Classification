import cv2
import os
import hashlib
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
categories   = ["Sehat", "Berlubang", "Kering"]
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
            data.append((img_bgr, img_path, category))  # ← simpan path untuk cek duplikat
            count += 1
        else:
            print(f"[WARNING] Gagal baca: {img_path}")
    print(f"  {category}: {count} gambar")

print(f"\nTotal data: {len(data)}")

# ── Cek Duplikasi ─────────────────────────────────────────
print("\nMengecek duplikasi gambar...")

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

hashes     = {}
duplicates = []

for _, img_path, label in data:
    h = get_hash(img_path)
    if h in hashes:
        duplicates.append((img_path, hashes[h]))
    else:
        hashes[img_path] = h  # simpan path → hash

# Deteksi hash yang muncul lebih dari sekali
hash_count = {}
for _, img_path, _ in data:
    h = get_hash(img_path)
    if h not in hash_count:
        hash_count[h] = []
    hash_count[h].append(img_path)

duplicates = {h: paths for h, paths in hash_count.items() if len(paths) > 1}

if duplicates:
    print(f"[WARNING] Ditemukan {len(duplicates)} grup duplikasi:")
    for h, paths in list(duplicates.items())[:10]:  # tampilkan 10 pertama
        print(f"  Hash: {h[:10]}...")
        for p in paths:
            print(f"    → {p}")
else:
    print("Tidak ada duplikasi ditemukan ✅")

# ── Preprocessing + Feature Extraction ───────────────────
X, y = [], []

for img_bgr, img_path, label in data:
    try:
        img_gray   = preprocess_image(img_bgr)
        feat_gray  = extract_features(img_gray)
        feat_color = extract_color_features(img_bgr)
        features   = np.concatenate([feat_gray, feat_color])
        X.append(features)
        y.append(label)
    except Exception as e:
        print(f"[WARNING] Gagal proses {img_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"\nShape X    : {X.shape}")
print(f"Distribusi : { {c: list(y).count(c) for c in categories} }")

# ── Split Data ────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # ← dikembalikan ke 0.2 (standar)
    random_state=42,
    stratify=y
)

print(f"\nData training : {len(X_train)}")
print(f"Data testing  : {len(X_test)}")

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