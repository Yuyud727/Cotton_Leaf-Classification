# 🥚 Deteksi Kualitas Telur
### Klasifikasi Otomatis Berbasis Computer Vision & Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey?logo=flask)

---

## 📋 Deskripsi Proyek

Sistem klasifikasi kualitas telur secara otomatis menggunakan teknik **computer vision** dan **machine learning**. Sistem mampu membedakan telur berkualitas baik (**Good**) dan telur retak (**Crack**) dari gambar digital melalui antarmuka web.

**Latar belakang:** Deteksi kualitas telur secara manual memerlukan waktu, tenaga, dan rentan terhadap kesalahan manusia. Sistem otomatis diperlukan untuk meningkatkan efisiensi dan konsistensi proses seleksi telur.

---

## 📁 Struktur Proyek

```
egg-classification/
├── main.py                          # Script training & evaluasi model
├── app.py                           # Web server Flask
├── dataset/
│   ├── crack/                       # Gambar telur retak
│   └── good/                        # Gambar telur bagus
├── preprocessing/
│   └── preprocess.py                # Resize, grayscale, CLAHE, Gaussian Blur
├── feature_extraction/
│   └── features.py                  # Ekstraksi fitur HOG + LBP
├── model/
│   ├── svm_model.py                 # Definisi, training, simpan & load model
│   └── svm_model.pkl                # File model hasil training
├── templates/
│   └── index.html                   # Antarmuka web
├── static/
│   └── style.css                    # Styling halaman web
└── README.md
```

---

## ⚙️ Pipeline Sistem

```
Input Gambar
     ↓
Preprocessing
├── Resize → 128x128 piksel
├── Grayscale
├── CLAHE  (normalisasi pencahayaan)
└── Gaussian Blur (reduksi noise)
     ↓
Feature Extraction
├── HOG — menangkap pola tepi & struktur retakan
└── LBP — menangkap tekstur permukaan cangkang
     ↓
Klasifikasi SVM
├── StandardScaler (normalisasi fitur)
├── Kernel RBF
└── class_weight='balanced'
     ↓
Output: Crack / Good + Confidence Score
```

---

## 🛠️ Teknologi yang Digunakan

| Library | Fungsi |
|---------|--------|
| `opencv-python` | Baca gambar, resize, grayscale, CLAHE, blur |
| `scikit-image` | HOG dan LBP feature extraction |
| `scikit-learn` | SVM, StandardScaler, evaluasi model |
| `numpy` | Operasi array dan matriks |
| `flask` | Web server dan routing API |
| `matplotlib` + `seaborn` | Visualisasi confusion matrix |
| `pickle` | Simpan dan load model |

---

## 🧠 Alasan Pemilihan Metode

### Kenapa SVM?
- Efektif untuk dataset kecil hingga menengah
- Tidak membutuhkan GPU
- Training cepat dan interpretable
- Kernel RBF mampu menangani data yang tidak linearly separable

### Kenapa HOG + LBP?
- **HOG** → mendeteksi perubahan gradien intensitas yang tajam — karakteristik utama retakan
- **LBP** → mendeteksi perbedaan tekstur antara permukaan mulus dan retak
- Kombinasi keduanya lebih representatif dibanding raw pixel (`img.flatten()`)

---

## 📊 Hasil Evaluasi

**Akurasi: `98.59%`** pada 71 data test (split 80/20 dari 352 data total)

| Metrik | Crack | Good | Weighted Avg |
|--------|-------|------|--------------|
| Precision | 100% | 97% | 99% |
| Recall | 97% | 100% | 99% |
| F1-Score | 98% | 99% | 99% |
| Support | 33 | 38 | 71 |

### Confusion Matrix

```
                 Predicted
                Crack    Good
Actual  Crack     32       1
        Good       0      38
```

---

## 🚀 Cara Menjalankan

### 1. Clone repository
```bash
git clone https://github.com/citra-af/egg-classification.git
cd egg-classification
```

### 2. Install dependencies
```bash
pip install opencv-python numpy scikit-learn scikit-image matplotlib seaborn flask
```

### 3. Siapkan dataset
Letakkan gambar di folder:
```
dataset/crack/   ← gambar telur retak
dataset/good/    ← gambar telur bagus
```

### 4. Training model
```bash
python main.py
```

### 5. Jalankan web app
```bash
python app.py
```
Buka browser: **http://localhost:5000**

---

## ⚠️ Limitasi

- Model hanya akurat pada gambar yang **mirip karakteristik dataset training** (1 telur per frame, background polos, pencahayaan konsisten)
- Dataset relatif kecil (352 gambar) — berpotensi overfitting pada kondisi real world
- Tidak dapat mendeteksi **multiple telur** dalam satu gambar sekaligus
- Confidence score menggunakan Platt Scaling — bukan true probability

---

## 🔮 Saran Pengembangan

**Jangka pendek:**
- Tambah dataset minimal 1.000 gambar per kelas
- Terapkan data augmentation (flip, rotate, brightness)
- Tambah k-fold cross validation

**Jangka panjang:**
- Migrasi ke CNN pretrained (MobileNetV2 / EfficientNet)
- Tambah kelas klasifikasi (Grade A/B/C, fertile/infertile)
- Integrasi object detection (YOLO) untuk deteksi multiple telur
- Deploy ke mobile application

---

## ❓ FAQ

| Pertanyaan | Jawaban |
|-----------|---------|
| Kenapa SVM bukan CNN? | Dataset kecil, SVM lebih efisien dan tidak butuh GPU |
| Kenapa kernel RBF? | Data tidak linearly separable, RBF lebih fleksibel |
| Kenapa HOG + LBP? | HOG tangkap struktur retakan, LBP tangkap tekstur |
| Akurasi 98% tapi salah di real world? | Model terspesialisasi pada dataset — butuh data lebih beragam |
| Bisa deteksi banyak telur sekaligus? | Belum — butuh object detection (YOLO) |

---

> **Catatan:** Akurasi model sangat bergantung pada kualitas dan keberagaman dataset training. Pastikan gambar input memiliki karakteristik yang serupa dengan data training untuk hasil optimal.
# Cotton_Leaf-Classification
