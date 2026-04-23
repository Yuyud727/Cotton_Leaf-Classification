# 🌿 LeafScan — Klasifikasi Kondisi Daun

Sistem klasifikasi kondisi daun berbasis **Machine Learning (SVM + RBF Kernel)** yang mampu mendeteksi 3 kondisi daun secara otomatis melalui antarmuka web.

---

## Kategori Klasifikasi

| Kategori | Deskripsi | Ciri Visual |
|---|---|---|
| ✅ **Sehat** | Daun dalam kondisi normal | Hijau merata, utuh, tidak ada kerusakan |
| ⚠️ **Berlubang** | Daun mengalami kerusakan fisik | Ada lubang atau sobekan di permukaan/pinggir |
| 🍂 **Kering** | Daun mengalami dehidrasi/mati | Warna coklat/merah, layu, mengkerut |

---

## Teknologi yang Digunakan

| Komponen | Teknologi |
|---|---|
| Machine Learning | Scikit-learn — SVM (RBF Kernel) |
| Computer Vision | OpenCV, Scikit-image |
| Fitur Ekstraksi | HOG, LBP, HSV, LAB Color |
| Preprocessing | CLAHE, Gaussian Blur, Morphological |
| Dimensi Reduksi | PCA (95% variance) |
| Normalisasi | RobustScaler |
| Web Framework | Flask |
| Frontend | HTML, CSS, JavaScript |

---

## Struktur Proyek

```
Leaf_classification/
│
├── dataset/
│   ├── Sehat/              ← gambar daun sehat
│   ├── Berlubang/          ← gambar daun berlubang
│   └── Kering/             ← gambar daun kering
│
├── feature_extraction/
│   ├── __init__.py
│   └── features.py         ← HOG + LBP + HSV + LAB + dark ratio
│
├── model/
│   ├── __init__.py
│   ├── svm_model.py        ← SVM RBF pipeline + GridSearchCV
│   └── svm_model.pkl       ← model tersimpan (auto-generated)
│
├── preprocessing/
│   ├── __init__.py
│   └── preprocess.py       ← CLAHE + Gaussian Blur + Morphological
│
├── templates/
│   └── index.html          ← antarmuka web
│
├── static/
│   ├── css/style.css
│   └── js/script.js
│
├── uploads/                ← folder sementara (auto-generated)
├── main.py                 ← script training
├── app.py                  ← Flask web server
├── confusion_matrix.png    ← hasil evaluasi (auto-generated)
└── requirements.txt
```

---

## Instalasi

### 1. Clone atau download proyek

```bash
git clone https://github.com/username/leaf-classification.git
cd leaf-classification
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Buat file `requirements.txt` dengan isi berikut:

```
opencv-python
numpy
scikit-learn
scikit-image
matplotlib
seaborn
flask
```

---

## Persiapan Dataset

Susun folder dataset seperti berikut:

```
dataset/
├── Sehat/
│   ├── daun_001.jpg
│   ├── daun_002.jpg
│   └── ...
├── Berlubang/
│   ├── daun_001.jpg
│   └── ...
└── Kering/
    ├── daun_001.jpg
    └── ...
```

> Minimal **500 gambar per kelas** untuk hasil optimal. Dataset yang digunakan dalam proyek ini berjumlah **2000 gambar** (1000 per kelas untuk 2 kelas, atau dibagi rata untuk 3 kelas).

---

## Cara Penggunaan

### Langkah 1 — Training Model

```bash
python main.py
```

Output yang diharapkan:

```
Total data: 1500
Data setelah preprocessing: 1500
Shape X       : (1500, 8250)
Distribusi    : {'Sehat': 500, 'Berlubang': 500, 'Kering': 500}

Training model...
Training selesai!

Cross Validation (5-fold)...
CV F1 Score   : 0.9450 ± 0.0120
Model berhasil disimpan!

Akurasi       : 94.00%
```

> Untuk tuning otomatis, set `USE_TUNING = True` di `main.py` (memerlukan waktu lebih lama ~5–15 menit).

### Langkah 2 — Jalankan Aplikasi Web

```bash
python app.py
```

Buka browser dan akses:

```
http://127.0.0.1:5000
```

### Langkah 3 — Prediksi

1. Upload gambar daun (PNG / JPG / JPEG)
2. Klik tombol **Analisis Gambar**
3. Hasil prediksi beserta confidence score per kelas akan ditampilkan

---

## Arsitektur Model

```
Gambar Daun (input)
        │
        ▼
┌───────────────────┐
│   Preprocessing   │  resize 128×128, CLAHE, Gaussian Blur, Morphological
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Extraction│  HOG + LBP + HSV Histogram + LAB Color + Dark Ratio
└────────┬──────────┘
         │  ~8200 fitur
         ▼
┌───────────────────┐
│   RobustScaler    │  normalisasi fitur
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   PCA (95% var)   │  reduksi dimensi
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   SVM RBF Kernel  │  C=10, gamma='scale', class_weight='balanced'
└────────┬──────────┘
         │
         ▼
  Sehat / Berlubang / Kering
```

---

## Penjelasan Fitur Ekstraksi

### HOG (Histogram of Oriented Gradients)
Mendeteksi bentuk dan tepi daun. Lubang pada daun menghasilkan tepi yang tajam dan berbeda dari daun utuh.

### LBP (Local Binary Pattern)
Mendeteksi pola tekstur permukaan daun. Daun kering memiliki tekstur berbeda dari daun sehat.

### HSV Color Histogram
Menganalisis distribusi warna dalam ruang Hue-Saturation-Value. Sangat efektif membedakan daun hijau (sehat) dari daun coklat (kering).

### LAB Color Space
Ruang warna yang mendekati persepsi manusia. Digunakan untuk mendeteksi perubahan warna yang halus pada daun kering.

### Dark Region Ratio
Proporsi piksel sangat gelap (nilai < 50). Lubang pada daun tampak gelap/transparan sehingga fitur ini tinggi untuk daun berlubang.

---

## Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| Akurasi | 94.50% |
| CV F1 Score | 0.9745 ± 0.0078 |
| Precision (rata-rata) | 95% |
| Recall (rata-rata) | 95% |

### Confusion Matrix

```
                Predicted
              Sehat  Berlubang  Kering
Actual Sehat  [ 186 ]  [ 10 ]   [  4 ]
    Berlubang [   8 ]  [185 ]   [  7 ]
       Kering [   3 ]  [  9 ]   [188 ]
```

---

## Konfigurasi Model

Edit `svm_model.py` untuk mengubah parameter:

```python
SVC(
    kernel='rbf',              # kernel RBF untuk data non-linear
    C=10,                      # toleransi kesalahan (1–200)
    gamma='scale',             # radius pengaruh tiap titik
    probability=True,          # aktifkan confidence score
    class_weight='balanced',   # tangani ketidakseimbangan kelas
    decision_function_shape='ovo'  # One-vs-One untuk multiclass
)
```

Untuk tuning otomatis dengan GridSearchCV:

```python
# main.py
USE_TUNING = True  # aktifkan GridSearchCV
```

---

## API Endpoint

### `GET /`
Menampilkan halaman utama aplikasi.

### `POST /predict`

**Request:** `multipart/form-data` dengan field `file` berisi gambar daun.

**Response:**

```json
{
  "result": "Sehat",
  "confidence": "96.42%",
  "detail": {
    "Sehat": "96.42%",
    "Berlubang": "2.31%",
    "Kering": "1.27%"
  }
}
```

**Error Response:**

```json
{
  "error": "Gambar tidak valid"
}
```

---

## Troubleshooting

### Model tidak ditemukan
```
FileNotFoundError: Model tidak ditemukan: model/svm_model.pkl
```
Jalankan `python main.py` terlebih dahulu untuk melatih dan menyimpan model.

### Folder dataset tidak ditemukan
```
[ERROR] Folder tidak ditemukan: dataset/Sehat
```
Pastikan nama folder di `dataset/` persis sama dengan `categories` di `main.py` (case-sensitive).

### Akurasi rendah
- Tambah jumlah data (minimal 500 per kelas)
- Aktifkan `USE_TUNING = True` untuk GridSearchCV
- Pastikan kualitas gambar dataset konsisten (pencahayaan, latar belakang)

---

## Lisensi

Proyek ini dibuat untuk keperluan akademik dan penelitian.

---

## Kontributor

Dikembangkan sebagai proyek klasifikasi citra daun menggunakan Support Vector Machine (SVM) dengan kernel RBF.