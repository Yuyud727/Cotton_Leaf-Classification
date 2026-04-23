import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel

def extract_features(img_gray):
    # ── HOG ──────────────────────────────────────────────
    hog_features = hog(
        img_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # ── LBP ──────────────────────────────────────────────
    lbp = local_binary_pattern(img_gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10), density=True)

    # ── Dark region (lubang) ──────────────────────────────
    dark_ratio      = np.mean(img_gray < 50)
    very_dark_ratio = np.mean(img_gray < 30)

    # ── Edge density ──────────────────────────────────────
    edges      = sobel(img_gray.astype(np.float64))
    edge_mean  = np.mean(edges)
    edge_std   = np.std(edges)

    # ── Intensity histogram ───────────────────────────────
    intensity_hist, _ = np.histogram(img_gray.ravel(), bins=32, range=(0, 256), density=True)

    extra = [dark_ratio, very_dark_ratio, edge_mean, edge_std]
    return np.concatenate([hog_features, lbp_hist, intensity_hist, extra])


def extract_color_features(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # ← tambah LAB untuk deteksi kering

    features = []

    # HSV histogram per channel
    for i in range(3):
        hist, _ = np.histogram(hsv[:, :, i], bins=16, density=True)
        features.extend(hist)
        features.append(np.mean(hsv[:, :, i]))
        features.append(np.std(hsv[:, :, i]))

    # LAB histogram per channel (bagus untuk deteksi warna coklat/kering)
    for i in range(3):
        hist, _ = np.histogram(lab[:, :, i], bins=16, density=True)
        features.extend(hist)
        features.append(np.mean(lab[:, :, i]))
        features.append(np.std(lab[:, :, i]))

    # Rasio warna coklat/kuning (indikator daun kering)
    # Daun kering: Hue 10-30, Saturation rendah
    hue = hsv[:, :, 0]
    brown_ratio  = np.mean((hue >= 10) & (hue <= 30))  # coklat/oranye
    yellow_ratio = np.mean((hue >= 20) & (hue <= 40))  # kuning
    green_ratio  = np.mean((hue >= 35) & (hue <= 85))  # hijau

    features.extend([brown_ratio, yellow_ratio, green_ratio])

    return np.array(features)