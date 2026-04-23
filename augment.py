# augment.py - tambahkan sebelum training di main.py
import cv2
import numpy as np

def augment_image(img):
    augmented = [img]

    # Flip horizontal & vertikal
    augmented.append(cv2.flip(img, 1))
    augmented.append(cv2.flip(img, 0))

    # Rotasi
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
        augmented.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

    # Brightness variation
    bright = np.clip(img.astype(np.int32) + 30, 0, 255).astype(np.uint8)
    dark   = np.clip(img.astype(np.int32) - 30, 0, 255).astype(np.uint8)
    augmented.extend([bright, dark])

    return augmented


# Di main.py, ganti bagian preprocessing:
processed_data = []
for img, label in data:
    try:
        img_processed = preprocess_image(img)
        for aug_img in augment_image(img_processed):   # tiap gambar jadi 7x
            processed_data.append((aug_img, label))
    except Exception as e:
        print(f"[WARNING] Preprocessing gagal: {e}")