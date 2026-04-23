import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # CLAHE - normalisasi pencahayaan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_eq = clahe.apply(img_gray)

    # Gaussian blur - kurangi noise
    img_blur = cv2.GaussianBlur(img_eq, (3, 3), 0)

    # Morphological opening - perjelas batas lubang
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_morph = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel)

    return img_morph