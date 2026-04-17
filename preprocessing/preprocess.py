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
    
    return img_blur