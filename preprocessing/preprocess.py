import cv2

IMG_SIZE = 100

def preprocess_image(img):
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_gray