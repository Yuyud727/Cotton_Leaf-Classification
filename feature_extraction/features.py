import numpy as np
from skimage.feature import hog, local_binary_pattern

def extract_features(img):
    # HOG - deteksi tepi dan struktur retakan
    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    # LBP - deteksi tekstur permukaan
    lbp = local_binary_pattern(img, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=10,
        range=(0, 10),
        density=True
    )

    # Gabungkan HOG + LBP
    return np.concatenate([hog_features, lbp_hist])