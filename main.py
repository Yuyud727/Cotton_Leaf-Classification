import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from model.svm_model import train_model, save_model
from preprocessing.preprocess import preprocess_image
from feature_extraction.features import extract_features
from model.svm_model import train_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# dataset
dataset_path = "dataset"
categories = ["crack", "good"]

data = []

# load data
for category in categories:
    folder = os.path.join(dataset_path, category)
    
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            data.append((img, category))

print("Total data:", len(data))

# preprocessing
processed_data = []

for img, label in data:
    try:
        img_processed = preprocess_image(img)
        processed_data.append((img_processed, label))
    except:
        pass

print("Data setelah preprocessing:", len(processed_data))

# feature extraction
X = []
y = []

for img, label in processed_data:
    features = extract_features(img)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Shape X:", X.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# train
model = train_model(X_train, y_train)

# simpan model
save_model(model)
print("Model berhasil disimpan!")

# predict
y_pred = model.predict(X_test)

# evaluasi
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# visualisasi
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()