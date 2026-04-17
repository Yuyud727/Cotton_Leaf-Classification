import pickle
import os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_model(X_train, y_train):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced'
        ))
    ])
    model.fit(X_train, y_train)
    return model

def save_model(model, filename="model/svm_model.pkl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename="model/svm_model.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)