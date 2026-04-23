import pickle
import os
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def train_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA(n_components=0.95)),
        ('svm', SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            decision_function_shape='ovo'  # ← One-vs-One untuk 3 kelas
        ))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_model_tuned(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('pca', PCA()),
        ('svm', SVC(
            kernel='rbf',
            probability=True,
            class_weight='balanced',
            decision_function_shape='ovo'
        ))
    ])

    param_grid = {
        'pca__n_components': [0.90, 0.95, 0.99],
        'svm__C':            [1, 10, 50, 100],
        'svm__gamma':        ['scale', 'auto', 0.001, 0.01],
    }

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv,
                        scoring='f1_macro', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print(f"Best params : {grid.best_params_}")
    print(f"Best CV F1  : {grid.best_score_:.4f}")
    return grid.best_estimator_

def save_model(model, filename="model/svm_model.pkl"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename="model/svm_model.pkl"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Model tidak ditemukan: {filename}")
    with open(filename, 'rb') as f:
        return pickle.load(f)