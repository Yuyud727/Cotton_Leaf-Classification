import pickle
from sklearn.svm import SVC

def train_model(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def save_model(model, filename="model/svm_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename="model/svm_model.pkl"):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model