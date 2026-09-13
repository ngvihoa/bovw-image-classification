# Use SVM for classifier - train SVM, save/load model

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import sklearn.svm as svm
import joblib

def train_classifier(x_train, y_train):
    model = svm.SVC(kernel='linear', C=1.0, random_state=config.RANDOM_STATE)
    model.fit(x_train, y_train)
    return model

def save_model(model: svm.SVC, vocab_size = config.DEFAULT_VOCAB_SIZE):
    os.makedirs(config.CLASSIFIERS_DIR, exist_ok=True)
    path = os.path.join(config.CLASSIFIERS_DIR, f"svm_model_k{vocab_size}.joblib")
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(vocab_size = config.DEFAULT_VOCAB_SIZE):
    path = os.path.join(config.CLASSIFIERS_DIR, f"svm_model_k{vocab_size}.joblib")
    print(f"Loading model from {path}")
    return joblib.load(path)

