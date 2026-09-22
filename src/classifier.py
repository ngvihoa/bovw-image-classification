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


# ==============================================================================
# MODULE CHO MÔ HÌNH V2: ADDITIVE CHI-SQUARE (CHI2) KERNEL SVM
# ==============================================================================
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import AdditiveChi2Sampler
import numpy as np


def train_chi2_svm(X_train_spm, y_train, C: float = 1.0, seed: int = config.RANDOM_STATE, sample_steps: int = 2):
    """
    Biến đổi không gian đặc trưng SPM sang Additive Chi2 và huấn luyện LinearSVC.
    """
    chi2_sampler = AdditiveChi2Sampler(sample_steps=sample_steps)
    X_train_chi2 = chi2_sampler.fit_transform(np.maximum(0, X_train_spm))
    clf = LinearSVC(C=C, random_state=seed, max_iter=2000, dual='auto')
    clf.fit(X_train_chi2, y_train)
    return clf, chi2_sampler


def predict_chi2_svm(model: LinearSVC, chi2_sampler: AdditiveChi2Sampler, X_spm):
    """
    Dự đoán nhãn cho vector đặc trưng SPM bằng mô hình Chi2-SVM v2.
    """
    X_chi2 = chi2_sampler.transform(np.maximum(0, X_spm))
    return model.predict(X_chi2)


def save_chi2_model(model: LinearSVC, chi2_sampler: AdditiveChi2Sampler, tfidf_model=None, vocab_size: int = config.DEFAULT_VOCAB_SIZE):
    """Lưu gói mô hình v2 gồm LinearSVC, chi2_sampler và tfidf_model."""
    os.makedirs(config.CLASSIFIERS_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "chi2_sampler": chi2_sampler,
        "tfidf_model": tfidf_model
    }
    path = os.path.join(config.CLASSIFIERS_DIR, f"chi2_svm_bundle_k{vocab_size}.joblib")
    joblib.dump(bundle, path)
    print(f"[v2] Đã lưu Chi2-SVM Bundle vào {path}")


def load_chi2_model(vocab_size: int = config.DEFAULT_VOCAB_SIZE):
    """Nạp gói mô hình v2."""
    path = os.path.join(config.CLASSIFIERS_DIR, f"chi2_svm_bundle_k{vocab_size}.joblib")
    print(f"[v2] Nạp Chi2-SVM Bundle từ {path}")
    return joblib.load(path)


