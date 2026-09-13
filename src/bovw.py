# Transform an image (descriptors) to features vector base on trained vocabulary


import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.cluster import MiniBatchKMeans
import config
import src.sift as sift
import numpy as np

def image_to_bovw(descriptors, kmeans: MiniBatchKMeans, vocab_size = config.DEFAULT_VOCAB_SIZE):
    """
    Transform an image (descriptors) to features vector base on trained vocabulary
    return:
    - features: np.ndarray of shape (vocab_size,)
    """
    if descriptors is None:
        return np.zeros(vocab_size, dtype=np.float32)

    word_ids = kmeans.predict(descriptors) # shape [(n_words, __)]
    features = np.bincount(word_ids, minlength=vocab_size).astype(np.float32)
    # normalize by L2 norm
    normalize = features / (np.linalg.norm(features) + 1e-8 )
    return normalize

def build_features(images_path, kmeans: MiniBatchKMeans, vocab_size = config.DEFAULT_VOCAB_SIZE):
    features = []
    for idx, image_path in enumerate(images_path):
        descriptors = sift.extract_sift(image_path)
        histogram = image_to_bovw(descriptors, kmeans, vocab_size)
        features.append(histogram)
    return np.array(features, dtype=np.float32)

def save_features(features, split, vocab_size = config.DEFAULT_VOCAB_SIZE):
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    path = os.path.join(config.FEATURES_DIR, f"features_{split}_k{vocab_size}.npy")
    np.save(path, features)
    print(f"Features saved to {path}")


def load_features(split, vocab_size = config.DEFAULT_VOCAB_SIZE):
    path = os.path.join(config.FEATURES_DIR, f"features_{split}_k{vocab_size}.npy")
    print(f"Loading features from {path}")
    return np.load(path)

    