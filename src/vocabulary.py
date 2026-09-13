# Receive descriptors from collect_training_descriptors(...)
# Then fit into MiniBatchKMeans

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
import sklearn.cluster as cluster
import joblib

def build_vocabulary(descriptors, vocab_size = config.DEFAULT_VOCAB_SIZE):
    """
    Build vocabulary from descriptors using MiniBatchKMeans
    return:
    - kmeans: MiniBatchKMeans model
    """
    initial = cluster.MiniBatchKMeans(
        n_clusters=vocab_size, 
        batch_size=2048,
        random_state=config.RANDOM_STATE,
        n_init="auto"
    )
    kmeans_model = initial.fit(descriptors)
    return kmeans_model

def save_vocabulary(kmeans, vocab_size = config.DEFAULT_VOCAB_SIZE):
    os.makedirs(config.VOCAB_DIR, exist_ok=True)
    path = os.path.join(config.VOCAB_DIR, f"vocab_{vocab_size}.joblib")
    joblib.dump(kmeans, path)
    print(f"Vocabulary saved to {path}")



def load_vocabulary(vocab_size = config.DEFAULT_VOCAB_SIZE):
    path = os.path.join(config.VOCAB_DIR, f"vocab_{vocab_size}.joblib")
    print(f"Loading vocabulary from {path}")
    return joblib.load(path)
