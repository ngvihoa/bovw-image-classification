# This file contains all constants use for this project

import os

RANDOM_STATE  = 42
DATA_DIR = 'data/raw/101_ObjectCategories'
CLASSES = ["airplanes", "Motorbikes", "Faces", "watch", "car_side"]

TEST_SIZE = 0.2 

DEFAULT_VOCAB_SIZE = 100
VOCAB_SIZES = [50, 100, 200, 500]

MAX_DESCRIPTOR_PER_IMAGE = 200
MAX_TOTAL_DESCRIPTORS = 100_000 

# folder to save artifacts and results
ARTIFACT_DIR = "artifacts"
VOCAB_DIR = os.path.join(ARTIFACT_DIR, "vocabularies")
FEATURES_DIR = os.path.join(ARTIFACT_DIR, "features")
CLASSIFIERS_DIR = os.path.join(ARTIFACT_DIR, "classifiers")

RESULTS_DIR = "results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CONFUSION_DIR = os.path.join(RESULTS_DIR, "confusion")