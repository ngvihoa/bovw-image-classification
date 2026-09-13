# Script for traning _ connect all modules to become a complete process

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataset import load_dataset
from src.sift import collect_training_descriptors
from src.vocabulary import build_vocabulary, save_vocabulary
from src.bovw import build_features, save_features
from src.classifier import train_classifier, save_model
from src.evaluate import compute_metrics, save_metrics, plot_confusion_matrix

import config
import argparse

def main(vocab_size = config.DEFAULT_VOCAB_SIZE):
    print(f"Starting training process with vocab_size={vocab_size}")
    # 1. Load dataset
    train_dataset, test_dataset, train_labels, test_labels = load_dataset()
    # 2. Features extraction with SIFT
    descriptors = collect_training_descriptors(train_dataset)
    # 3. Build vocabulary with MiniBatchKMeans
    kmeans_model = build_vocabulary(descriptors, vocab_size=vocab_size)
    save_vocabulary(kmeans_model, vocab_size=vocab_size)
    # 4. Build BOVW features
    train_features = build_features(train_dataset, kmeans_model, vocab_size=vocab_size)
    save_features(train_features, 'train', vocab_size=vocab_size)
    test_features = build_features(test_dataset, kmeans_model, vocab_size=vocab_size)
    save_features(test_features, 'test', vocab_size=vocab_size)
    # 5. Train classifier with SVM
    classifier = train_classifier(train_features, train_labels)
    save_model(classifier, vocab_size=vocab_size)
    # 6. Evaluate model
    test_prediction = classifier.predict(test_features)
    metrics = compute_metrics(test_labels, test_prediction)
    save_metrics(metrics, vocab_size=vocab_size)
    plot_confusion_matrix(test_labels, test_prediction, class_names=config.CLASSES, vocab_size=vocab_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Classification with SIFT + BOVW + SVM")
    parser.add_argument("--vocab_size", type=int, default=config.DEFAULT_VOCAB_SIZE)
    args=parser.parse_args()
    main(vocab_size=args.vocab_size)
