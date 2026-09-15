#!/usr/bin/env python3
"""Train and evaluate SIFT + BoVW + SVM on Kaggle.

The script is self-contained: it only needs a Caltech-101 dataset attached to
the Kaggle notebook. It automatically searches /kaggle/input for the directory
that contains the configured class folders.

Examples (run from a Kaggle notebook cell):

    !python scripts/kaggle_train.py
    !python scripts/kaggle_train.py --vocab-sizes 100 500
    !python scripts/kaggle_train.py --data-dir /kaggle/input/caltech101/101_ObjectCategories

Outputs are written to /kaggle/working/bovw_output by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

# Kaggle and restricted containers may not allow writing to the default config path.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


DEFAULT_CLASSES = ("airplanes", "Motorbikes", "Faces", "watch", "car_side")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Image classification on Kaggle with SIFT + BoVW + linear SVM"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the class folders. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/kaggle/input"),
        help="Root searched when --data-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/bovw_output"),
        help="Directory for trained artifacts and evaluation results.",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--vocab-sizes", nargs="+", type=int, default=[50, 100, 200, 500]
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-descriptors-per-image", type=int, default=200)
    parser.add_argument("--max-total-descriptors", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--sift-nfeatures",
        type=int,
        default=0,
        help="Maximum SIFT features per image; 0 lets OpenCV keep all features.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional image limit per class for a quick test; 0 uses all images.",
    )
    return parser.parse_args()


def find_dataset(input_root: Path, classes: Sequence[str]) -> Path:
    """Find a directory whose direct children include all requested classes."""
    if not input_root.exists():
        raise FileNotFoundError(
            f"Input root does not exist: {input_root}. "
            "Attach Caltech-101 to the notebook or pass --data-dir."
        )

    required = set(classes)
    for current, directories, _files in os.walk(input_root):
        if required.issubset(set(directories)):
            return Path(current)
    raise FileNotFoundError(
        f"Could not find folders {list(classes)} below {input_root}. "
        "Pass the exact directory with --data-dir."
    )


def resolve_data_dir(args: argparse.Namespace) -> Path:
    data_dir = args.data_dir or find_dataset(args.input_root, args.classes)
    data_dir = data_dir.resolve()
    missing = [name for name in args.classes if not (data_dir / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folders under {data_dir}: {', '.join(missing)}"
        )
    return data_dir


def load_image_paths(
    data_dir: Path, classes: Sequence[str], limit_per_class: int
) -> tuple[list[Path], np.ndarray, dict[str, int]]:
    paths: list[Path] = []
    labels: list[int] = []
    counts: dict[str, int] = {}

    for label, class_name in enumerate(classes):
        class_paths = sorted(
            path
            for path in (data_dir / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class > 0:
            class_paths = class_paths[:limit_per_class]
        if len(class_paths) < 2:
            raise ValueError(
                f"Class '{class_name}' has only {len(class_paths)} usable image(s)."
            )
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))
        counts[class_name] = len(class_paths)

    return paths, np.asarray(labels, dtype=np.int64), counts


def show_progress(items: Iterable, description: str) -> Iterable:
    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=description)
    except ImportError:
        print(description)
        return items


def extract_descriptors(
    paths: Sequence[Path], sift_nfeatures: int
) -> tuple[list[np.ndarray], int]:
    try:
        sift = cv2.SIFT_create(nfeatures=sift_nfeatures)
    except AttributeError as error:
        raise RuntimeError(
            "This OpenCV build has no SIFT. Install a recent opencv-python package."
        ) from error

    all_descriptors: list[np.ndarray] = []
    unreadable = 0
    for path in show_progress(paths, "Extracting SIFT"):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            unreadable += 1
            all_descriptors.append(np.empty((0, 128), dtype=np.float32))
            continue
        _keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        all_descriptors.append(descriptors.astype(np.float32, copy=False))
    return all_descriptors, unreadable


def sample_training_descriptors(
    descriptors: Sequence[np.ndarray],
    train_indices: np.ndarray,
    max_per_image: int,
    max_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for index in train_indices:
        current = descriptors[int(index)]
        if len(current) == 0:
            continue
        if len(current) > max_per_image:
            selected = rng.choice(len(current), size=max_per_image, replace=False)
            current = current[selected]
        sampled.append(current)

    if not sampled:
        raise RuntimeError("No SIFT descriptors were found in the training split.")

    pool = np.vstack(sampled).astype(np.float32, copy=False)
    if len(pool) > max_total:
        selected = rng.choice(len(pool), size=max_total, replace=False)
        pool = pool[selected]
    return pool


def descriptors_to_bovw(
    descriptors: Sequence[np.ndarray],
    indices: np.ndarray,
    vocabulary: MiniBatchKMeans,
    vocab_size: int,
    description: str,
) -> np.ndarray:
    features = np.zeros((len(indices), vocab_size), dtype=np.float32)
    for row, index in enumerate(show_progress(indices, description)):
        current = descriptors[int(index)]
        if len(current) == 0:
            continue
        word_ids = vocabulary.predict(current)
        histogram = np.bincount(word_ids, minlength=vocab_size).astype(np.float32)
        features[row] = histogram / (np.linalg.norm(histogram) + 1e-8)
    return features


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    vocab_size: int,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    figure, axis = plt.subplots(figsize=(9, 7))
    display = ConfusionMatrixDisplay(matrix, display_labels=classes)
    display.plot(ax=axis, cmap="Blues", colorbar=True, xticks_rotation=35)
    axis.set_title(f"Confusion matrix (K={vocab_size})")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_predictions(
    output_path: Path,
    paths: Sequence[Path],
    indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_label", "predicted_label", "correct"])
        for index, true_label, predicted_label in zip(indices, y_true, y_pred):
            writer.writerow(
                [
                    str(paths[int(index)]),
                    classes[int(true_label)],
                    classes[int(predicted_label)],
                    bool(true_label == predicted_label),
                ]
            )


def run_experiment(
    vocab_size: int,
    descriptor_pool: np.ndarray,
    all_descriptors: Sequence[np.ndarray],
    paths: Sequence[Path],
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, float | int]:
    if vocab_size <= 1:
        raise ValueError(f"Vocabulary size must be greater than 1, got {vocab_size}.")
    if vocab_size > len(descriptor_pool):
        raise ValueError(
            f"K={vocab_size} exceeds the {len(descriptor_pool)} sampled descriptors."
        )

    experiment_dir = args.output_dir / f"k{vocab_size}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    print(f"\n{'=' * 68}\nTraining vocabulary with K={vocab_size}")
    vocabulary = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=args.batch_size,
        random_state=args.seed,
        n_init=10,
    )
    vocabulary.fit(descriptor_pool)

    train_features = descriptors_to_bovw(
        all_descriptors,
        train_indices,
        vocabulary,
        vocab_size,
        f"BoVW train K={vocab_size}",
    )
    test_features = descriptors_to_bovw(
        all_descriptors,
        test_indices,
        vocabulary,
        vocab_size,
        f"BoVW test K={vocab_size}",
    )

    classifier = SVC(
        kernel="linear", C=args.svm_c, random_state=args.seed
    ).fit(train_features, labels[train_indices])
    predictions = classifier.predict(test_features)
    metrics = calculate_metrics(labels[test_indices], predictions)
    elapsed = time.perf_counter() - start
    result: dict[str, float | int] = {
        "vocab_size": vocab_size,
        **metrics,
        "elapsed_seconds": round(elapsed, 3),
    }

    joblib.dump(vocabulary, experiment_dir / "vocabulary.joblib")
    joblib.dump(classifier, experiment_dir / "svm_model.joblib")
    np.save(experiment_dir / "features_train.npy", train_features)
    np.save(experiment_dir / "features_test.npy", test_features)
    with (experiment_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    with (experiment_dir / "classification_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(
            classification_report(
                labels[test_indices],
                predictions,
                labels=np.arange(len(args.classes)),
                target_names=args.classes,
                digits=4,
                zero_division=0,
            )
        )
    save_confusion_plot(
        labels[test_indices],
        predictions,
        args.classes,
        vocab_size,
        experiment_dir / "confusion_matrix.png",
    )
    save_predictions(
        experiment_dir / "predictions.csv",
        paths,
        test_indices,
        labels[test_indices],
        predictions,
        args.classes,
    )

    print(
        f"K={vocab_size}: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['f1_macro']:.4f}, time={elapsed:.1f}s"
    )
    return result


def save_summary(results: Sequence[dict[str, float | int]], output_dir: Path) -> None:
    fieldnames = [
        "vocab_size",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "elapsed_seconds",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_data_dir(args)
    paths, labels, class_counts = load_image_paths(
        data_dir, args.classes, args.limit_per_class
    )
    indices = np.arange(len(paths))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    print(f"Dataset: {data_dir}")
    print(f"Classes: {class_counts}")
    print(f"Train images: {len(train_indices)} | Test images: {len(test_indices)}")
    print(f"Output: {args.output_dir.resolve()}")

    all_descriptors, unreadable = extract_descriptors(paths, args.sift_nfeatures)
    descriptor_pool = sample_training_descriptors(
        all_descriptors,
        train_indices,
        args.max_descriptors_per_image,
        args.max_total_descriptors,
        rng,
    )
    print(
        f"Vocabulary descriptor pool: {descriptor_pool.shape}; "
        f"unreadable images: {unreadable}"
    )

    metadata = {
        "data_dir": str(data_dir),
        "classes": args.classes,
        "class_counts": class_counts,
        "total_images": len(paths),
        "train_images": len(train_indices),
        "test_images": len(test_indices),
        "test_size": args.test_size,
        "seed": args.seed,
        "vocab_sizes": args.vocab_sizes,
        "max_descriptors_per_image": args.max_descriptors_per_image,
        "max_total_descriptors": args.max_total_descriptors,
        "sampled_descriptor_count": len(descriptor_pool),
        "sift_nfeatures": args.sift_nfeatures,
        "svm_c": args.svm_c,
        "opencv_version": cv2.__version__,
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    results = [
        run_experiment(
            vocab_size,
            descriptor_pool,
            all_descriptors,
            paths,
            labels,
            train_indices,
            test_indices,
            args,
        )
        for vocab_size in args.vocab_sizes
    ]
    save_summary(results, args.output_dir)

    best = max(results, key=lambda item: float(item["f1_macro"]))
    print(f"\nFinished. Summary: {args.output_dir / 'summary.csv'}")
    print(
        f"Best macro F1: K={best['vocab_size']}, "
        f"accuracy={best['accuracy']:.4f}, macro_f1={best['f1_macro']:.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
