# Script for training Model v2 (SIFT + SPM 3-level + Additive Chi2 SVM)
# Connects all modernized modules in src/ into a complete, reproducible pipeline.

import os
import sys
import time
import argparse
import numpy as np

# Thêm thư mục gốc dự án vào sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from src.dataset import load_caltech_benchmark, load_dataset, auto_detect_dataset
from src.sift import collect_training_descriptors
from src.vocabulary import build_vocabulary, save_vocabulary
from src.spm import extract_all_spm, apply_tfidf_weighting, save_spm_features
from src.classifier import train_chi2_svm, predict_chi2_svm, save_chi2_model
from src.evaluate import compute_metrics, save_metrics, plot_confusion_matrix


def main(
    vocab_size: int = 500,
    train_per_class: int = 24,
    test_per_class: int = 6,
    data_dir: str = None,
    use_tfidf: bool = True,
    c_param: float = 1.0,
    max_descriptors: int = 100_000
):
    print("=" * 75)
    print("KHỞI CHẠY QUY TRÌNH HUẤN LUYỆN MODEL V2 (SPM 3 TẦNG + ADDITIVE CHI-SQUARE SVM)")
    print(f"• Vocab Size (K): {vocab_size}")
    print(f"• Tỷ lệ phân chia: {train_per_class} Train : {test_per_class} Test mỗi lớp")
    print(f"• TF-IDF Weighting: {'BẬT' if use_tfidf else 'TẮT'}")
    print(f"• C Parameter: {c_param}")
    print("=" * 75)

    start_total_t = time.time()

    # 1. Nạp dữ liệu
    print("\n[Bước 1/6] Nạp dữ liệu...")
    try:
        # Ưu tiên load benchmark 101 lớp nếu có thư mục dữ liệu
        X_train, X_test, y_train, y_test, class_names = load_caltech_benchmark(
            data_dir=data_dir,
            train_per_class=train_per_class,
            max_test_per_class=test_per_class
        )
        print(f"-> Đã nạp thành công {len(class_names)} lớp từ Caltech-101 Benchmark.")
    except Exception as e:
        print(f"-> Chú ý ({e}), chuyển sang nạp tập dữ liệu mặc định từ config...")
        train_data, test_data, y_train, y_test = load_dataset()
        X_train, X_test = train_data, test_data
        class_names = config.CLASSES
        print(f"-> Đã nạp thành công {len(class_names)} lớp từ config.CLASSES.")

    print(f"-> Số ảnh Train: {len(X_train)} | Số ảnh Test: {len(X_test)}")

    # 2. Thu thập SIFT descriptors huấn luyện từ điển
    print("\n[Bước 2/6] Thu thập SIFT descriptors huấn luyện...")
    descriptors = collect_training_descriptors(
        X_train, 
        max_total_descriptors=max_descriptors
    )
    print(f"-> Thu được: {len(descriptors):,} descriptors.")

    # 3. Xây dựng Visual Vocabulary bằng MiniBatchKMeans
    print(f"\n[Bước 3/6] Xây dựng Từ điển thị giác (K={vocab_size})...")
    kmeans_model = build_vocabulary(descriptors, vocab_size=vocab_size)
    save_vocabulary(kmeans_model, vocab_size=vocab_size)

    # 4. Trích xuất đặc trưng SPM 3 tầng (1x1, 2x2, 4x4 -> 21 * K chiều)
    print(f"\n[Bước 4/6] Trích xuất Kim tự tháp không gian SPM 3 tầng (21 x {vocab_size} = {21 * vocab_size:,} chiều)...")
    t_spm = time.time()
    train_features = extract_all_spm(X_train, kmeans_model, vocab_size=vocab_size)
    test_features = extract_all_spm(X_test, kmeans_model, vocab_size=vocab_size)
    print(f"-> Hoàn tất trích xuất SPM trong {time.time() - t_spm:.2f}s.")

    if use_tfidf:
        print("-> Đang áp dụng trọng số TF-IDF lọc nhiễu nền...")
        train_features, test_features, _ = apply_tfidf_weighting(train_features, test_features)

    # Lưu ma trận đặc trưng
    save_spm_features(train_features, 'train', vocab_size=vocab_size)
    save_spm_features(test_features, 'test', vocab_size=vocab_size)

    # 5. Huấn luyện bộ phân loại Additive Chi2 Linear SVM
    print(f"\n[Bước 5/6] Huấn luyện Additive Chi-Square (χ²) Kernel SVM...")
    t_svm = time.time()
    chi2_pipeline = train_chi2_svm(train_features, y_train, C=c_param)
    save_chi2_model(chi2_pipeline, vocab_size=vocab_size)
    print(f"-> Huấn luyện SVM hoàn tất sau {time.time() - t_svm:.2f}s.")

    # 6. Đánh giá và báo cáo hiệu năng
    print("\n[Bước 6/6] Dự đoán và Đánh giá hiệu năng...")
    test_prediction = predict_chi2_svm(chi2_pipeline, test_features)
    metrics = compute_metrics(y_test, test_prediction)

    # Lưu metrics và biểu đồ nếu số lớp nhỏ
    save_metrics(metrics, vocab_size=vocab_size)
    if len(class_names) <= 20:
        plot_confusion_matrix(y_test, test_prediction, class_names=class_names, vocab_size=vocab_size)

    total_time = time.time() - start_total_t

    print("\n" + "=" * 75)
    print("KẾT QUẢ THỰC NGHIỆM MODEL V2 (SPM + CHI2 SVM):")
    print(f"• Accuracy (Độ chính xác) : {metrics['accuracy'] * 100:.2f}%")
    print(f"• Macro F1-Score          : {metrics['f1_score'] * 100:.2f}%")
    print(f"• Macro Precision         : {metrics['precision'] * 100:.2f}%")
    print(f"• Macro Recall            : {metrics['recall'] * 100:.2f}%")
    print(f"• Tổng thời gian thực thi : {total_time:.1f} giây")
    print("=" * 75)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Model v2: SIFT + SPM 3 Tầng + Additive Chi2 SVM")
    parser.add_argument("--vocab_size", type=int, default=500, help="Kích thước từ điển K (mặc định: 500)")
    parser.add_argument("--train_per_class", type=int, default=24, help="Số ảnh train mỗi lớp (mặc định: 24)")
    parser.add_argument("--test_per_class", type=int, default=6, help="Số ảnh test mỗi lớp (mặc định: 6)")
    parser.add_argument("--data_dir", type=str, default=None, help="Đường dẫn thư mục tập dữ liệu")
    parser.add_argument("--no_tfidf", action="store_true", help="Tắt trọng số TF-IDF")
    parser.add_argument("--c_param", type=float, default=1.0, help="Hệ số phạt C của SVM (mặc định: 1.0)")
    parser.add_argument("--max_descriptors", type=int, default=100_000, help="Số descriptor tối đa để học K-Means")

    args = parser.parse_args()

    main(
        vocab_size=args.vocab_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        data_dir=args.data_dir,
        use_tfidf=not args.no_tfidf,
        c_param=args.c_param,
        max_descriptors=args.max_descriptors
    )

