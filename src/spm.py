# Spatial Pyramid Matching (SPM) module for Model v2
# Implements multi-level spatial grid histogram pooling (Lazebnik et al., CVPR 2006)
# Level 0 (1x1, w=0.25) + Level 1 (2x2, w=0.25) + Level 2 (4x4, w=0.50) -> 21 cells

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import config
from src.sift import extract_sift_spm


def compute_spm_features(desc, coords, kmeans_model: MiniBatchKMeans, vocab_size: int = 500):
    """
    Biểu diễn đặc trưng Spatial Pyramid Matching 3 tầng (1x1, 2x2 và 4x4) theo chuẩn Lazebnik (2006).
    
    Tham số:
    - desc: np.ndarray shape (N, 128) - tập SIFT descriptors của 1 ảnh
    - coords: np.ndarray shape (N, 2) - tọa độ chuẩn hóa (x, y) trong [0, 1]
    - kmeans_model: bộ từ điển MiniBatchKMeans đã huấn luyện
    - vocab_size: kích thước từ điển K
    
    Trả về:
    - combined: vector đặc trưng 1D kích thước (21 * vocab_size,) đã qua Power & L2 norm
    """
    total_cells = 21  # 1 + 4 + 16
    if desc is None or len(desc) == 0:
        return np.zeros(vocab_size * total_cells, dtype=np.float32)
        
    words = kmeans_model.predict(desc.astype(np.float32))
    spm_feature = []
    
    # --- Tầng 0: Toàn bộ ảnh 1x1 (Trọng số w = 0.25) ---
    hist_l0 = np.zeros(vocab_size, dtype=np.float32)
    counts_l0 = Counter(words)
    for w_idx, count in counts_l0.items():
        hist_l0[w_idx] = count
    norm_l0 = np.linalg.norm(hist_l0)
    if norm_l0 > 0:
        hist_l0 /= norm_l0
    spm_feature.append(hist_l0 * 0.25)
    
    # --- Tầng 1: Lưới 2x2 gồm 4 ô không gian (Trọng số w = 0.25) ---
    for r in range(2):
        for c in range(2):
            mask = (
                (coords[:, 0] >= c * 0.5) & (coords[:, 0] < (c + 1) * 0.5) &
                (coords[:, 1] >= r * 0.5) & (coords[:, 1] < (r + 1) * 0.5)
            )
            sub_words = words[mask]
            hist_sub = np.zeros(vocab_size, dtype=np.float32)
            if len(sub_words) > 0:
                counts_sub = Counter(sub_words)
                for w_idx, count in counts_sub.items():
                    hist_sub[w_idx] = count
            norm_sub = np.linalg.norm(hist_sub)
            if norm_sub > 0:
                hist_sub /= norm_sub
            spm_feature.append(hist_sub * 0.25)
            
    # --- Tầng 2: Lưới 4x4 gồm 16 ô không gian (Trọng số w = 0.50) ---
    for r in range(4):
        for c in range(4):
            mask = (
                (coords[:, 0] >= c * 0.25) & (coords[:, 0] < (c + 1) * 0.25) &
                (coords[:, 1] >= r * 0.25) & (coords[:, 1] < (r + 1) * 0.25)
            )
            sub_words = words[mask]
            hist_sub = np.zeros(vocab_size, dtype=np.float32)
            if len(sub_words) > 0:
                counts_sub = Counter(sub_words)
                for w_idx, count in counts_sub.items():
                    hist_sub[w_idx] = count
            norm_sub = np.linalg.norm(hist_sub)
            if norm_sub > 0:
                hist_sub /= norm_sub
            spm_feature.append(hist_sub * 0.50)
            
    # Nối tất cả 21 histogram thành vector 21 * vocab_size chiều
    combined = np.concatenate(spm_feature)
    
    # Power normalization (Signed Square Root) để giảm ảnh hưởng của từ thị giác quá phổ biến
    combined = np.sign(combined) * np.sqrt(np.abs(combined))
    
    # L2 Normalization toàn cục
    norm_total = np.linalg.norm(combined)
    if norm_total > 0:
        combined /= norm_total
        
    return combined.astype(np.float32)


def extract_all_spm(image_paths, kmeans_model, vocab_size: int = 500, max_dim: int = 400):
    """
    Sinh vector đặc trưng SPM 3 tầng (21 * vocab_size chiều) cho danh sách ảnh.
    """
    features = []
    for path in image_paths:
        desc, coords = extract_sift_spm(path, max_dim=max_dim)
        feat = compute_spm_features(desc, coords, kmeans_model, vocab_size=vocab_size)
        features.append(feat)
    return np.array(features, dtype=np.float32)


def apply_tfidf_weighting(train_features, test_features):
    """
    Áp dụng trọng số TF-IDF lên ma trận đặc trưng SPM để làm giảm trọng số của các từ nhiễu ở nền.
    """
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    train_tfidf = tfidf.fit_transform(train_features).toarray().astype(np.float32)
    test_tfidf = tfidf.transform(test_features).toarray().astype(np.float32)
    return train_tfidf, test_tfidf, tfidf


def save_spm_features(features, split: str, vocab_size: int = 500):
    """Lưu ma trận đặc trưng SPM ra file .npy."""
    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    path = os.path.join(config.FEATURES_DIR, f"spm_features_{split}_k{vocab_size}.npy")
    np.save(path, features)
    print(f"[SPM] Đã lưu đặc trưng vào {path}")


def load_spm_features(split: str, vocab_size: int = 500):
    """Nạp ma trận đặc trưng SPM từ file .npy."""
    path = os.path.join(config.FEATURES_DIR, f"spm_features_{split}_k{vocab_size}.npy")
    print(f"[SPM] Nạp đặc trưng từ {path}")
    return np.load(path)

