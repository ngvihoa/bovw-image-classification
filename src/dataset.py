# This file is used for load dataset from the given path
# Then split into train/test dataset

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import glob
import sklearn.model_selection as model_selection
import config


def load_dataset(
        data_dir = config.DATA_DIR,
        classes = config.CLASSES,
        test_size = config.TEST_SIZE,
        random_state = config.RANDOM_STATE
    ):
    """
    Load dataset and split into train/test dataset
    return:
    - train_dataset: list of train dataset - list[str]
    - test_dataset: list of test dataset - list[str]
    - train_labels: list of train labels - list[int]
    - test_labels: list of test labels - list[int]
    """
    images_path = []
    images_label = []

    for idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        explore_path = glob.glob(os.path.join(cls_path, "*.jpg"))
        images_path += explore_path
        images_label += [idx] * len(explore_path)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        images_path, 
        images_label, 
        test_size=test_size, 
        random_state=random_state,
        stratify=images_label
    )
    return X_train, X_test, y_train, y_test


# ==============================================================================
# HÀM NẠP TẬP DỮ LIỆU BENCHMARK 101 LỚP CALTECH-101 (CHO MÔ HÌNH V2)
# ==============================================================================
from pathlib import Path
import re
import numpy as np


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def auto_detect_dataset(search_root="data/raw"):
    """
    Tự động quét tìm thư mục chứa các thư mục con lớp ảnh (Caltech-101 hoặc Swedish Leaf).
    """
    search_path = Path(search_root)
    if not search_path.exists():
        return None

    best_candidate = None
    max_subdirs = 0
    for root, dirs, _files in os.walk(search_path):
        valid_dirs = [d for d in dirs if not d.startswith('.') and d.lower() != 'background_google']
        if len(valid_dirs) > max_subdirs:
            current = Path(root)
            has_images = False
            for d in valid_dirs[:5]:
                sub = current / d
                if any(sub.glob("*.jpg")) or any(sub.glob("*.png")) or any(sub.glob("*.jpeg")) or any(sub.glob("*.tif")):
                    has_images = True
                    break
            if has_images:
                max_subdirs = len(valid_dirs)
                best_candidate = current

    return best_candidate


def load_caltech_benchmark(data_dir=None, train_per_class: int = 24, max_test_per_class: int = 6, random_state: int = config.RANDOM_STATE):
    """
    Nạp dữ liệu Caltech-101 theo chuẩn Benchmark cân bằng (mặc định 24 Train : 6 Test, tỷ lệ 80/20).
    Loại bỏ thư mục nền BACKGROUND_Google để giữ chuẩn xác 101 lớp vật thể.
    """
    if data_dir is None:
        data_dir = auto_detect_dataset()
        if data_dir is None:
            raise FileNotFoundError("Không tìm thấy thư mục tập dữ liệu Caltech-101.")
            
    data_path = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    
    class_dirs = sorted([
        d for d in data_path.iterdir() 
        if d.is_dir() and not d.name.startswith('.') and d.name.lower() != 'background_google'
    ], key=lambda d: natural_sort_key(d.name))
    
    classes = [d.name for d in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    for cls in classes:
        cls_folder = data_path / cls
        files = sorted([p for p in cls_folder.iterdir() if p.suffix.lower() in extensions], key=lambda p: natural_sort_key(p.name))
        
        rng = np.random.RandomState(random_state + class_to_idx[cls])
        indices = rng.permutation(len(files))
        
        n_total = len(files)
        n_tr = min(train_per_class, max(1, n_total - 1))
        n_te = min(max_test_per_class, n_total - n_tr)
        
        tr_files = [str(files[i]) for i in indices[:n_tr]]
        te_files = [str(files[i]) for i in indices[n_tr:n_tr + n_te]]
        
        X_train.extend(tr_files)
        y_train.extend([class_to_idx[cls]] * len(tr_files))
        X_test.extend(te_files)
        y_test.extend([class_to_idx[cls]] * len(te_files))
        
    return X_train, X_test, y_train, y_test, classes