# Extract sift descriptors from images
# Collect descriptors from train_set (used for k-mean later)

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import config
import numpy as np

sift = cv2.SIFT_create()

def extract_sift(image_path):
    """
    Extract sift descriptors from image
    return:
    - descriptors: np.ndarray of shape (n_descriptors, 128)
    """
    image = cv2.imread(image_path)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted = sift.detectAndCompute(gray_scale, None)
    if extracted[1] is None:
        return None
    return extracted[1]

def collect_training_descriptors(
        images_path,
        max_descriptor_per_image = config.MAX_DESCRIPTOR_PER_IMAGE,
        max_total_descriptors = config.MAX_TOTAL_DESCRIPTORS
        ):
    all_descriptors = []
    for _idx, image_path in enumerate(images_path):
        descriptors = extract_sift(image_path)
        if descriptors is None:
            continue
        if len(descriptors) > max_descriptor_per_image:
            indices = np.random.choice(len(descriptors), max_descriptor_per_image, replace=False)
            descriptors = descriptors[indices]

        all_descriptors.append(descriptors)
    lst = np.vstack(all_descriptors)

    if len(lst) > max_total_descriptors:
        indices = np.random.choice(len(lst), max_total_descriptors, replace=False)
        lst = lst[indices]

    return lst


def extract_sift_spm(image_path, max_dim=400, contrast_threshold=0.03, edge_threshold=10):
    """
    Trích xuất SIFT descriptors kèm tọa độ không gian chuẩn hóa [0, 1] x [0, 1] cho mô hình v2 (SPM).
    Returns:
    - descriptors: np.ndarray shape (N, 128)
    - coordinates: np.ndarray shape (N, 2) với x_norm, y_norm in [0, 1]
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
    sift_spm = cv2.SIFT_create(contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
    kps, desc = sift_spm.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return None, None
        
    coords = np.array([[kp.pt[0] / w, kp.pt[1] / h] for kp in kps], dtype=np.float32)
    return desc, coords