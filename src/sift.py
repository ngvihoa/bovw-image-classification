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


# if __name__ == "__main__":
#     from dataset import load_dataset
#     X_train, X_test, y_train, y_test = load_dataset()
#     descriptors = collect_training_descriptors(X_train[:10])
#     print(f"Total descriptors: {len(descriptors)}")