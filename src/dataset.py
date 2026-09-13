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


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = load_dataset()
#     print(f"Train dataset: {len(X_train)}")
#     print(f"Test dataset: {len(X_test)}")