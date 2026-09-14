# Image Classification using SIFT, Bag of Visual Words and SVM

A traditional computer vision image-classification pipeline inspired by Csurka et al. (2004), *"Visual Categorization with Bags of Keypoints"*.

This implementation intentionally avoids deep-learning feature extractors such as CNNs and Vision Transformers.

## Pipeline

```
Image
  ↓
SIFT feature extraction
  ↓
Local descriptors (128-D)
  ↓
MiniBatch K-Means → Visual Vocabulary
  ↓
Bag of Visual Words histogram
  ↓
L2 normalization
  ↓
SVM classifier
  ↓
Predicted category
```

## Method

1. **SIFT** — extract local handcrafted descriptors from each image (`cv2.SIFT_create()`)
2. **MiniBatch K-Means** — cluster training descriptors to build a visual vocabulary of size K
3. **BoVW histogram** — represent each image as a K-dimensional frequency vector, L2-normalized
4. **Linear SVM** — train a multi-class classifier on the histogram features

## Dataset

**Caltech-101** — 5 selected classes:

| Class | Images |
|---|---:|
| airplanes | 800 |
| Motorbikes | 798 |
| Faces | 435 |
| watch | 239 |
| car_side | 123 |
| **Total** | **2395** |

**Download:** [Caltech-101 (Google Drive / Caltech)](https://data.caltech.edu/records/mzrjq-6wc02)

After downloading, extract to:
```
data/raw/101_ObjectCategories/
```

## Project Structure

```
bovw-image-classification/
├── config.py                  # Global constants and paths
├── requirements.txt
├── data/
│   └── raw/
│       └── 101_ObjectCategories/
├── src/
│   ├── dataset.py             # Dataset loading and train/test split
│   ├── sift.py                # SIFT extraction and descriptor sampling
│   ├── vocabulary.py          # MiniBatchKMeans visual vocabulary
│   ├── bovw.py                # BoVW histogram construction
│   ├── classifier.py          # SVM training and inference
│   └── evaluate.py            # Metrics and confusion matrix
├── scripts/
│   └── train.py               # End-to-end training script
├── artifacts/
│   ├── vocabularies/          # Saved K-Means models (.joblib)
│   ├── features/              # Cached BoVW feature matrices (.npy)
│   └── classifiers/           # Saved SVM models (.joblib)
└── results/
    ├── metrics/               # JSON metric files
    ├── plots/                 # Accuracy / F1 plots
    └── confusion/             # Confusion matrix images
```

## Setup

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone <repo-url>
cd bovw-image-classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify SIFT is available

```bash
python scripts/test.py
```

## Usage

### Train with default vocabulary size (K=100)

```bash
python scripts/train.py
```

### Train with a specific vocabulary size

```bash
python scripts/train.py --vocab_size 200
```

### Run experiments across multiple vocabulary sizes

```bash
for K in 50 100 200 500; do
    python scripts/train.py --vocab_size $K
done
```

Results are saved automatically to `results/metrics/` and `results/confusion/`.

## Configuration

Edit [`config.py`](config.py) to change:

| Parameter | Default | Description |
|---|---|---|
| `CLASSES` | 5 classes | Target class names from Caltech-101 |
| `TEST_SIZE` | `0.2` | Fraction of data held out for testing |
| `DEFAULT_VOCAB_SIZE` | `100` | Default K for K-Means vocabulary |
| `VOCAB_SIZES` | `[50, 100, 200, 500]` | Vocabulary sizes to experiment with |
| `MAX_DESCRIPTOR_PER_IMAGE` | `200` | Max SIFT descriptors sampled per image |
| `MAX_TOTAL_DESCRIPTORS` | `100,000` | Max total descriptors fed to K-Means |
| `RANDOM_STATE` | `42` | Global random seed |

## Results

Vocabulary size experiment (linear SVM, `C=1.0`):

| Vocabulary size | Accuracy | Macro F1 |
|---:|---:|---:|
| 50 | — | — |
| 100 | — | — |
| 200 | — | — |
| 500 | — | — |

*(Fill in after running experiments)*

## Citation

```bibtex
@inproceedings{csurka2004visual,
  title     = {Visual Categorization with Bags of Keypoints},
  author    = {Csurka, Gabriella and Dance, Christopher R. and Fan, Lixin and
               Willamowski, Jutta and Bray, C{\'e}dric},
  booktitle = {ECCV Workshop on Statistical Learning in Computer Vision},
  year      = {2004}
}
```

> This work is a simplified reimplementation inspired by the Bag-of-Visual-Words framework proposed by Csurka et al. (2004), using SIFT descriptors, K-Means vocabulary construction, and SVM classification.

