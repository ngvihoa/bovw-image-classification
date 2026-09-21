import json
from pathlib import Path

cells = []

def add_md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

# ==============================================================================
# PHẦN MỞ ĐẦU: TIÊU ĐỀ & HƯỚNG DẪN KAGGLE
# ==============================================================================
add_md("""# Phân Loại Lá Cây (Leaf Classification) với SIFT + BoVW + SVM

**Báo cáo Thực nghiệm Khoa học:** Đánh Giá & Cải Tiến Mô Hình Thị Giác Máy Tính Truyền Thống  
**Tác giả:** ngvihoa  

---
### 📌 Bố Cục Báo Cáo Thực Nghiệm (Gồm 2 Chương Rõ Rệt):
* **CHƯƠNG 1: HIỆN THỰC LẠI MÔ HÌNH CƠ SỞ (MODEL V1 - BASELINE)**
  * Tái hiện toàn bộ pipeline nguyên bản của **Csurka et al. (2004)** trên tập dữ liệu Lá cây mới.
  * Từng module cấu thành được viết độc lập tương ứng với các file trong `/src/`:
    * *1.1 Cấu hình & Dữ liệu (`config.py` & `src/dataset.py`)*
    * *1.2 Trích xuất SIFT tiêu chuẩn (`src/sift.py`)*
    * *1.3 Xây dựng Từ điển thị giác (`src/vocabulary.py`)*
    * *1.4 Biểu diễn BoVW đếm tần suất (`src/bovw.py`)*
    * *1.5 Bộ phân loại Linear SVM (`src/classifier.py`)*
    * *1.6 Đánh giá & Thiết lập mốc Baseline (`src/evaluate.py` & `scripts/train.py`)*
* **CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (MODEL V2 - IMPROVED)**
  * Khắc phục các hạn chế của v1 bằng bộ 4 kỹ thuật nâng cao:
    * *2.1 Tuned SIFT: Tinh chỉnh ngưỡng bắt gân lá mảnh & chuẩn hóa tọa độ $(x, y)$ (`src/sift.py`)*
    * *2.2 Visual Vocabulary tối ưu (`src/vocabulary.py`)*
    * *2.3 Kim tự tháp không gian Spatial Pyramid Matching đa tầng $1\\times1 + 2\\times2$ (`src/spm.py`)*
    * *2.4 Trọng số TF-IDF triệt tiêu nhiễu nền (`src/bovw.py`)*
    * *2.5 Bộ phân loại Additive Chi-Square ($\\chi^2$) Kernel SVM (`src/classifier.py`)*
    * *2.6 Đánh giá mô hình cải tiến v2*
    * *2.7 So sánh đối đầu (Ablation Study) & Phân tích lỗi (Error Analysis)*
    * *2.8 Đóng gói kết quả (Export ZIP Output)*

---
### 🚀 Hướng Dẫn Chạy Trên Kaggle (1-Click):
1. Bấm nút **Add Input** (ở góc trên bên phải).
2. Tìm kiếm và thêm một bộ dữ liệu lá cây, ví dụ: **`Swedish Leaf Dataset`** (hoặc `Flavia Leaf Dataset`).
3. Nhấn **Run All**. Notebook sẽ tự động tìm thư mục ảnh, chạy tuần tự 2 chương, vẽ biểu đồ so sánh và nén file kết quả zip.""")

# ==============================================================================
# CHƯƠNG 1: HIỆN THỰC LẠI MODEL V1 (BASELINE)
# ==============================================================================
add_md("""# ==========================================================
# CHƯƠNG 1: HIỆN THỰC LẠI MÔ HÌNH CƠ SỞ (MODEL V1 - BASELINE)
# ==========================================================
Mục tiêu của Chương 1 là hiện thực lại trung thực pipeline gốc của phiên bản v1 (theo kiến trúc Csurka et al., 2004) và áp dụng trực tiếp lên tập dữ liệu lá cây mới để thiết lập mốc điểm cơ sở (Baseline).""")

# --- 1.1 Config & Data ---
add_md("""### Module 1.1: Cấu hình Hệ thống & Nạp Dữ liệu Lá cây (`config.py` & `src/dataset.py`)
Tự động phát hiện thư mục lá cây trong `/kaggle/input`, đọc nhãn lớp từ các thư mục con và chia tập Train/Test theo phương pháp phân tầng (Stratified 80/20).""")

add_code("""import os
import sys
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# 1. Cấu hình môi trường & Tắt cảnh báo log TIFF của OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# 2. Đường dẫn (tự động nhận diện môi trường Kaggle hoặc Local)
INPUT_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("data/raw")
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_output")
OUTPUT_DIR = WORKING_DIR / "bovw_leaf_output"
RESULTS_DIR = OUTPUT_DIR / "results"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

(RESULTS_DIR / "ch1_baseline").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "ch2_improved").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "comparison").mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Siêu tham số
RANDOM_STATE = 42
TEST_SIZE = 0.2
VOCAB_SIZE_V1 = 200             # Số cụm từ thị giác K
MAX_DESC_PER_IMG = 300          # Lấy mẫu tối đa mỗi ảnh
MAX_TOTAL_DESC = 120_000        # Tổng mẫu tối đa cho K-Means
MAX_IMAGE_DIM = 600             # Thu nhỏ chiều lớn nhất của ảnh scan lá về 600px để tăng tốc SIFT gấp 10-15 lần
TRAIN_PER_CLASS = 25            # Chuẩn Söderkvist (2001): 25 ảnh train / 50 ảnh test mỗi loài để loại bỏ Data Leakage

# 4. Hàm sắp xếp tự nhiên (Leaf 0, Leaf 1, ..., Leaf 14)
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# 5. Hàm tự động dò tìm thư mục Dataset
def auto_detect_dataset(search_root=INPUT_ROOT):
    search_path = Path(search_root)
    if not search_path.exists():
        local_dir = Path("data/raw/101_ObjectCategories")
        return local_dir if local_dir.exists() else None

    # Tìm thư mục chứa ảnh lá cây
    for root, dirs, _files in os.walk(search_path):
        d_lower = {d.lower(): d for d in dirs}
        if "train" in d_lower and "test" in d_lower:
            return Path(root)

    for root, dirs, _files in os.walk(search_path):
        current = Path(root)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        if len(dirs) >= 2:
            img_subdirs = 0
            for d in dirs[:5]:
                sub = current / d
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]:
                    if any(sub.glob(ext)):
                        img_subdirs += 1
                        break
            if img_subdirs >= 2:
                return current
    return None

DATA_DIR = auto_detect_dataset()

if DATA_DIR is None:
    raise FileNotFoundError("Chưa tìm thấy tập dữ liệu trong /kaggle/input. Vui lòng bấm 'Add Input' để thêm dataset lá cây!")

# 6. Nạp dữ liệu chuẩn khoa học: Khử trùng lặp (Anti-Leakage) & Chia 25 Train / 50 Test
def load_dataset(data_dir, train_per_class=TRAIN_PER_CLASS, random_state=RANDOM_STATE):
    data_path = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    
    # Quét toàn bộ file ảnh và gom theo lớp, khử trùng lặp (Deduplication) qua tên file
    class_images = {}
    for p in data_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            cls_name = p.parent.name
            if cls_name.lower() in ["train", "test"]:
                continue
            if cls_name not in class_images:
                class_images[cls_name] = {}
            # Khử trùng lặp: Nếu file cùng tên xuất hiện ở cả Train và Test, chỉ giữ 1 bản duy nhất
            class_images[cls_name][p.name] = str(p)
            
    classes = sorted(class_images.keys(), key=natural_sort_key)
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    for cls in classes:
        unique_imgs = sorted(list(class_images[cls].values()))
        np.random.seed(random_state)
        shuffled = np.random.permutation(unique_imgs)
        
        # Lấy cố định train_per_class (mặc định 25 ảnh) để train, còn lại để test
        n_tr = min(train_per_class, len(shuffled) - 1) if len(shuffled) > train_per_class else int(len(shuffled) * 0.5)
        
        tr = shuffled[:n_tr]
        te = shuffled[n_tr:]
        
        X_train.extend(tr)
        y_train.extend([class_to_idx[cls]] * len(tr))
        X_test.extend(te)
        y_test.extend([class_to_idx[cls]] * len(te))
        
    # Kiểm tra rò rỉ dữ liệu (Data Leakage Verification)
    tr_names = {Path(p).name for p in X_train}
    te_names = {Path(p).name for p in X_test}
    leakage = tr_names.intersection(te_names)
    assert len(leakage) == 0, f"Phát hiện rò rỉ dữ liệu: {len(leakage)} file bị trùng!"
    
    print("=" * 65)
    print(f"ĐÃ NẠP & PHÂN CHIA TẬP DỮ LIỆU CHUẨN KHOA HỌC (Söderkvist, 2001)")
    print(f"• Thư mục nguồn      : {data_path}")
    print(f"• Số loài lá (Classes): {len(classes)} loài")
    print(f"• Tập Huấn Luyện (Train): {len(X_train)} ảnh ({len(X_train)//len(classes)} ảnh/loài)")
    print(f"• Tập Kiểm Tra   (Test) : {len(X_test)} ảnh ({len(X_test)//len(classes)} ảnh/loài)")
    print(f"• Rò rỉ dữ liệu (Leakage): 0 file trùng nhau (Tuyệt đối an toàn)")
    print("=" * 65)
    return X_train, X_test, y_train, y_test, classes

X_train, X_test, y_train, y_test, classes = load_dataset(DATA_DIR, train_per_class=TRAIN_PER_CLASS, random_state=RANDOM_STATE)""")

# --- 1.2 Feature Extraction: SIFT Baseline ---
add_md("""### Module 1.2: Trích xuất Đặc trưng SIFT Tiêu chuẩn (`src/sift.py` - Baseline)
Sử dụng bộ phát hiện và mô tả SIFT tiêu chuẩn của OpenCV (`cv2.SIFT_create`) với các ngưỡng mặc định.""")

add_code("""import cv2
import numpy as np

# Bộ trích xuất SIFT mặc định của v1
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

sift_v1 = cv2.SIFT_create(
    nfeatures=0,
    contrastThreshold=0.04,
    edgeThreshold=10,
    sigma=1.6
)

def extract_sift_v1(image_path, max_dim=MAX_IMAGE_DIM):
    \"\"\"Trích xuất SIFT descriptors (N, 128) từ ảnh xám (đã resize tối ưu tốc độ).\"\"\"
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _kps, descs = sift_v1.detectAndCompute(gray, None)
    if descs is None or len(descs) == 0:
        return None
    return descs

def collect_descriptors_v1(images_path, max_per_image=MAX_DESC_PER_IMG, max_total=MAX_TOTAL_DESC, seed=RANDOM_STATE):
    \"\"\"Thu thập tập descriptors huấn luyện từ điển thị giác.\"\"\"
    np.random.seed(seed)
    desc_list = []
    for p in images_path:
        d = extract_sift_v1(p)
        if d is None:
            continue
        if len(d) > max_per_image:
            idx = np.random.choice(len(d), max_per_image, replace=False)
            d = d[idx]
        desc_list.append(d)
        
    stacked = np.vstack(desc_list)
    if len(stacked) > max_total:
        idx = np.random.choice(len(stacked), max_total, replace=False)
        stacked = stacked[idx]
    print(f"[v1] Đã thu thập {len(stacked):,} descriptors từ tập Train.")
    return stacked""")

# --- 1.3 Vocabulary: MiniBatchKMeans ---
add_md("""### Module 1.3: Xây dựng Từ Điển Thị Giác (`src/vocabulary.py` - Baseline)
Gom cụm toàn bộ local descriptors thành $K=200$ cụm từ thị giác (Visual Codebook) bằng `MiniBatchKMeans`.""")

add_code("""from sklearn.cluster import MiniBatchKMeans

def build_vocabulary_v1(descriptors, vocab_size=VOCAB_SIZE_V1, seed=RANDOM_STATE):
    print(f"[v1] Đang học từ điển thị giác MiniBatchKMeans (K={vocab_size})...")
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=2048,
        random_state=seed,
        n_init="auto"
    )
    kmeans.fit(descriptors)
    print("[v1] Học từ điển hoàn tất!")
    return kmeans""")

# --- 1.4 BoVW Histogram & L2 Norm ---
add_md("""### Module 1.4: Biểu diễn Bag-of-Visual-Words Histogram & Chuẩn Hóa L2 (`src/bovw.py` - Baseline)
Gán cứng từng descriptor vào 1 visual word gần nhất (`kmeans.predict`), đếm tần suất xuất hiện và chuẩn hóa vector bằng chuẩn $L_2$.""")

add_code("""def image_to_bovw_v1(descriptors, kmeans_model, vocab_size=VOCAB_SIZE_V1):
    \"\"\"Tạo vector histogram tần suất BoVW kích thước K.\"\"\"
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab_size, dtype=np.float32)
    word_ids = kmeans_model.predict(descriptors)
    hist = np.bincount(word_ids, minlength=vocab_size).astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 1e-8:
        hist /= norm
    return hist

def build_features_v1(images_path, kmeans_model, vocab_size=VOCAB_SIZE_V1):
    feats = []
    for p in images_path:
        d = extract_sift_v1(p)
        h = image_to_bovw_v1(d, kmeans_model, vocab_size=vocab_size)
        feats.append(h)
    return np.array(feats, dtype=np.float32)""")

# --- 1.5 Classifier: Linear SVM ---
add_md("""### Module 1.5: Bộ Phân Loại Linear SVM (`src/classifier.py` - Baseline)
Huấn luyện bộ phân loại đa lớp Linear SVM với tham số $C=1.0$ trên ma trận đặc trưng $K$ chiều.""")

add_code("""from sklearn.svm import LinearSVC

def train_classifier_v1(X_train, y_train, C=1.0, seed=RANDOM_STATE):
    print(f"[v1] Huấn luyện Linear SVM (C={C})...")
    clf = LinearSVC(C=C, random_state=seed, max_iter=3000)
    clf.fit(X_train, y_train)
    print("[v1] Huấn luyện SVM hoàn tất!")
    return clf""")

# --- 1.6 Evaluation: Metrics & Confusion Matrix (Chương 1) ---
add_md("""### Module 1.6: [THỰC THI CHƯƠNG 1] Huấn Luyện & Ghi Nhận Mốc Baseline v1 (`scripts/train.py`)
Khởi chạy toàn bộ pipeline v1 từ đầu đến cuối, ghi nhận thời gian thực thi, các chỉ số đánh giá và ma trận nhầm lẫn cơ sở.""")

add_code("""from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("=" * 70)
print("BẮT ĐẦU CHẠY THỰC NGHIỆM CHƯƠNG 1: MÔ HÌNH GỐC (BASELINE V1)")
print("Pipeline: SIFT tiêu chuẩn -> BoVW (K=200) -> Chuẩn hóa L2 -> Linear SVM")
print("=" * 70)

t0_v1 = time.time()

# 1. Thu thập descriptors & Học K-Means
train_desc_v1 = collect_descriptors_v1(X_train)
vocab_v1 = build_vocabulary_v1(train_desc_v1, vocab_size=VOCAB_SIZE_V1)

# 2. Xây dựng ma trận đặc trưng BoVW
print("[v1] Trích xuất vector BoVW cho tập Train và Test...")
X_train_v1 = build_features_v1(X_train, vocab_v1, vocab_size=VOCAB_SIZE_V1)
X_test_v1 = build_features_v1(X_test, vocab_v1, vocab_size=VOCAB_SIZE_V1)

# 3. Huấn luyện Linear SVM
clf_v1 = train_classifier_v1(X_train_v1, y_train)

# 4. Dự đoán và tính toán chỉ số
y_pred_v1 = clf_v1.predict(X_test_v1)
time_v1 = time.time() - t0_v1

metrics_v1 = {
    "accuracy": float(accuracy_score(y_test, y_pred_v1)),
    "precision": float(precision_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "time_seconds": round(time_v1, 2)
}

print("")
print("*" * 50)
print(">>> KẾT QUẢ MỐC CƠ SỞ (BASELINE V1) TRÊN TẬP LÁ CÂY:")
print(f"• Accuracy : {metrics_v1['accuracy'] * 100:.2f}%")
print(f"• Macro F1 : {metrics_v1['f1_score'] * 100:.2f}%")
print(f"• Precision: {metrics_v1['precision'] * 100:.2f}%")
print(f"• Recall   : {metrics_v1['recall'] * 100:.2f}%")
print(f"• Thời gian: {metrics_v1['time_seconds']:.1f} giây")
print("*" * 50)

# Vẽ ma trận nhầm lẫn (Tương thích cả seaborn và matplotlib)
fig, ax = plt.subplots(figsize=(10, 8))
try:
    import seaborn as sns
    cm_v1 = confusion_matrix(y_test, y_pred_v1)
    sns.heatmap(cm_v1, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
except ImportError:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_v1, display_labels=classes, cmap="Blues", ax=ax, xticks_rotation=45)

plt.title(f"Chương 1: Ma Trận Nhầm Lẫn Baseline v1 (Acc: {metrics_v1['accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
plt.ylabel("Thực Tế (True Label)")
plt.xlabel("Dự Đoán (Predicted Label)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ch1_baseline" / "confusion_matrix_v1.png")
plt.show()""")

# ==============================================================================
# CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (MODEL V2 - IMPROVED)
# ==============================================================================
add_md("""# ==========================================================
# CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (MODEL V2 - IMPROVED)
# ==========================================================
### 💡 Phân tích nguyên nhân giới hạn của Model v1 trên tập Lá Cây:
1. **Mất thông tin cấu trúc không gian (Orderless):** Chiếc lá có cấu trúc rõ ràng (cuống ở đáy, gân lá ở giữa, đỉnh ở trên). BoVW v1 chỉ đếm tổng số từ mà không biết từ đó nằm ở đâu.
2. **Nhiễu từ nền trắng và mảng xanh thông thường:** Các visual words xuất hiện tràn lan ở viền trắng hay màu xanh chung không giúp phân biệt các loài lá nhưng lại chiếm tần suất áp đảo.
3. **Mô hình tuyến tính không tối ưu cho Histogram:** Khoảng cách Euclid thẳng của `Linear SVM` không phản ánh đúng sự tương đồng giữa hai phân bố xác suất biểu đồ.
4. **Bỏ sót gân lá mảnh:** Ngưỡng mặc định của SIFT bỏ qua các đường vân gân lá có độ tương phản thấp.

$\\Rightarrow$ **Chương 2 triển khai đồng bộ 4 kỹ thuật nâng cấp:** Tuned SIFT $\\rightarrow$ Spatial Pyramid Matching $\\rightarrow$ Trọng số TF-IDF $\\rightarrow$ Additive Chi-Square ($\\chi^2$) Kernel SVM.""")

# --- 2.1 Tuned SIFT ---
add_md("""### Module 2.1: Tuned SIFT Bắt Gân Lá & Chuẩn Hóa Tọa Độ Không Gian (`src/sift.py` - Improved)
* Hạ `contrastThreshold=0.02` để bắt các đường gân lá mảnh và vân răng cưa.
* Tăng `edgeThreshold=15` để không loại bỏ nhầm các gân lá dạng gờ.
* Lưu lại tọa độ chuẩn hóa $(x_{norm}, y_{norm}) \\in [0, 1]^2$ của từng keypoint để chia lưới không gian đa tầng.""")

add_code("""# Bộ trích xuất SIFT tối ưu cho gân lá
sift_v2 = cv2.SIFT_create(
    nfeatures=0,
    contrastThreshold=0.02,  # Tăng độ nhạy với gân lá tương phản thấp
    edgeThreshold=15,        # Bắt viền răng cưa và gân mảnh
    sigma=1.6
)

def extract_sift_v2(image_path, max_dim=MAX_IMAGE_DIM):
    \"\"\"Trích xuất descriptors (N, 128) và tọa độ chuẩn hóa (N, 2) từ ảnh đã resize tối ưu.\"\"\"
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        w, h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kps, descs = sift_v2.detectAndCompute(gray, None)
    
    if descs is None or len(descs) == 0:
        return None, None
        
    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    pts_norm = np.zeros_like(pts)
    pts_norm[:, 0] = pts[:, 0] / max(w, 1)  # x trong khoảng [0, 1]
    pts_norm[:, 1] = pts[:, 1] / max(h, 1)  # y trong khoảng [0, 1]
    
    return descs, pts_norm

def collect_descriptors_v2(images_path, max_per_image=MAX_DESC_PER_IMG, max_total=MAX_TOTAL_DESC, seed=RANDOM_STATE):
    np.random.seed(seed)
    desc_list = []
    for p in images_path:
        d, _ = extract_sift_v2(p)
        if d is None:
            continue
        if len(d) > max_per_image:
            idx = np.random.choice(len(d), max_per_image, replace=False)
            d = d[idx]
        desc_list.append(d)
        
    stacked = np.vstack(desc_list)
    if len(stacked) > max_total:
        idx = np.random.choice(len(stacked), max_total, replace=False)
        stacked = stacked[idx]
    print(f"[v2] Đã thu thập {len(stacked):,} tuned descriptors để học từ điển.")
    return stacked""")

# --- 2.2 Vocabulary v2 ---
add_md("""### Module 2.2: Xây dựng Từ Điển Thị Giác Nâng Cao (`src/vocabulary.py` - Improved)
Học bộ Visual Words mới từ các Tuned Descriptors giàu chi tiết gân lá.""")

add_code("""def build_vocabulary_v2(descriptors, vocab_size=VOCAB_SIZE_V1, seed=RANDOM_STATE):
    print(f"[v2] Đang học từ điển thị giác mới cho Tuned SIFT (K={vocab_size})...")
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=2048,
        random_state=seed,
        n_init="auto"
    )
    kmeans.fit(descriptors)
    print("[v2] Học từ điển v2 hoàn tất!")
    return kmeans""")

# --- 2.3 Spatial Pyramid Matching ---
add_md("""### Module 2.3: Kim Tự Tháp Không Gian (Spatial Pyramid Matching - `src/spm.py`)
Khắc phục nhược điểm mất cấu trúc của BoVW bằng cách tích hợp tọa độ không gian đa tầng (Lazebnik et al., 2006):
* **Level 0 ($1\\times1$):** Toàn bộ chiếc lá $\\rightarrow$ Trọng số $w_0 = 0.5$.
* **Level 1 ($2\\times2$):** Chia thành 4 ô vuông không gian: Trên-Trái, Trên-Phải, Dưới-Trái (cuống lá), Dưới-Phải $\\rightarrow$ Trọng số $w_1 = 0.5$.
* Ghép $1 + 4 = 5$ histogram con thành vector đặc trưng **$5 \\times K$ chiều** ($5 \\times 200 = 1,000$ chiều).""")

add_code("""def compute_spm_histogram(descriptors, pts_norm, kmeans_model, vocab_size=VOCAB_SIZE_V1):
    \"\"\"Tạo vector đặc trưng Spatial Pyramid Matching đa tầng (5 * K chiều).\"\"\"
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(5 * vocab_size, dtype=np.float32)
        
    word_ids = kmeans_model.predict(descriptors)
    
    # 1. Level 0: Toàn bộ ảnh (1x1)
    h0 = np.bincount(word_ids, minlength=vocab_size).astype(np.float32)
    norm0 = np.linalg.norm(h0)
    if norm0 > 1e-8:
        h0 /= norm0
    h0 *= 0.5  # Trọng số tầng 0
    
    # 2. Level 1: Lưới 2x2 (4 ô phần tư)
    cell_x = (pts_norm[:, 0] >= 0.5).astype(int)
    cell_y = (pts_norm[:, 1] >= 0.5).astype(int)
    cell_ids = cell_y * 2 + cell_x  # Chỉ số 0, 1, 2, 3
    
    sub_hists = []
    for c in range(4):
        mask = (cell_ids == c)
        if np.any(mask):
            sub_h = np.bincount(word_ids[mask], minlength=vocab_size).astype(np.float32)
            norm_sub = np.linalg.norm(sub_h)
            if norm_sub > 1e-8:
                sub_h /= norm_sub
        else:
            sub_h = np.zeros(vocab_size, dtype=np.float32)
        sub_hists.append(sub_h * 0.5)  # Trọng số tầng 1
        
    spm_vec = np.concatenate([h0] + sub_hists)
    total_norm = np.linalg.norm(spm_vec)
    if total_norm > 1e-8:
        spm_vec /= total_norm
    return spm_vec

def build_spm_features(images_path, kmeans_model, vocab_size=VOCAB_SIZE_V1):
    feats = []
    for p in images_path:
        d, pts = extract_sift_v2(p)
        spm_h = compute_spm_histogram(d, pts, kmeans_model, vocab_size=vocab_size)
        feats.append(spm_h)
    return np.array(feats, dtype=np.float32)""")

# --- 2.4 TF-IDF Weighting ---
add_md("""### Module 2.4: Trọng Số TF-IDF Triệt Tiêu Nhiễu Nền (`src/bovw.py` - Improved)
Tính vector nghịch đảo tần suất văn bản (IDF) trên tập Train và nhân trọng số để dập tắt các từ thị giác xuất hiện quá phổ biến ở mọi ảnh:
$$\\text{IDF}_j = \\log\\left(\\frac{N + 1}{1 + \\text{DF}_j}\\right) + 1$$""")

add_code("""def compute_idf_v2(train_features):
    \"\"\"Tính vector IDF từ tập đặc trưng huấn luyện SPM.\"\"\"
    N = len(train_features)
    doc_freq = np.sum(train_features > 0, axis=0)
    idf = np.log((N + 1.0) / (1.0 + doc_freq)) + 1.0
    return idf

def apply_tfidf_v2(features, idf_vector):
    \"\"\"Nhân trọng số TF-IDF và chuẩn hóa lại L2.\"\"\"
    weighted = features * idf_vector
    norms = np.linalg.norm(weighted, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return weighted / norms""")

# --- 2.5 Classifier: Additive Chi-Square Kernel SVM ---
add_md("""### Module 2.5: Phân Loại Bằng Additive Chi-Square ($\\chi^2$) Kernel SVM (`src/classifier.py` - Improved)
Sử dụng `AdditiveChi2Sampler` để ánh xạ vector sang không gian phi tuyến $\\chi^2$ (chuẩn mực lý thuyết cho histogram xác suất), sau đó tối ưu bằng `LinearSVC`.""")

add_code("""from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.pipeline import Pipeline

def train_classifier_v2(X_train, y_train, C=1.0, seed=RANDOM_STATE):
    \"\"\"Huấn luyện Additive Chi-Square Kernel SVM.\"\"\"
    print(f"[v2] Huấn luyện Additive Chi-Square Kernel SVM (C={C})...")
    pipeline = Pipeline([
        ('chi2_sampler', AdditiveChi2Sampler(sample_steps=2)),
        ('linear_svc', LinearSVC(C=C, random_state=seed, max_iter=3000))
    ])
    pipeline.fit(X_train, y_train)
    print("[v2] Huấn luyện Chi2-SVM hoàn tất!")
    return pipeline""")

# --- 2.6 Evaluation: Metrics & Confusion Matrix (Chương 2) ---
add_md("""### Module 2.6: [THỰC THI CHƯƠNG 2] Huấn Luyện & Ghi Nhận Kết Quả Cải Tiến v2
Khởi chạy quy trình nâng cấp toàn diện và ghi nhận kết quả của Model v2.""")

add_code("""print("=" * 70)
print("BẮT ĐẦU CHẠY THỰC NGHIỆM CHƯƠNG 2: MÔ HÌNH CẢI TIẾN (IMPROVED V2)")
print("Pipeline: Tuned SIFT -> SPM (1x1, 2x2) -> TF-IDF Weighting -> Chi2-SVM")
print("=" * 70)

t0_v2 = time.time()

# 1. Thu thập Tuned descriptors & Học K-Means v2
train_desc_v2 = collect_descriptors_v2(X_train)
vocab_v2 = build_vocabulary_v2(train_desc_v2, vocab_size=VOCAB_SIZE_V1)

# 2. Xây dựng ma trận đặc trưng Spatial Pyramid Matching (5 * K chiều)
print("[v2] Trích xuất vector Spatial Pyramid Matching đa tầng...")
X_train_spm = build_spm_features(X_train, vocab_v2, vocab_size=VOCAB_SIZE_V1)
X_test_spm = build_spm_features(X_test, vocab_v2, vocab_size=VOCAB_SIZE_V1)

# 3. Tính toán và áp dụng trọng số TF-IDF
print("[v2] Tính toán và nhân trọng số TF-IDF...")
idf_vec_v2 = compute_idf_v2(X_train_spm)
X_train_v2 = apply_tfidf_v2(X_train_spm, idf_vec_v2)
X_test_v2 = apply_tfidf_v2(X_test_spm, idf_vec_v2)

# 4. Huấn luyện Additive Chi-Square Kernel SVM
clf_v2 = train_classifier_v2(X_train_v2, y_train, C=1.0)

# 5. Đánh giá kết quả
y_pred_v2 = clf_v2.predict(X_test_v2)
time_v2 = time.time() - t0_v2

metrics_v2 = {
    "accuracy": float(accuracy_score(y_test, y_pred_v2)),
    "precision": float(precision_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "time_seconds": round(time_v2, 2)
}

print("")
print("*" * 50)
print(">>> KẾT QUẢ MÔ HÌNH CẢI TIẾN (IMPROVED V2) TRÊN TẬP LÁ CÂY:")
print(f"• Accuracy : {metrics_v2['accuracy'] * 100:.2f}%")
print(f"• Macro F1 : {metrics_v2['f1_score'] * 100:.2f}%")
print(f"• Precision: {metrics_v2['precision'] * 100:.2f}%")
print(f"• Recall   : {metrics_v2['recall'] * 100:.2f}%")
print(f"• Thời gian: {metrics_v2['time_seconds']:.1f} giây")
print("*" * 50)

# Vẽ ma trận nhầm lẫn v2 (Tương thích cả seaborn và matplotlib)
fig, ax = plt.subplots(figsize=(10, 8))
try:
    import seaborn as sns
    cm_v2 = confusion_matrix(y_test, y_pred_v2)
    sns.heatmap(cm_v2, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=ax)
except ImportError:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_v2, display_labels=classes, cmap="Greens", ax=ax, xticks_rotation=45)

plt.title(f"Chương 2: Ma Trận Nhầm Lẫn Improved v2 (Acc: {metrics_v2['accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
plt.ylabel("Thực Tế (True Label)")
plt.xlabel("Dự Đoán (Predicted Label)")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ch2_improved" / "confusion_matrix_v2.png")
plt.show()""")

# --- 2.7 Head-to-Head Comparison & Error Analysis ---
add_md("""### Module 2.7: Báo Cáo Đối Đầu (Ablation Study) & Phân Tích Lỗi (Error Analysis)
Tổng hợp bảng so sánh đối đầu giữa **Chương 1 (Baseline v1)** và **Chương 2 (Improved v2)** trên cùng tập dữ liệu kiểm tra, vẽ biểu đồ so sánh cột và trực quan hóa các ca đoán sai đã được sửa đúng.""")

add_code("""# 1. Bảng số liệu tổng hợp đối đầu
summary_data = {
    "Chỉ số Đánh Giá": ["Accuracy (Độ chính xác)", "Macro F1-Score", "Macro Precision", "Macro Recall", "Thời gian huấn luyện"],
    "Chương 1: Baseline v1": [
        f"{metrics_v1['accuracy'] * 100:.2f}%",
        f"{metrics_v1['f1_score'] * 100:.2f}%",
        f"{metrics_v1['precision'] * 100:.2f}%",
        f"{metrics_v1['recall'] * 100:.2f}%",
        f"{metrics_v1['time_seconds']:.1f}s"
    ],
    "Chương 2: Improved v2": [
        f"{metrics_v2['accuracy'] * 100:.2f}%",
        f"{metrics_v2['f1_score'] * 100:.2f}%",
        f"{metrics_v2['precision'] * 100:.2f}%",
        f"{metrics_v2['recall'] * 100:.2f}%",
        f"{metrics_v2['time_seconds']:.1f}s"
    ],
    "Mức Độ Cải Thiện (+/-)": [
        f"{(metrics_v2['accuracy'] - metrics_v1['accuracy']) * 100:+.2f}%",
        f"{(metrics_v2['f1_score'] - metrics_v1['f1_score']) * 100:+.2f}%",
        f"{(metrics_v2['precision'] - metrics_v1['precision']) * 100:+.2f}%",
        f"{(metrics_v2['recall'] - metrics_v1['recall']) * 100:+.2f}%",
        f"{metrics_v2['time_seconds'] - metrics_v1['time_seconds']:+.1f}s"
    ]
}

print("=" * 65)
print("BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU CHƯƠNG 1 (V1) VS CHƯƠNG 2 (V2)")
print("=" * 65)

try:
    import pandas as pd
    from IPython.display import display
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)
except ImportError:
    for k, v1, v2, diff in zip(summary_data["Chỉ số Đánh Giá"], summary_data["Chương 1: Baseline v1"], summary_data["Chương 2: Improved v2"], summary_data["Mức Độ Cải Thiện (+/-)"]):
        print(f"• {k:<25}: v1 = {v1:<8} | v2 = {v2:<8} | Cải thiện: {diff}")

# 2. Biểu đồ cột so sánh trực quan
metric_keys = ["Accuracy", "Macro F1", "Precision", "Recall"]
v1_vals = [metrics_v1['accuracy'] * 100, metrics_v1['f1_score'] * 100, metrics_v1['precision'] * 100, metrics_v1['recall'] * 100]
v2_vals = [metrics_v2['accuracy'] * 100, metrics_v2['f1_score'] * 100, metrics_v2['precision'] * 100, metrics_v2['recall'] * 100]

x = np.arange(len(metric_keys))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, v1_vals, width, label='Chương 1: Baseline v1', color='#4c72b0')
plt.bar(x + width/2, v2_vals, width, label='Chương 2: Improved v2', color='#2ca02c')

for i in range(len(metric_keys)):
    plt.text(x[i] - width/2, v1_vals[i] + 1.2, f"{v1_vals[i]:.1f}%", ha='center', fontsize=10)
    plt.text(x[i] + width/2, v2_vals[i] + 1.2, f"{v2_vals[i]:.1f}%", ha='center', fontsize=10, fontweight='bold')

plt.ylabel('Tỷ Lệ Phần Trăm (%)', fontsize=11)
plt.title('So Sánh Hiệu Năng Đối Đầu: Baseline v1 vs Improved v2', fontsize=13, fontweight='bold', pad=12)
plt.xticks(x, metric_keys, fontsize=11)
plt.ylim(0, 112)
plt.legend(loc='lower right', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison" / "comparison_chart.png")
plt.show()

# 3. Phân tích lỗi (Error Analysis): Ảnh v1 đoán sai được v2 sửa đúng
fixed_samples = []
for idx, (true_y, p1, p2) in enumerate(zip(y_test, y_pred_v1, y_pred_v2)):
    if p1 != true_y and p2 == true_y:
        fixed_samples.append((X_test[idx], true_y, p1, p2))

print("")
print(f"[Error Analysis] Số lượng mẫu ảnh v1 đoán sai đã được v2 sửa đúng: {len(fixed_samples)} ảnh.")

if len(fixed_samples) > 0:
    n_show = min(3, len(fixed_samples))
    fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
    if n_show == 1:
        axes = [axes]
    for i, (path, true_y, p1, p2) in enumerate(fixed_samples[:n_show]):
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(f"Thực tế: {classes[true_y]} | v1 sai: {classes[p1]} -> v2 đúng: {classes[p2]}", 
                          fontsize=10, color='darkgreen', pad=8)
        axes[i].axis('off')
    plt.suptitle("Minh Chứng Thực Nghiệm: Các Ảnh Được Mô Hình Cải Tiến Sửa Sai Thành Công", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "comparison" / "error_analysis_fixed_samples.png")
    plt.show()""")

# --- 2.8 Export Zip Output ---
add_md("""### Module 2.8: Đóng Gói Toàn Bộ Kết Quả (Export ZIP Output)
Tự động xuất file `leaf_classification_results_v2.zip` vào `/kaggle/working/` chứa toàn bộ bảng tóm tắt JSON, ma trận nhầm lẫn 2 chương và biểu đồ so sánh để tải về máy chỉ với 1 click.""")

add_code("""import shutil
import json

# Lưu file tổng kết đối đầu JSON
final_report = {
    "dataset": DATA_DIR.name,
    "classes": classes,
    "ch1_baseline": metrics_v1,
    "ch2_improved": metrics_v2,
    "improvement": {
        "accuracy_gain": round((metrics_v2['accuracy'] - metrics_v1['accuracy']) * 100, 2),
        "f1_gain": round((metrics_v2['f1_score'] - metrics_v1['f1_score']) * 100, 2)
    }
}

with open(RESULTS_DIR / "comparison" / "final_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

# Nén thư mục kết quả thành file zip
archive_name = WORKING_DIR / "leaf_classification_results_v2"
shutil.make_archive(str(archive_name), 'zip', OUTPUT_DIR)

print("=" * 70)
print("XUẤT KẾT QUẢ THÀNH CÔNG!")
print(f"File zip hoàn chỉnh: {archive_name}.zip")
print("Bạn có thể tải trực tiếp file zip này tại tab 'Output' của Kaggle để phục vụ báo cáo/nộp bài!")
print("=" * 70)""")

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "dockerImageVersionId": None,
            "isGpuEnabled": False,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

output_path = Path("kaggle_bovw_v2.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Hoàn thành tạo kaggle_bovw_v2.ipynb!")
