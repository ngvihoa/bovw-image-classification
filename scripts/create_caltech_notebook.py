import json
from pathlib import Path

VOCAB_SIZE = 500
TRAIN_PER_CLASS = 24
MAX_TEST_PER_CLASS = 6

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
add_md("""# Phân Loại Toàn Bộ 101 Lớp Caltech-101 bằng SIFT + BoVW + Spatial Pyramid Matching + SVM

**Báo Cáo Thực Nghiệm & Nghiên Cứu Chuyên Sâu:** Đánh Giá Hiệu Năng Với Từ Điển Quy Mô $K=500$ Kết Hợp Kim Tự Tháp Không Gian Toàn Diện (SPM $1\times1 + 2\times2 + 4\times4$) và Nhân Additive $\chi^2$ trên Toàn Bộ 101 Lớp Caltech-101.  
**Tác giả:** ngvihoa  

---
### 📌 Bố Cục Báo Cáo Thực Nghiệm (Gồm 2 Chương Rõ Rệt):
* **CHƯƠNG 1: HIỆN THỰC LẠI MÔ HÌNH CƠ SỞ (MODEL V1 - BASELINE)**
  * Tái hiện pipeline nguyên bản (Csurka et al., 2004) với từ điển chuẩn **$K=500$**:
    * SIFT tiêu chuẩn -> BoVW (K=500) -> Chuẩn hóa $L_2$ -> Linear SVM.
    * Đánh giá hiệu năng mốc cơ sở trên toàn bộ 101 lớp của Caltech-101 (chia tỷ lệ vàng 24 Train : 6 Test).
* **CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (MODEL V2 - IMPROVED)**
  * Khắc phục triệt để hạn chế mất thông tin không gian của BoVW bằng kiến trúc kinh điển của **Lazebnik et al. (CVPR 2006)**:
    * Giữ nguyên kích thước từ điển quy mô **$K=500$**.
    * **Spatial Pyramid Matching (SPM)** đa tầng $1\times1 + 2\times2 + 4\times4$ (21 ô không gian) mở rộng biểu diễn lên $21 \times 500 = \mathbf{10.500}$ chiều.
    * Trọng số **TF-IDF** triệt tiêu nhiễu nền vật thể.
    * Bộ phân loại **Additive Chi-Square ($\chi^2$) Kernel SVM** đo lường độ tương đồng phân phối histogram.
* **CHƯƠNG 3: BÁO CÁO ĐỐI ĐẦU (ABLATION STUDY), DEMO & ĐÓNG GÓI OUTPUT**
  * So sánh đối đầu trực tiếp: Baseline v1 ($K=500$) vs Improved v2 ($K=500$ + SPM + $\chi^2$).
  * Phân tích các ca đoán sai của v1 đã được v2 khắc phục (Error Analysis).
  * Demo dự đoán ảnh bất kỳ (External Image Inference).
  * Tự động đóng gói toàn bộ biểu đồ, kết quả thành file `.zip` tải về máy.

---
### 🚀 Hướng Dẫn Chạy Trên Kaggle (1-Click - Không Cần Clone GitHub):
1. Bấm nút **Add Input** (ở góc trên bên phải giao diện Kaggle).
2. Tìm kiếm và thêm bộ dữ liệu **`Caltech 101`** (hoặc `101_ObjectCategories`).
3. Nhấn **Run All**. Notebook sẽ tự động tìm kiếm thư mục dữ liệu, nạp toàn bộ 101 lớp, huấn luyện và xuất báo cáo hoàn chỉnh!""")

# ==============================================================================
# MODULE 1.1: CẤU HÌNH & TỰ ĐỘNG QUÉT DỮ LIỆU
# ==============================================================================
add_md("""### Module 1.1: Cấu hình Hệ thống & Tự Động Quét Toàn Bộ 101 Lớp Caltech-101
Tự động quét và định vị thư mục chứa 101 lớp vật thể trong `/kaggle/input`, loại bỏ thư mục nền `BACKGROUND_Google` để tuân thủ chuẩn Benchmark học thuật (Fei-Fei Li et al.), co ảnh về độ phân giải chuẩn và phân chia tập Train/Test cân bằng theo từng lớp.""")

add_code("""import os
import sys
import time
import re
from pathlib import Path
from collections import Counter
import numpy as np

# 1. Tắt cảnh báo log OpenCV
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

# 2. Định vị đường dẫn môi trường (Tự động thích ứng Kaggle hoặc Local)
INPUT_ROOT = Path("/kaggle/input") if Path("/kaggle/input").exists() else Path("data/raw")
WORKING_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("kaggle_output")
OUTPUT_DIR = WORKING_DIR / "caltech101_output"
RESULTS_DIR = OUTPUT_DIR / "results"
ARTIFACTS_DIR = OUTPUT_DIR / "artifacts"

(RESULTS_DIR / "ch1_baseline").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "ch2_improved").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "comparison").mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Siêu tham số thực nghiệm (K=500, Tỷ lệ vàng 24 Train : 6 Test, SPM 4x4)
RANDOM_STATE = 42
VOCAB_SIZE = 500                 # Từ điển K=500 từ thị giác
TRAIN_PER_CLASS = 24             # 24 ảnh train mỗi lớp (chuẩn tỷ lệ 80%)
MAX_TEST_PER_CLASS = 6           # 6 ảnh test mỗi lớp (chuẩn tỷ lệ 20%)
MAX_IMAGE_DIM = 400              # Co ảnh về kích thước tối đa 400px (tăng tốc trích xuất 5-10 lần)
MAX_DESC_PER_IMG = 300           # Giới hạn số descriptor lấy mẫu mỗi ảnh
MAX_TOTAL_DESC = 120_000         # Giới hạn tổng số descriptor huấn luyện K-Means

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

# 4. Thuật toán tự động tìm kiếm thư mục Caltech-101
def auto_detect_caltech(search_root=INPUT_ROOT):
    search_path = Path(search_root)
    if not search_path.exists():
        return None

    # Tìm thư mục chứa nhiều thư mục con là các lớp vật thể (>= 20 thư mục con)
    best_candidate = None
    max_subdirs = 0
    for root, dirs, _files in os.walk(search_path):
        valid_dirs = [d for d in dirs if not d.startswith('.') and d.lower() != 'background_google']
        if len(valid_dirs) > max_subdirs:
            # Kiểm tra xem các thư mục con này có chứa ảnh hay không
            current = Path(root)
            has_images = False
            for d in valid_dirs[:5]:
                sub = current / d
                if any(sub.glob("*.jpg")) or any(sub.glob("*.png")) or any(sub.glob("*.jpeg")):
                    has_images = True
                    break
            if has_images:
                max_subdirs = len(valid_dirs)
                best_candidate = current

    return best_candidate

DATA_DIR = auto_detect_caltech()
if DATA_DIR is None:
    raise FileNotFoundError("Chưa tìm thấy tập dữ liệu Caltech-101 trong /kaggle/input. Vui lòng bấm 'Add Input' để thêm Caltech-101!")

# 5. Hàm nạp dữ liệu chuẩn Benchmark học thuật (Balanced Train/Test Split)
def load_caltech_dataset(data_dir, train_per_class=TRAIN_PER_CLASS, max_test_per_class=MAX_TEST_PER_CLASS, seed=RANDOM_STATE):
    data_path = Path(data_dir)
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".bmp"}
    
    # Quét tất cả thư mục lớp (loại bỏ BACKGROUND_Google để giữ 101 lớp vật thể chuẩn)
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
        
        # Xáo trộn có kiểm soát bằng seed
        rng = np.random.RandomState(seed + class_to_idx[cls])
        indices = rng.permutation(len(files))
        
        # Đảm bảo mỗi lớp có đủ ảnh train và ít nhất 5 ảnh test
        n_total = len(files)
        n_tr = min(train_per_class, max(1, n_total - 5))
        n_te = min(max_test_per_class, n_total - n_tr)
        
        tr_files = [str(files[i]) for i in indices[:n_tr]]
        te_files = [str(files[i]) for i in indices[n_tr:n_tr + n_te]]
        
        X_train.extend(tr_files)
        y_train.extend([class_to_idx[cls]] * len(tr_files))
        X_test.extend(te_files)
        y_test.extend([class_to_idx[cls]] * len(te_files))
        
    # Kiểm tra rò rỉ dữ liệu (Leakage check)
    tr_set = set(X_train)
    te_set = set(X_test)
    assert len(tr_set.intersection(te_set)) == 0, "Phát hiện rò rỉ dữ liệu giữa tập Train và Test!"
    
    print("=" * 70)
    print(f"ĐÃ NẠP THÀNH CÔNG TẬP DỮ LIỆU CALTECH-101 CHUẨN BENCHMARK")
    print(f"• Thư mục dữ liệu       : {data_path}")
    print(f"• Tổng số lớp (Classes) : {len(classes)} lớp (Đã loại bỏ thư mục nền BACKGROUND_Google)")
    print(f"• Tập Huấn Luyện (Train): {len(X_train)} ảnh (Trung bình {len(X_train)/len(classes):.1f} ảnh/lớp)")
    print(f"• Tập Kiểm Tra (Test)   : {len(X_test)} ảnh (Trung bình {len(X_test)/len(classes):.1f} ảnh/lớp)")
    print(f"• Rò rỉ dữ liệu         : 0 file trùng lặp (100% An toàn)")
    print("=" * 70)
    return X_train, X_test, y_train, y_test, classes

X_train, X_test, y_train, y_test, classes = load_caltech_dataset(DATA_DIR)""")

# ==============================================================================
# CHƯƠNG 1: HIỆN THỰC MÔ HÌNH CƠ SỞ (MODEL V1 - BASELINE)
# ==============================================================================
add_md("""# ==========================================================
# CHƯƠNG 1: HIỆN THỰC MÔ HÌNH CƠ SỞ (MODEL V1 - BASELINE)
# ==========================================================
Mục tiêu của Chương 1 là thiết lập mốc đo lường cơ sở (Baseline) của phương pháp Bag of Visual Words truyền thống (Csurka et al., 2004) với từ điển thị giác trên toàn bộ 101 lớp của Caltech-101. Pipeline gồm:
1. Trích xuất đặc trưng SIFT tiêu chuẩn từ ảnh xám.
2. Gom cụm K-Means tạo từ điển thị giác gồm $K$ visual words (mặc định $K=500$).
3. Biểu diễn histogram túi từ (BoVW) chuẩn hóa $L_2$.
4. Huấn luyện bộ phân loại Linear SVM đa lớp (One-vs-Rest).""")

# --- 1.2 Feature Extraction: SIFT Baseline ---
add_md("""### Module 1.2: Trích xuất Đặc trưng SIFT Tiêu chuẩn (Baseline)
Đọc ảnh xám, co kích thước về tối đa `MAX_IMAGE_DIM = 400` bằng phép nội suy vùng ảnh `cv2.INTER_AREA` và trích xuất vector 128 chiều đặc trưng SIFT.""")

add_code("""import cv2

def extract_sift_v1(image_path, max_dim=MAX_IMAGE_DIM):
    \"\"\"Đọc ảnh xám, co tỉ lệ chuẩn và trích xuất descriptor SIFT 128 chiều.\"\"\"
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Co ảnh về kích thước chuẩn nếu vượt quá max_dim
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors

def collect_descriptors_v1(image_paths, max_per_img=MAX_DESC_PER_IMG, max_total=MAX_TOTAL_DESC, seed=RANDOM_STATE):
    \"\"\"Thu thập tập mẫu descriptor từ tập huấn luyện để xây dựng từ điển thị giác.\"\"\"
    rng = np.random.RandomState(seed)
    collected = []
    total_count = 0
    
    print(f"[v1] Đang trích xuất SIFT mẫu từ {len(image_paths)} ảnh train...")
    for idx, path in enumerate(image_paths):
        desc = extract_sift_v1(path)
        if desc is not None and len(desc) > 0:
            if len(desc) > max_per_img:
                chosen = rng.choice(len(desc), size=max_per_img, replace=False)
                desc = desc[chosen]
            collected.append(desc)
            total_count += len(desc)
            if total_count >= max_total:
                break
        if (idx + 1) % 400 == 0 or (idx + 1) == len(image_paths):
            print(f"  -> Đã duyệt {idx + 1}/{len(image_paths)} ảnh (Thu được: {total_count} descriptors)")
            
    all_desc = np.vstack(collected).astype(np.float32)
    print(f"[v1] Hoàn tất thu thập: {all_desc.shape[0]} descriptors (kích thước {all_desc.shape[1]} chiều).")
    return all_desc""")

# --- 1.3 Vocabulary: K-Means ---
add_md(f"""### Module 1.3: Xây dựng Từ Điển Thị Giác $K={VOCAB_SIZE}$ (Visual Vocabulary)
Sử dụng `MiniBatchKMeans` với số cụm $K={VOCAB_SIZE}$ để tạo {VOCAB_SIZE} từ thị giác (Visual Words) đại diện cho các mẫu cục bộ phổ biến trong ảnh.""")

add_code("""from sklearn.cluster import MiniBatchKMeans
import joblib

def build_vocabulary_v1(descriptors, k=VOCAB_SIZE, seed=RANDOM_STATE):
    \"\"\"Huấn luyện MiniBatchKMeans để xây dựng từ điển thị giác.\"\"\"
    print(f"[v1] Đang phân cụm MiniBatchKMeans với K={k} từ thị giác...")
    start_t = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=seed,
        batch_size=1024,
        max_iter=100,
        n_init=3,
        reassignment_ratio=0.01
    )
    kmeans.fit(descriptors)
    print(f"[v1] Xây dựng từ điển hoàn tất sau {time.time() - start_t:.2f} giây.")
    return kmeans""")

# --- 1.4 BoVW Representation ---
add_md(f"""### Module 1.4: Biểu Diễn Túi Từ Thị Giác Chuẩn Hóa $L_2$ (BoVW)
Gán từng descriptor vào cụm từ gần nhất, lập histogram tần suất {VOCAB_SIZE} chiều và chuẩn hóa $L_2$ vector đặc trưng của mỗi ảnh.""")

add_code("""def compute_bovw_v1(descriptors, kmeans_model, k=VOCAB_SIZE):
    \"\"\"Chuyển tập descriptor của 1 ảnh thành vector BoVW chuẩn hóa L2.\"\"\"
    hist = np.zeros(k, dtype=np.float32)
    if descriptors is not None and len(descriptors) > 0:
        words = kmeans_model.predict(descriptors.astype(np.float32))
        counts = Counter(words)
        for w_idx, count in counts.items():
            hist[w_idx] = count
            
    # Chuẩn hóa L2
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist

def extract_features_v1(image_paths, kmeans_model, k=VOCAB_SIZE):
    \"\"\"Biểu diễn toàn bộ tập ảnh thành ma trận đặc trưng (N, K).\"\"\"
    features = []
    for path in image_paths:
        desc = extract_sift_v1(path)
        feat = compute_bovw_v1(desc, kmeans_model, k=k)
        features.append(feat)
    return np.array(features, dtype=np.float32)""")

# --- 1.5 Classifier: Linear SVM ---
add_md(f"""### Module 1.5: Bộ Phân Loại Linear SVM Đa Lớp
Sử dụng `LinearSVC(C=1.0)` với chiến lược One-vs-Rest để phân loại 101 lớp vật thể trên vector BoVW {VOCAB_SIZE} chiều.""")

add_code("""from sklearn.svm import LinearSVC

def train_classifier_v1(X_train, y_train, C=1.0, seed=RANDOM_STATE):
    \"\"\"Huấn luyện bộ phân loại Linear SVM.\"\"\"
    print(f"[v1] Huấn luyện Linear SVM đa lớp (C={C})...")
    clf = LinearSVC(C=C, random_state=seed, max_iter=2000, dual='auto')
    clf.fit(X_train, y_train)
    print(f"[v1] Huấn luyện SVM hoàn tất!")
    return clf""")

# --- 1.6 Execute Chapter 1 ---
add_md(f"""### Module 1.6: [THỰC THI CHƯƠNG 1] Huấn Luyện & Đánh Giá Mốc Baseline v1
Khởi chạy toàn bộ quy trình Chương 1: Trích xuất SIFT -> Tạo từ điển $K={VOCAB_SIZE}$ -> Tính vector BoVW -> Huấn luyện Linear SVM -> Ghi nhận độ chính xác cơ sở và ma trận nhầm lẫn.""")

add_code("""from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

print("=" * 70)
print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM CHƯƠNG 1: MÔ HÌNH GỐC (BASELINE V1 - K={VOCAB_SIZE})")
print(f"Pipeline: SIFT chuẩn -> BoVW (K={VOCAB_SIZE}) -> Chuẩn hóa L2 -> Linear SVM")
print("=" * 70)

t0 = time.time()

# 1. Trích xuất descriptor huấn luyện từ điển
desc_train_v1 = collect_descriptors_v1(X_train)

# 2. Xây dựng từ điển K-Means
vocab_v1 = build_vocabulary_v1(desc_train_v1, k=VOCAB_SIZE)

# 3. Trích xuất vector BoVW cho Train & Test
print(f"[v1] Đang sinh vector BoVW cho {len(X_train)} ảnh Train...")
feat_train_v1 = extract_features_v1(X_train, vocab_v1, k=VOCAB_SIZE)

print(f"[v1] Đang sinh vector BoVW cho {len(X_test)} ảnh Test...")
feat_test_v1 = extract_features_v1(X_test, vocab_v1, k=VOCAB_SIZE)

# 4. Huấn luyện Linear SVM
model_v1 = train_classifier_v1(feat_train_v1, y_train)

# 5. Dự đoán và đo lường hiệu năng
y_pred_v1 = model_v1.predict(feat_test_v1)
time_v1 = time.time() - t0

metrics_v1 = {
    "accuracy": float(accuracy_score(y_test, y_pred_v1)),
    "f1_score": float(f1_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "precision": float(precision_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_v1, average="macro", zero_division=0)),
    "time_seconds": round(time_v1, 2)
}

print("*" * 55)
print(f">>> KẾT QUẢ MỐC CƠ SỞ (BASELINE V1 - K={VOCAB_SIZE}) TRÊN 101 LỚP CALTECH:")
print(f"• Accuracy : {metrics_v1['accuracy'] * 100:.2f}%")
print(f"• Macro F1 : {metrics_v1['f1_score'] * 100:.2f}%")
print(f"• Precision: {metrics_v1['precision'] * 100:.2f}%")
print(f"• Recall   : {metrics_v1['recall'] * 100:.2f}%")
print(f"• Thời gian: {metrics_v1['time_seconds']:.1f} giây")
print("*" * 55)

# Lưu ma trận nhầm lẫn dạng ảnh
cm_v1 = confusion_matrix(y_test, y_pred_v1)
plt.figure(figsize=(10, 8))
plt.imshow(cm_v1, cmap='Blues', interpolation='nearest')
plt.title(f"Chương 1: Ma Trận Nhầm Lẫn 101 Lớp Baseline v1 (K={VOCAB_SIZE}, Acc: {metrics_v1['accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Class Index")
plt.ylabel("True Class Index")
plt.colorbar()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ch1_baseline" / "confusion_matrix_v1.png", dpi=150)
plt.show()""")

# ==============================================================================
# CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (IMPROVED V2)
# ==============================================================================
add_md(f"""# ==========================================================
# CHƯƠNG 2: TRIỂN KHAI MÔ HÌNH CẢI TIẾN (MODEL V2 - IMPROVED)
# ==========================================================
Ở Chương 2, chúng ta giải quyết trực tiếp hạn chế lớn nhất của BoVW v1: **sự biến mất hoàn toàn của thông tin không gian (Spatial Layout)**.  
Với kích thước từ điển quy mô **$K={VOCAB_SIZE}$**, chúng ta áp dụng kiến trúc toàn diện của **Lazebnik, Schmid & Ponce (CVPR 2006)**:
1. **Tuned SIFT & Tọa độ không gian $(x, y)$**: Lưu lại tọa độ chuẩn hóa $[0, 1] \\times [0, 1]$ của từng keypoint.
2. **Spatial Pyramid Matching (SPM)**: Phân chia không gian 3 tầng ($1\\times1, 2\\times2, 4\\times4$), mở rộng vector đặc trưng từ ${VOCAB_SIZE}$ chiều lên **${VOCAB_SIZE} \\times (1 + 4 + 16) = 21 \\times {VOCAB_SIZE} = {21 * VOCAB_SIZE:,}$ chiều**.
3. **Trọng số TF-IDF**: Giảm nhẹ ảnh hưởng của các từ thị giác xuất hiện quá phổ biến ở vùng nền.
4. **Power Normalization & $L_2$ Normalization**: Khử độ lệch phân phối thưa của histogram.
5. **Additive Chi-Square ($\\chi^2$) Kernel SVM**: Ánh xạ xấp xỉ hàm nhân phi tuyến $\\chi^2$ (31.500 chiều) để đo lường độ tương đồng phân phối histogram vượt trội so với tích vô hướng tuyến tính phẳng.""")

# --- 2.1 Tuned SIFT with Normalized Coordinates ---
add_md("""### Module 2.1: SIFT Cải Tiến với Tọa Độ Không Gian Chuẩn Hóa $[0, 1]$
Trích xuất đồng thời descriptor 128 chiều và tọa độ tương đối $(x_{norm}, y_{norm})$ của từng keypoint.""")

add_code("""def extract_sift_spm(image_path, max_dim=MAX_IMAGE_DIM):
    \"\"\"Trích xuất descriptors SIFT cùng tọa độ (x, y) chuẩn hóa trong đoạn [0, 1].\"\"\"
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        
    sift = cv2.SIFT_create(contrastThreshold=0.03, edgeThreshold=10)
    kps, desc = sift.detectAndCompute(img, None)
    if desc is None or len(desc) == 0:
        return None, None
        
    # Chuẩn hóa tọa độ keypoint về khoảng [0, 1]
    coords = np.array([[kp.pt[0] / w, kp.pt[1] / h] for kp in kps], dtype=np.float32)
    return desc, coords""")

# --- 2.2 Spatial Pyramid Matching ---
add_md("""### Module 2.2: Kim Tự Tháp Không Gian Toàn Diện (Spatial Pyramid Matching $1\\times1 + 2\\times2 + 4\\times4$)
Chia ảnh làm 3 mức kim tự tháp theo đúng chuẩn kinh điển của Lazebnik et al. (CVPR 2006):
* **Mức 0 ($1\\times1$)**: Toàn bộ ảnh (vector $K$ chiều, trọng số $w = \\frac{1}{4}$).
* **Mức 1 ($2\\times2$)**: 4 ô lưới không gian (vector $4K$ chiều, trọng số $w = \\frac{1}{4}$).
* **Mức 2 ($4\\times4$)**: 16 ô lưới không gian (vector $16K$ chiều, trọng số $w = \\frac{1}{2}$).  
Tổng số ô lưới: $1 + 4 + 16 = 21$ ô $\\rightarrow$ Tổng kích thước vector: $21 \\times K = 21 \\times 500 = \\mathbf{10.500}$ chiều!""")

add_code("""def compute_spm_features(desc, coords, kmeans_model, k=VOCAB_SIZE):
    \"\"\"Biểu diễn đặc trưng Spatial Pyramid Matching 3 tầng (1x1, 2x2 và 4x4). Tổng: 21*K chiều.\"\"\"
    spm_feature = []
    
    if desc is None or len(desc) == 0:
        return np.zeros(k * 21, dtype=np.float32)
        
    words = kmeans_model.predict(desc.astype(np.float32))
    
    # --- Tầng 0: Toàn bộ ảnh 1x1 (Trọng số w = 0.25) ---
    hist_l0 = np.zeros(k, dtype=np.float32)
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
            hist_sub = np.zeros(k, dtype=np.float32)
            if len(sub_words) > 0:
                counts_sub = Counter(sub_words)
                for w_idx, count in counts_sub.items():
                    hist_sub[w_idx] = count
            norm_sub = np.linalg.norm(hist_sub)
            if norm_sub > 0:
                hist_sub /= norm_sub
            spm_feature.append(hist_sub * 0.25)
            
    # --- Tầng 2: Lưới 4x4 gồm 16 ô không gian (Trọng số w = 0.5) ---
    for r in range(4):
        for c in range(4):
            mask = (
                (coords[:, 0] >= c * 0.25) & (coords[:, 0] < (c + 1) * 0.25) &
                (coords[:, 1] >= r * 0.25) & (coords[:, 1] < (r + 1) * 0.25)
            )
            sub_words = words[mask]
            hist_sub = np.zeros(k, dtype=np.float32)
            if len(sub_words) > 0:
                counts_sub = Counter(sub_words)
                for w_idx, count in counts_sub.items():
                    hist_sub[w_idx] = count
            norm_sub = np.linalg.norm(hist_sub)
            if norm_sub > 0:
                hist_sub /= norm_sub
            spm_feature.append(hist_sub * 0.5)
            
    # Nối tất cả 21 histogram lại thành vector 21 * K chiều
    combined = np.concatenate(spm_feature)
    
    # Power normalization (Signed Square Root)
    combined = np.sign(combined) * np.sqrt(np.abs(combined))
    
    # L2 Normalization cuối cùng
    norm_total = np.linalg.norm(combined)
    if norm_total > 0:
        combined /= norm_total
        
    return combined

def extract_all_spm(image_paths, kmeans_model, k=VOCAB_SIZE):
    \"\"\"Sinh vector SPM 21*K chiều cho toàn bộ tập ảnh.\"\"\"
    features = []
    for path in image_paths:
        desc, coords = extract_sift_spm(path)
        feat = compute_spm_features(desc, coords, kmeans_model, k=k)
        features.append(feat)
    return np.array(features, dtype=np.float32)""")

# --- 2.3 TF-IDF Weighting ---
add_md("""### Module 2.3: Trọng Số TF-IDF (Term Frequency - Inverse Document Frequency)
Tính trọng số nghịch đảo tần suất xuất hiện trong tập huấn luyện để giảm thiểu ảnh hưởng của các từ thị giác nền lặp đi lặp lại.""")

add_code("""from sklearn.feature_extraction.text import TfidfTransformer

def apply_tfidf_weighting(train_feats, test_feats):
    \"\"\"Áp dụng chuẩn hóa trọng số TF-IDF lên vector đặc trưng.\"\"\"
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    train_tfidf = tfidf.fit_transform(train_feats).toarray().astype(np.float32)
    test_tfidf = tfidf.transform(test_feats).toarray().astype(np.float32)
    return train_tfidf, test_tfidf, tfidf""")

# --- 2.4 Additive Chi-Squared Kernel SVM ---
add_md("""### Module 2.4: Bộ Phân Loại Additive Chi-Square ($\\chi^2$) Kernel SVM
Sử dụng `AdditiveChi2Sampler` từ `sklearn.kernel_approximation` để ánh xạ vector đặc trưng sang không gian xấp xỉ hàm nhân $\\chi^2$. Phép biến đổi này giúp mô hình phân loại với độ chính xác của hàm nhân $\\chi^2$ phi tuyến nhưng vẫn duy trì tốc độ huấn luyện tuyến tính cực nhanh.""")

add_code("""from sklearn.kernel_approximation import AdditiveChi2Sampler

def train_chi2_svm(X_train_spm, y_train, C=1.0, seed=RANDOM_STATE):
    \"\"\"Biến đổi sang không gian Chi2 và huấn luyện LinearSVC.\"\"\"
    print("[v2] Đang ánh xạ đặc trưng sang không gian Additive Chi2...")
    chi2_sampler = AdditiveChi2Sampler(sample_steps=2)
    X_train_chi2 = chi2_sampler.fit_transform(np.maximum(0, X_train_spm))
    
    print(f"[v2] Huấn luyện LinearSVC trên không gian Chi2 (Số chiều mới: {X_train_chi2.shape[1]})...")
    clf = LinearSVC(C=C, random_state=seed, max_iter=2000, dual='auto')
    clf.fit(X_train_chi2, y_train)
    print("[v2] Huấn luyện Chi2-SVM hoàn tất!")
    return clf, chi2_sampler""")

# --- 2.5 Execute Chapter 2 ---
add_md("""### Module 2.5: [THỰC THI CHƯƠNG 2] Huấn Luyện & Đánh Giá Mô Hình Cải Tiến v2
Khởi chạy quy trình Chương 2: Trích xuất SIFT + Tọa độ không gian -> SPM $1\\times1 + 2\\times2 + 4\\times4$ (21 ô không gian, $21 \\times K$ chiều) -> Trọng số TF-IDF -> Ánh xạ $\\chi^2$ Kernel -> Huấn luyện SVM -> Đo lường và vẽ ma trận nhầm lẫn.""")

add_code("""print("=" * 70)
print(f"BẮT ĐẦU CHẠY THỰC NGHIỆM CHƯƠNG 2: MÔ HÌNH CẢI TIẾN (IMPROVED V2 - K={VOCAB_SIZE})")
print(f"Pipeline: SIFT tọa độ -> SPM (1x1 + 2x2 + 4x4, 21 ô, {21 * VOCAB_SIZE} dim) -> TF-IDF -> Additive Chi2 SVM")
print("=" * 70)

t0_v2 = time.time()

# 1. Trích xuất đặc trưng SPM (1x1 + 2x2 + 4x4 = 21 * K chiều)
print(f"[v2] Đang sinh vector SPM cho {len(X_train)} ảnh Train...")
spm_train_raw = extract_all_spm(X_train, vocab_v1, k=VOCAB_SIZE)

print(f"[v2] Đang sinh vector SPM cho {len(X_test)} ảnh Test...")
spm_test_raw = extract_all_spm(X_test, vocab_v1, k=VOCAB_SIZE)

# 2. Áp dụng trọng số TF-IDF
spm_train_tfidf, spm_test_tfidf, tfidf_model = apply_tfidf_weighting(spm_train_raw, spm_test_raw)

# 3. Huấn luyện Additive Chi2 Kernel SVM
model_v2, chi2_sampler = train_chi2_svm(spm_train_tfidf, y_train, C=1.0)

# 4. Dự đoán trên tập kiểm tra
spm_test_chi2 = chi2_sampler.transform(np.maximum(0, spm_test_tfidf))
y_pred_v2 = model_v2.predict(spm_test_chi2)
time_v2 = time.time() - t0_v2

metrics_v2 = {
    "accuracy": float(accuracy_score(y_test, y_pred_v2)),
    "f1_score": float(f1_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "precision": float(precision_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "recall": float(recall_score(y_test, y_pred_v2, average="macro", zero_division=0)),
    "time_seconds": round(time_v2, 2)
}

print("*" * 55)
print(f">>> KẾT QUẢ MÔ HÌNH CẢI TIẾN (IMPROVED V2 - K={VOCAB_SIZE}) TRÊN 101 LỚP:")
print(f"• Accuracy : {metrics_v2['accuracy'] * 100:.2f}%")
print(f"• Macro F1 : {metrics_v2['f1_score'] * 100:.2f}%")
print(f"• Precision: {metrics_v2['precision'] * 100:.2f}%")
print(f"• Recall   : {metrics_v2['recall'] * 100:.2f}%")
print(f"• Thời gian: {metrics_v2['time_seconds']:.1f} giây")
print("*" * 55)

# Lưu ma trận nhầm lẫn v2 dạng ảnh
cm_v2 = confusion_matrix(y_test, y_pred_v2)
plt.figure(figsize=(10, 8))
plt.imshow(cm_v2, cmap='Greens', interpolation='nearest')
plt.title(f"Chương 2: Ma Trận Nhầm Lẫn 101 Lớp Improved v2 (K={VOCAB_SIZE}+SPM+Chi2, Acc: {metrics_v2['accuracy']*100:.1f}%)", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Class Index")
plt.ylabel("True Class Index")
plt.colorbar()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "ch2_improved" / "confusion_matrix_v2.png", dpi=150)
plt.show()""")

# ==============================================================================
# CHƯƠNG 3: BÁO CÁO ĐỐI ĐẦU (ABLATION STUDY), DEMO & EXPORT
# ==============================================================================
add_md(f"""# ==========================================================
# CHƯƠNG 3: BÁO CÁO ĐỐI ĐẦU, PHÂN TÍCH LỖI & ĐÓNG GÓI OUTPUT
# ==========================================================
Tổng hợp so sánh trực diện giữa **Baseline v1 ($K={VOCAB_SIZE}$)** và **Improved v2 ($K={VOCAB_SIZE}$ + SPM 3 Tầng + $\\chi^2$)** trên toàn bộ 101 lớp của Caltech-101 để trả lời câu hỏi: *Việc kết hợp cấu trúc không gian kim tự tháp (21 ô lưới) và hàm nhân $\\chi^2$ phi tuyến mang lại bao nhiêu % cải thiện?*""")

add_code("""# 1. Bảng số liệu tổng hợp đối đầu
summary_data = {
    "Chỉ số Đánh Giá": ["Accuracy (Độ chính xác)", "Macro F1-Score", "Macro Precision", "Macro Recall", "Thời gian huấn luyện"],
    f"Chương 1: Baseline v1 (K={VOCAB_SIZE})": [
        f"{metrics_v1['accuracy'] * 100:.2f}%",
        f"{metrics_v1['f1_score'] * 100:.2f}%",
        f"{metrics_v1['precision'] * 100:.2f}%",
        f"{metrics_v1['recall'] * 100:.2f}%",
        f"{metrics_v1['time_seconds']:.1f}s"
    ],
    f"Chương 2: Improved v2 (K={VOCAB_SIZE} + SPM + Chi2)": [
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

print("=" * 70)
print(f"BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU TRÊN TOÀN BỘ {len(classes)} LỚP CALTECH-101")
print("=" * 70)

try:
    import pandas as pd
    from IPython.display import display
    summary_df = pd.DataFrame(summary_data)
    display(summary_df)
except ImportError:
    for k, v1, v2, diff in zip(summary_data["Chỉ số Đánh Giá"], summary_data[f"Chương 1: Baseline v1 (K={VOCAB_SIZE})"], summary_data[f"Chương 2: Improved v2 (K={VOCAB_SIZE} + SPM + Chi2)"], summary_data["Mức Độ Cải Thiện (+/-)"]):
        print(f"• {k:<25}: v1 = {v1:<8} | v2 = {v2:<8} | Cải thiện: {diff}")

# 2. Biểu đồ cột so sánh trực quan
metric_keys = ["Accuracy", "Macro F1", "Precision", "Recall"]
v1_scores = [metrics_v1["accuracy"] * 100, metrics_v1["f1_score"] * 100, metrics_v1["precision"] * 100, metrics_v1["recall"] * 100]
v2_scores = [metrics_v2["accuracy"] * 100, metrics_v2["f1_score"] * 100, metrics_v2["precision"] * 100, metrics_v2["recall"] * 100]

x = np.arange(len(metric_keys))
width = 0.35

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, v1_scores, width, label=f'Chương 1: Baseline v1 (K={VOCAB_SIZE})', color='#5c6bc0')
rects2 = plt.bar(x + width/2, v2_scores, width, label=f'Chương 2: Improved v2 (K={VOCAB_SIZE}+SPM+Chi2)', color='#00c853')

plt.ylabel('Tỷ Lệ Phần Trăm (%)', fontsize=12)
plt.title(f'So Sánh Hiệu Năng Đối Đầu Trên {len(classes)} Lớp Caltech-101 (K={VOCAB_SIZE})', fontsize=14, fontweight='bold', pad=15)
plt.xticks(x, metric_keys, fontsize=11)
plt.ylim(0, 105)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='lower right', fontsize=11)

for rect in rects1:
    height = rect.get_height()
    plt.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10)
for rect in rects2:
    height = rect.get_height()
    plt.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width()/2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison" / "model_comparison_bar.png", dpi=150)
plt.show()

# 3. Phân tích lỗi (Error Analysis): Số ca v1 đoán sai được v2 sửa đúng
y_test_arr = np.array(y_test)
v1_wrong_idx = np.where(y_pred_v1 != y_test_arr)[0]
v2_fixed_idx = np.where((y_pred_v1 != y_test_arr) & (y_pred_v2 == y_test_arr))[0]

print(f"\\n[Error Analysis] Tổng số mẫu ảnh tập Test: {len(y_test)} ảnh")
print(f"• Số lượng mẫu v1 đoán sai : {len(v1_wrong_idx)} ảnh (Tỷ lệ lỗi: {len(v1_wrong_idx)/len(y_test)*100:.2f}%)")
print(f"• Số lượng mẫu v2 đoán sai : {np.sum(y_pred_v2 != y_test_arr)} ảnh (Tỷ lệ lỗi: {np.sum(y_pred_v2 != y_test_arr)/len(y_test)*100:.2f}%)")
print(f"• Số mẫu v1 đoán sai đã được v2 SỬA ĐÚNG: {len(v2_fixed_idx)} ảnh!")
if len(v1_wrong_idx) > 0:
    print(f"• Tỷ lệ khắc phục lỗi (Relative Error Reduction): {len(v2_fixed_idx)/len(v1_wrong_idx)*100:.1f}%")""")

# --- Demo: Single Image Inference ---
add_md("""### Module 3.2: Dự Đoán Ảnh Bất Kỳ (Single Image Inference Demo)
Cho phép truyền vào đường dẫn của bất kỳ file ảnh nào để chạy qua pipeline mô hình cải tiến v2 và đưa ra dự đoán cùng ảnh minh họa.""")

add_code("""from PIL import Image

def predict_single_image(image_path, model=model_v2, sampler=chi2_sampler, tfidf=tfidf_model, kmeans=vocab_v1, class_names=classes):
    \"\"\"Dự đoán nhãn cho 1 bức ảnh bất kỳ bằng mô hình cải tiến v2.\"\"\"
    path = Path(image_path)
    if not path.is_file():
        print(f"Không tìm thấy file ảnh: {path}")
        return None
        
    desc, coords = extract_sift_spm(path)
    feat = compute_spm_features(desc, coords, kmeans, k=VOCAB_SIZE).reshape(1, -1)
    feat_tfidf = tfidf.transform(feat).toarray().astype(np.float32)
    feat_chi2 = sampler.transform(np.maximum(0, feat_tfidf))
    
    pred_idx = model.predict(feat_chi2)[0]
    pred_label = class_names[pred_idx]
    
    # Hiển thị ảnh
    plt.figure(figsize=(5, 5))
    img_pil = Image.open(path).convert('RGB')
    plt.imshow(img_pil)
    plt.title(f"Dự đoán: {pred_label}", fontsize=13, fontweight='bold', color='green')
    plt.axis('off')
    plt.show()
    return pred_label

# Thử nghiệm dự đoán trên 1 ảnh ngẫu nhiên từ tập Test
if len(X_test) > 0:
    sample_img = X_test[0]
    true_cls = classes[y_test[0]]
    print(f"Thử nghiệm dự đoán ảnh mẫu (Nhãn thực tế: {true_cls}):")
    predict_single_image(sample_img)""")

# --- Module 3.3: Export ZIP ---
add_md("""### Module 3.3: Đóng Gói Toàn Bộ Kết Quả (Export ZIP Output)
Tự động nén toàn bộ kết quả, biểu đồ so sánh, ma trận nhầm lẫn và file JSON tổng hợp thành file `caltech101_bovw_results.zip` nằm trong `/kaggle/working/` để tải về máy chỉ với 1 click.""")

add_code("""import shutil
import json

final_report = {
    "dataset": "Caltech-101 (Full 101 Categories)",
    "classes_count": len(classes),
    "vocab_size": VOCAB_SIZE,
    "train_images": len(X_train),
    "test_images": len(X_test),
    "ch1_baseline_v1": metrics_v1,
    "ch2_improved_v2": metrics_v2,
    "improvement": {
        "accuracy_gain": round((metrics_v2['accuracy'] - metrics_v1['accuracy']) * 100, 2),
        "f1_gain": round((metrics_v2['f1_score'] - metrics_v1['f1_score']) * 100, 2),
        "time_diff_seconds": round(metrics_v2['time_seconds'] - metrics_v1['time_seconds'], 2)
    }
}

with open(RESULTS_DIR / "summary_report.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=2, ensure_ascii=False)

zip_output = WORKING_DIR / "caltech101_bovw_results"
shutil.make_archive(str(zip_output), 'zip', str(OUTPUT_DIR))

print("=" * 70)
print(f"ĐÃ HOÀN TẤT VÀ ĐÓNG GÓI TOÀN BỘ KẾT QUẢ THỰC NGHIỆM:")
print(f"• File nén ZIP: {zip_output}.zip")
print(f"• Bạn có thể tải file này về từ mục 'Output' ở bảng điều khiển bên phải Kaggle!")
print("=" * 70)""")

# ==============================================================================
# GHI RA FILE NOTEBOOK
# ==============================================================================
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

output_path = Path("kaggle_bovw.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Đã tạo thành công notebook {output_path} với {len(cells)} cells!")

