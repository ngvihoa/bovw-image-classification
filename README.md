# Phân loại ảnh Caltech-101 bằng SIFT, BoVW, SPM và SVM

Đồ án xây dựng một hệ thống phân loại ảnh bằng **thị giác máy tính truyền thống**, không sử dụng CNN hay Vision Transformer. Thực nghiệm chính được triển khai dưới dạng Kaggle Notebook trên toàn bộ **101 lớp vật thể của Caltech-101**, đồng thời so sánh mô hình Bag of Visual Words cơ sở với phiên bản cải tiến có bổ sung thông tin không gian.

## Tổng quan

Hai pipeline được đánh giá với cùng từ điển thị giác có kích thước `K = 500`:

```text
Baseline v1
Ảnh → SIFT → MiniBatchKMeans → BoVW → L2 Normalization → Linear SVM

Improved v2
Ảnh → SIFT + tọa độ keypoint → MiniBatchKMeans
    → SPM (1×1 + 2×2 + 4×4) → TF-IDF + Power/L2 Normalization
    → Additive Chi-Square Map → Linear SVM
```

Mô hình v2 sử dụng **Spatial Pyramid Matching (SPM)** gồm 21 vùng không gian. Với `K = 500`, mỗi ảnh được biểu diễn bởi vector SPM `21 × 500 = 10.500` chiều trước khi được ánh xạ sang không gian đặc trưng Additive Chi-Square.

## Bộ dữ liệu và thiết lập thực nghiệm

- Dataset: **Caltech-101**.
- Số lớp: **101**; loại bỏ thư mục `BACKGROUND_Google`.
- Tập huấn luyện: **24 ảnh/lớp**, tổng cộng **2.424 ảnh**.
- Tập kiểm tra: **6 ảnh/lớp**, tổng cộng **606 ảnh**.
- Tổng số ảnh được sử dụng: **3.030 ảnh**.
- Random seed: `42`.
- Visual vocabulary: `K = 500`.
- Môi trường chạy chính: Kaggle Notebook, CPU.

Cách chia cố định theo từng lớp giúp hai mô hình được đánh giá trên cùng một tập dữ liệu cân bằng.

## Kết quả

| Chỉ số | Baseline v1 | Improved v2 | Chênh lệch |
|---|---:|---:|---:|
| Accuracy | 25,74% | **40,10%** | **+14,36 điểm %** |
| Macro F1-score | 23,57% | **37,90%** | **+14,33 điểm %** |
| Macro Precision | 24,22% | **39,40%** | **+15,18 điểm %** |
| Macro Recall | 25,74% | **40,10%** | **+14,36 điểm %** |
| Thời gian huấn luyện SVM | 66,3 giây | **64,8 giây** | -1,5 giây |

Trong thiết lập này, phiên bản cải tiến tăng Accuracy từ **25,74% lên 40,10%**, tương đương mức tăng tương đối khoảng **55,8%** so với baseline. Các số liệu trên là kết quả của cấu hình thực nghiệm 24 train/6 test mỗi lớp được trình bày trong báo cáo dự án.

## Chạy demo trên Kaggle

Notebook chính: [`kaggle_bovw_caltech_101.ipynb`](kaggle_bovw_caltech_101.ipynb)

1. Tạo một Kaggle Notebook mới hoặc tải notebook của dự án lên Kaggle.
2. Chọn **Add Input** và thêm dataset **Caltech 101** có thư mục `101_ObjectCategories`.
3. Bảo đảm Accelerator được đặt thành **None**; pipeline có thể chạy hoàn toàn bằng CPU.
4. Chọn **Run All**.
5. Xem bảng so sánh, ma trận nhầm lẫn, demo dự đoán và tải file kết quả `.zip` từ `/kaggle/working`.

Notebook tự động dò thư mục dữ liệu bên trong `/kaggle/input`, loại bỏ lớp nền, chia dữ liệu và thực thi toàn bộ quy trình huấn luyện–đánh giá.

## Chạy phiên bản cơ sở trên máy cá nhân

Yêu cầu Python 3.10 trở lên.

```bash
git clone <repository-url>
cd bovw-image-classification

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tải Caltech-101 và giải nén theo cấu trúc:

```text
data/raw/101_ObjectCategories/
├── airplanes/
├── Motorbikes/
├── Faces/
└── ...
```

Kiểm tra OpenCV SIFT:

```bash
python scripts/test.py
```

Chạy pipeline v1 trên năm lớp được khai báo trong `config.py`:

```bash
python scripts/train.py --vocab_size 500
```

Có thể thử nhiều kích thước từ điển:

```bash
for K in 50 100 200 500; do
    python scripts/train.py --vocab_size "$K"
done
```

> `scripts/train.py` là thí nghiệm v1 cục bộ trên 5 lớp. Thực nghiệm và demo đầy đủ 101 lớp nằm trong notebook Kaggle.

## Cấu trúc dự án

```text
.
├── kaggle_bovw_caltech_101.ipynb   # Demo và thực nghiệm chính trên 101 lớp
├── kaggle_bovw_swedish_leaf.ipynb  # Notebook thử nghiệm Swedish Leaf
├── config.py                       # Cấu hình pipeline v1 cục bộ
├── requirements.txt
├── src/
│   ├── dataset.py                  # Đọc và chia dữ liệu
│   ├── sift.py                     # Trích xuất SIFT và tọa độ keypoint
│   ├── vocabulary.py               # Xây dựng visual vocabulary
│   ├── bovw.py                     # Biểu diễn Bag of Visual Words
│   ├── spm.py                      # Spatial Pyramid Matching và TF-IDF
│   ├── classifier.py               # SVM tuyến tính và Additive Chi-Square
│   └── evaluate.py                 # Metrics và confusion matrix
├── scripts/
│   ├── train.py                    # Huấn luyện baseline v1 cục bộ
│   ├── train_v2.py                 # Pipeline module hóa cho model v2
│   └── create_caltech_notebook.py  # Sinh notebook Caltech-101
├── artifacts/                      # Vocabulary, features và model đã lưu
├── results/                        # Metrics và biểu đồ
└── docs/
    ├── report.md                   # Báo cáo thực nghiệm 101 lớp
    └── report_research_paper.md    # Báo cáo phiên bản 5 lớp
```

## Công nghệ sử dụng

- Python
- OpenCV và SIFT
- NumPy
- scikit-learn
- MiniBatchKMeans
- Spatial Pyramid Matching
- TF-IDF và Additive Chi-Square feature map
- Linear SVM
- Matplotlib
- Kaggle Notebook

## Hạn chế

- BoVW lượng tử hóa descriptor nên làm mất một phần thông tin cục bộ.
- SPM chỉ mô tả bố cục theo các lưới cố định.
- Mỗi lớp chỉ sử dụng 24 ảnh huấn luyện, vì vậy bài toán 101 lớp vẫn mang tính few-shot.
- Kết quả phụ thuộc vào cách chia dữ liệu và các siêu tham số; cần cross-validation để có kết luận tổng quát hơn.
- Pipeline đặc trưng thủ công phù hợp cho mục đích học tập và khả năng giải thích, nhưng không nhằm cạnh tranh với các mô hình deep learning hiện đại.

## Tài liệu tham khảo

1. D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” *IJCV*, 2004.
2. G. Csurka et al., “Visual Categorization with Bags of Keypoints,” *ECCV Workshop*, 2004.
3. S. Lazebnik, C. Schmid, and J. Ponce, “Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories,” *CVPR*, 2006.
4. A. Vedaldi and A. Zisserman, “Efficient Additive Kernels via Explicit Feature Maps,” *IEEE TPAMI*, 2012.

Chi tiết phương pháp và phân tích kết quả xem tại [`docs/report.md`](docs/report.md).
