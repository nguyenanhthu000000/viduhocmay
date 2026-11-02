## 1. Tổng quan nhiệm vụ (Task Overview)
Task 3 đóng vai trò xây dựng hệ thống nền tảng (**Baseline & Classical Machine Learning**) cho toàn bộ đồ án phát hiện giọng nói giả. Nhiệm vụ cốt lõi bao gồm:
* **Chuyển đổi tín hiệu âm thanh:** Biến đổi các file âm thanh thô thành vector đặc trưng âm học chuyên sâu thông qua kỹ thuật **MFCC** và **LFCC**.
* **Huấn luyện mô hình:** Xây dựng hệ thống học máy cổ điển gồm 4 mô hình (*Logistic Regression, SVM, Random Forest, XGBoost*).
* **Kiểm định chéo an toàn:** Áp dụng chiến lược **GroupKFold** chia theo `speaker_id` để ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (Data Leakage) giữa người nói tập train và test.
* **Đánh giá hiệu năng:** Xuất các chỉ số chuẩn quốc tế (**ROC-AUC, F1-Score, EER**) kèm theo biểu đồ trực quan và bảng kết quả thực nghiệm.

## 2. Cấu trúc các file và Chi tiết kỹ thuật

| File | Nội dung & Kỹ thuật áp dụng |
| :--- | :--- |
| `src/features_handcrafted.py` | • Trích xuất 13 hệ số **MFCC** và **LFCC** chuẩn hóa có đầy đủ cả `mean` (trung bình) và `std` (độ lệch chuẩn).<br>• Sử dụng bộ lọc tuyến tính (Linear-frequency filterbank) và biến đổi DCT cho LFCC.<br>• Kiểm tra và xử lý linh hoạt tên cột trong metadata để tránh lỗi đường dẫn, đóng gói nén vào `features/mfcc_features.npz`. |
| `src/train_classical.py` | • Thực hiện ghép dữ liệu chuẩn xác qua `clip_id` và lọc đúng tập `split == 'train'`.<br>• Sử dụng **GroupKFold (5 folds)** dựa trên `speaker_id` để tìm kiếm siêu tham số tối ưu.<br>• Tự động `fit()` lại mô hình tốt nhất trên **toàn bộ tập train** trước khi lưu artifact vào `models/best_model.pkl` để phục vụ tích hợp Web Demo (Task 5). |
| `src/evaluate.py` | • Tự động nạp mô hình, lọc tập kiểm tra `test_seen` và tính toán các chỉ số chuyên sâu.<br>• Tính điểm cân bằng lỗi **EER (Equal Error Rate)** thông qua nội suy hàm số.<br>• Xuất bảng kết quả chi tiết ra file `results/test_seen_predictions.csv` và vẽ biểu đồ đường cong ROC (`results/roc_curve.png`). |
| `requirements.txt` | • Danh sách các thư viện Python cần thiết (`librosa`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`,...). |

---

## 3. Hướng dẫn cài đặt và tái lập thực nghiệm

Chạy toàn bộ các lệnh sau từ **thư mục gốc** của dự án:

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```
### Bước 2: Trích xuất đặc trưng âm thanh (Handcrafted Features)

Chạy lệnh sau tại thư mục gốc của dự án:

```bash
python src/features_handcrafted.py
```

**Kết quả:** Tạo tệp đặc trưng âm thanh đã được trích xuất và lưu dưới dạng NumPy nén tại:

```text
features/mfcc_features.npz
```

### Bước 3: Huấn luyện mô hình và tìm kiếm siêu tham số

```bash
python src/train_classical.py
```

**Kết quả:**

- Lưu mô hình tối ưu tại `models/best_model.pkl`.
- Xuất trọng số đặc trưng tại `results/feature_importance_weights.csv`.

### Bước 4: Đánh giá mô hình và xuất báo cáo

```bash
python src/evaluate.py
```

**Kết quả:** Xuất các biểu đồ trực quan và báo cáo đánh giá hiệu năng mô hình vào thư mục `results/`.

---

## 4. Báo cáo kết quả thực nghiệm (Experimental Results)

Phần này trình bày kết quả đánh giá mô hình trên tập dữ liệu kiểm thử `test_seen`.

- **Mô hình tối ưu:** SVM với kernel RBF, \(C = 0.1\) hoặc mô hình Ensemble, tùy theo mô hình thực tế được lựa chọn trong quá trình huấn luyện.
- **Tổng số mẫu đánh giá:** 99 mẫu, bao gồm hai lớp Real và Fake.

### 4.1. Các chỉ số đánh giá

| Chỉ số đánh giá | Giá trị thực nghiệm | Ý nghĩa |
|---|---:|---|
| **ROC-AUC Score** | **0.8018** | Đánh giá khả năng phân biệt giữa giọng nói thật và giọng nói giả. |
| **F1-Score** | **0.7129** | Đánh giá sự cân bằng giữa Precision và Recall. |
| **Equal Error Rate (EER)** | **0.3000** | Tỷ lệ lỗi tại ngưỡng mà tỷ lệ chấp nhận sai và tỷ lệ từ chối sai bằng nhau. |

### 4.2. Ma trận nhầm lẫn (Confusion Matrix)

Ma trận nhầm lẫn thể hiện số lượng mẫu được phân loại đúng và phân loại sai giữa hai lớp Real và Fake.

```text
                  Dự đoán Real    Dự đoán Fake
Nhãn Real              34              16
Nhãn Fake              13              36
```

**Nhận xét:**

- Có 34 mẫu Real được phân loại đúng và 16 mẫu Real bị phân loại nhầm thành Fake.
- Có 36 mẫu Fake được phân loại đúng và 13 mẫu Fake bị phân loại nhầm thành Real.
- Tổng cộng có 70/99 mẫu được phân loại đúng, tương ứng độ chính xác khoảng 70,71%.

Biểu đồ ma trận nhầm lẫn được lưu tại:

`results/confusion_matrix.png`

### 4.3. Giải thích các chỉ số

- **ROC-AUC:** Đo lường khả năng phân biệt hai lớp trên nhiều ngưỡng phân loại khác nhau. Giá trị càng gần 1 thì khả năng phân biệt nhìn chung càng tốt.
- **F1-Score:** Là trung bình điều hòa của Precision và Recall, phù hợp khi cần xem xét đồng thời hai chỉ số.
- **EER:** Là điểm mà tỷ lệ chấp nhận sai (FAR) bằng tỷ lệ từ chối sai (FRR). Giá trị càng thấp thường cho thấy hiệu năng xác thực càng tốt.

---

## 5. Danh mục các tệp đầu ra (Generated Artifacts)

Sau khi hoàn tất quy trình xử lý, hệ thống tạo ra các tệp kết quả sau trong thư mục `results/`:

| STT | Tệp đầu ra | Mô tả |
|---:|---|---|
| 1 | `results/confusion_matrix.png` | Biểu đồ ma trận nhầm lẫn, thể hiện số lượng mẫu được phân loại đúng và sai giữa hai lớp Real và Fake. |
| 2 | `results/roc_curve.png` | Biểu đồ đường cong ROC, thể hiện mối quan hệ giữa TPR và FPR tại các ngưỡng phân loại khác nhau. |
| 3 | `results/evaluation_metrics_summary.csv` | Bảng tổng hợp các chỉ số đánh giá như ROC-AUC, EER và F1-Score. |
| 4 | `results/feature_importance_weights.csv` | Bảng trọng số hoặc mức độ đóng góp của các đặc trưng MFCC/LFCC theo phương pháp được triển khai. |
| 5 | `results/test_seen_predictions.csv` | Bảng kết quả dự đoán trên tập kiểm thử, bao gồm nhãn dự đoán và xác suất dự đoán nếu được xuất bởi chương trình. |

Ngoài ra, các tệp phục vụ huấn luyện và triển khai mô hình được lưu tại:

- `features/mfcc_features.npz`: dữ liệu đặc trưng âm thanh.
- `models/best_model.pkl`: mô hình học máy đã được lựa chọn sau quá trình huấn luyện.

**Lưu ý:** Danh sách trên mô tả các tệp dự kiến của pipeline. Tên và nội dung thực tế cần khớp với những gì mã nguồn tạo ra.

---

## 6. Quy trình đồng bộ Git và nộp bài (Submission Workflow)

Quy trình dưới đây được sử dụng để cập nhật nhánh `main`, tạo nhánh tính năng cho Task 3 và đưa các tệp đã hoàn thiện từ nhánh `Task3` sang nhánh mới.

### Bước 1: Cập nhật nhánh `main`

```bash
git checkout main
git pull origin main
```

### Bước 2: Tạo nhánh tính năng cho Task 3

```bash
git checkout -b feature/task3-final-update
```

### Bước 3: Lấy các tệp đã hoàn thiện từ nhánh `Task3`

```bash
git checkout Task3 -- src/ features/ models/ results/ requirements.txt .gitignore README.md
```

Lệnh trên lấy các đường dẫn được chỉ định từ nhánh `Task3` và cập nhật chúng vào nhánh hiện tại. Những thay đổi khác trên nhánh hiện tại không bị thay thế nếu nằm ngoài các đường dẫn được chỉ định.

### Bước 4: Kiểm tra và commit thay đổi

```bash
git status
git add .
git commit -m "Complete Task 3: Feature extraction, model training and evaluation"
```

### Bước 5: Đẩy nhánh lên GitHub

```bash
git push -u origin feature/task3-final-update
```
---

## 7. Hướng dẫn tích hợp inference cho App Demo (Task 5 / UI Integration)

Để tích hợp mô hình Handcrafted (Task 3) vào giao diện demo chung của nhóm mà không bị xung đột với các luồng Deep Learning khác, chúng tôi cung cấp lớp wrapper độc lập **`HandcraftedPredictor`**.

### 7.1. Cấu trúc module tích hợp
- `src/inference_handcrafted.py`: Tự động hóa toàn bộ quy trình `Trích xuất đặc trưng -> StandardScaler -> Dự đoán`.

### 7.2. Code mẫu gọi trực tiếp trong App Demo
```python
from src.inference_handcrafted import HandcraftedPredictor

# 1. Khởi tạo predictor (load sẵn mô hình và scaler)
predictor = HandcraftedPredictor(
    model_path="models/best_model.pkl", 
    scaler_path="models/scaler.pkl"  # Hoặc đường dẫn tương đương nếu có lưu riêng scaler
)

# 2. Dự đoán cho một file âm thanh đầu vào từ giao diện
result = predictor.predict_audio("path/to/uploaded_audio.wav")

if result["status"] == "success":
    print(f"Nhãn dự đoán: {result['prediction']}")
    print(f"Xác suất: {result['probability']}")
else:
    print(f"Lỗi xử lý: {result['message']}")
