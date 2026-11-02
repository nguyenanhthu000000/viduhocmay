# Task 3 – Baseline & Classical Machine Learning (phát hiện giọng nói giả tiếng Việt)

Task 3 xây dựng hệ thống nền tảng cho đồ án phát hiện giọng nói giả:

- Trích đặc trưng thủ công **MFCC + LFCC** (vector 52 chiều) từ file âm thanh.
- Huấn luyện 4 mô hình học máy cổ điển: **Logistic Regression, SVM (RBF), Random Forest, XGBoost**.
- Chọn siêu tham số bằng **GroupKFold theo `speaker_id`** để tránh rò rỉ người nói giữa train và validation.
- Đánh giá trên tập `test_seen` bằng **ROC-AUC, EER, F1**, kèm ma trận nhầm lẫn và đường cong ROC.
- Cung cấp lớp `HandcraftedPredictor` để tích hợp vào app demo (Task 5).

> **Trạng thái kết quả:** xem mục 7. Các số liệu cũ (AUC 0,8018 / F1 0,7129 / EER 0,30 trên 99 mẫu) đã bị **gỡ bỏ** vì
> được tính từ nhãn không hợp lệ (nhãn được sinh theo thứ tự dòng khi dữ liệu chỉ có một lớp). Kết quả hợp lệ sẽ được cập nhật sau khi chạy lại trên metadata đã chuẩn hóa.

---

## 1. Cấu trúc thư mục

```
├── data/
│   ├── metadata_real.csv        # giọng thật (đã đúng chuẩn 9 cột)
│   ├── metadata_fake.csv        # giọng giả (từ Task 2)
│   └── metadata_combined.csv    # do src/prepare_metadata.py tạo ra
├── src/
│   ├── prepare_metadata.py      # gộp + kiểm tra metadata
│   ├── common.py                # nạp metadata + đặc trưng, ghép theo clip_id
│   ├── features_handcrafted.py  # trích MFCC/LFCC
│   ├── train_classical.py       # huấn luyện + chọn siêu tham số
│   ├── evaluate.py              # đánh giá trên test_seen
│   └── inference_handcrafted.py # wrapper cho app demo
├── features/mfcc_features.npz   # đặc trưng đã trích (clip_id, X, feature_names)
├── models/                      # best_model.pkl, model_config.json
├── results/                     # bảng và biểu đồ kết quả
└── requirements.txt
```

File audio (`.wav`) **không** được đưa lên repo (xem `.gitignore`).

## 2. Dữ liệu và quy ước

`data/metadata_combined.csv` có đúng 9 cột:

| Cột | Ý nghĩa |
|---|---|
| `clip_id` | Mã duy nhất của clip (khóa ghép với đặc trưng) |
| `path` | Đường dẫn tương đối tới file audio |
| `label` | `real` hoặc `fake` (mã hóa: real = 0, fake = 1) |
| `speaker_id` | Người nói. Với giọng giả: **người bị clone** (người nói giọng mẫu) |
| `source` | Nguồn dữ liệu (ví dụ `vivos`) |
| `generator` | Công cụ sinh giọng giả; `none` với giọng thật |
| `channel` | Kênh thu (ví dụ `clean`) |
| `split` | `train`, `dev`, `test_seen`, `test_unseen`, `test_phone` |
| `duration` | Độ dài clip (giây) |

Các quy tắc được code **kiểm tra và dừng nếu vi phạm** (không tự sửa hay tự gán):

- `clip_id` không trùng; `label` chỉ nhận `real`/`fake`; `split` thuộc danh sách trên.
- Một người nói chỉ nằm trong **một** split (chống rò rỉ).
- `train` và `test_seen` phải có đủ cả real và fake.
- Giọng giả trong `train` phải có **ít nhất 2 `speaker_id`** (để GroupKFold có fold chứa cả hai lớp).
- Mọi file audio trong metadata phải tồn tại.

Đặc trưng được ghép với metadata **theo `clip_id`**, không theo thứ tự dòng. Thiếu đặc trưng cho clip nào là báo lỗi.

## 3. Cài đặt

```bash
pip install -r requirements.txt
```

Thư viện: `numpy`, `scipy`, `librosa`, `scikit-learn`, `xgboost`, `joblib`, `pandas`, `matplotlib`, `seaborn`.
Phiên bản scikit-learn dùng khi huấn luyện được ghi trong `models/model_config.json`; nên dùng cùng phiên bản khi nạp mô hình.

## 4. Cách chạy (từ thư mục gốc của repo)

```bash
# 1. Gộp và kiểm tra metadata (tạo data/metadata_combined.csv)
python src/prepare_metadata.py --audio-dir data/fake --max-real-per-speaker 50

# 2. Trích đặc trưng -> features/mfcc_features.npz
python src/features_handcrafted.py

# 3. Huấn luyện và chọn siêu tham số -> models/, results/
python src/train_classical.py

# 4. Đánh giá trên test_seen -> results/
python src/evaluate.py
```

Ghi chú:

- `--audio-dir`: thư mục chứa audio giả khi đường dẫn trong `metadata_fake.csv` là đường dẫn trên máy khác (ví dụ `C:\...`). Code tìm theo tên file và báo lỗi nếu có tên trùng giữa các thư mục.
- `--max-real-per-speaker N`: lấy tối đa N clip real mỗi người nói (seed cố định = 42) để giảm mất cân bằng lớp và thời gian trích đặc trưng. Không thay đổi nhãn, người nói hay split.
- `evaluate.py --split test_unseen` cho phép đánh giá trên split khác.

## 5. Chi tiết kỹ thuật

### 5.1. Đặc trưng (`src/features_handcrafted.py`)

- Âm thanh được nạp ở 16 kHz, mono; `n_fft = 2048`, `hop = 512`.
- **MFCC**: 13 hệ số (`librosa`). **LFCC**: filterbank tam giác cách đều theo thang Hz, log năng lượng, DCT, giữ 13 hệ số đầu.
- Mỗi loại lấy `mean` và `std` theo thời gian, tổng cộng **52 chiều**:
  `mfcc_mean_0..12 | mfcc_std_0..12 | lfcc_mean_0..12 | lfcc_std_0..12`.
- Tên 52 chiều được lưu trong file `.npz` (`feature_names`). Clip quá ngắn hoặc đặc trưng không hữu hạn sẽ gây lỗi rõ ràng.
- Hàm `extract_vector(audio_path)` dùng chung cho huấn luyện và suy luận.

### 5.2. Huấn luyện (`src/train_classical.py`)

- Chỉ dùng clip có `split == train`. Nếu train chỉ có một lớp hoặc dưới 2 người nói, chương trình dừng và báo lỗi.
- **GroupKFold** (tối đa 5 fold) theo `speaker_id`; điểm chọn mô hình là ROC-AUC trung bình. Fold chỉ có một lớp bị bỏ qua và số fold thực dùng được ghi lại.
- Lưới siêu tham số:
  - Logistic Regression: `C ∈ {0.1, 1, 10}` (có `StandardScaler`)
  - SVM RBF: `C ∈ {0.1, 1, 10}`, `gamma ∈ {scale, 0.01}` (có `StandardScaler`)
  - Random Forest: `n_estimators ∈ {100, 300}`, `max_depth ∈ {None, 8}`
  - XGBoost: `n_estimators ∈ {100, 300}`, `max_depth ∈ {3, 5}`
- Cấu hình tốt nhất được **huấn luyện lại trên toàn bộ tập train** rồi lưu. `StandardScaler` nằm trong Pipeline nên **không cần file scaler riêng**.
- Trọng số đặc trưng: hệ số (Logistic Regression), `feature_importances_` (Random Forest, XGBoost) hoặc permutation importance (SVM; dùng tập `dev` nếu hợp lệ, nếu không dùng train và ghi rõ trong cột `method`).

### 5.3. Đánh giá (`src/evaluate.py`)

- Dùng **nhãn gốc** trong metadata, ngưỡng phân loại 0,5 cho F1.
- **EER**: điểm trên đường ROC mà FPR ≈ FNR (lấy trung bình hai giá trị tại điểm gần nhất); ngưỡng tương ứng được lưu.
- Nếu split đánh giá chỉ có một lớp, chương trình dừng vì AUC/EER không xác định.

## 6. Các tệp đầu ra

| Tệp | Mô tả |
|---|---|
| `features/mfcc_features.npz` | `clip_id`, ma trận `X` (N × 52), `feature_names` |
| `models/best_model.pkl` | Pipeline/mô hình tốt nhất đã huấn luyện lại trên toàn bộ train |
| `models/model_config.json` | Thuật toán, tham số, CV AUC, tên đặc trưng, phiên bản scikit-learn |
| `results/cv_results.csv` | Toàn bộ cấu hình đã thử và ROC-AUC GroupKFold (mean, std, số fold) |
| `results/cv_best_per_model.csv` | Cấu hình tốt nhất của từng mô hình (để so sánh 4 mô hình) |
| `results/feature_importance_weights.csv` | Trọng số/mức quan trọng của 52 đặc trưng, kèm cột `type`, `stat`, `method` |
| `results/evaluation_metrics_summary.csv` | ROC-AUC, EER, ngưỡng EER, F1, số mẫu, số người nói |
| `results/test_seen_predictions.csv` | Xác suất fake và nhãn dự đoán cho từng clip |
| `results/confusion_matrix.png` | Ma trận nhầm lẫn |
| `results/roc_curve.png` | Đường cong ROC |

## 7. Kết quả thực nghiệm

**Chưa có kết quả hợp lệ.** Kết quả sẽ được điền sau khi chạy pipeline trên `metadata_combined.csv` với:

- giọng giả có `speaker_id` là người bị clone và có `split` tương ứng;
- tập `test_seen` có đủ real và fake.

| Chỉ số (test_seen) | Giá trị |
|---|---|
| Số mẫu (real / fake) | _chưa có_ |
| Số người nói | _chưa có_ |
| ROC-AUC | _chưa có_ |
| EER | _chưa có_ |
| F1 (ngưỡng 0,5) | _chưa có_ |

Khi cập nhật, ghi kèm: nguồn dữ liệu, số clip và số người nói của từng split, mô hình được chọn và CV AUC.
Nếu EER gần 0, cần kiểm tra rò rỉ dữ liệu (cùng người nói/cùng câu ở train và test) trước khi kết luận.

## 8. Tích hợp vào app demo (Task 5)

```python
from src.inference_handcrafted import HandcraftedPredictor

predictor = HandcraftedPredictor(model_path="models/best_model.pkl")
result = predictor.predict_audio("path/to/audio.wav")

if result["status"] == "success":
    print(result["prediction"])   # "real" hoặc "fake"
    print(result["prob_fake"])    # xác suất giọng giả, từ 0 đến 1
else:
    print(result["message"])      # mô tả lỗi
```

- `predict_audio` trích đặc trưng 52 chiều từ file, rồi gọi Pipeline đã chứa `StandardScaler`; **không cần `scaler.pkl`** và không scale hai lần.
- Lỗi (file hỏng, quá ngắn,...) được trả về trong `status = "error"` thay vì làm sập app.

## 9. Hạn chế và lưu ý

- Mô hình chỉ dùng đặc trưng thủ công mức clip, không xét thông tin theo thời gian; hiệu năng trên giọng giả sinh bởi công cụ chưa gặp (`test_unseen`) có thể thấp hơn nhiều so với `test_seen`.
- Mất cân bằng lớp (real nhiều hơn fake rất nhiều) được giảm bằng `--max-real-per-speaker`; cần ghi rõ giá trị đã dùng khi báo cáo.
- Nếu dùng bộ dữ liệu tạm khi dữ liệu chính thức chưa sẵn sàng, phải ghi rõ nguồn, số clip, số người nói và gọi đó là kết quả tạm.
- Không tự gán nhãn, `speaker_id` hay `split` khi metadata thiếu: code sẽ dừng và báo lỗi để người dùng sửa dữ liệu gốc.

## 10. Quy trình Git

```bash
git add src requirements.txt README.md .gitignore
git commit -m "Mô tả rõ thay đổi"
git push

# Sau khi chạy xong pipeline:
git add data/metadata_combined.csv features results models
git commit -m "Add Task 3 results"
git push
```

Commit và đẩy kết quả ngay sau khi chạy để tránh mất bằng chứng thực nghiệm.
