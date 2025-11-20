# MABe Mouse Behavior Detection – Project README

> **Kaggle Competition**: MABe Mouse Behavior Detection

Dự án này triển khai một pipeline hoàn chỉnh nhằm phát hiện hành vi của chuột dựa trên dữ liệu tracking từ video. Pipeline bao gồm việc load dữ liệu, tạo đặc trưng (feature engineering), inference bằng XGBoost, hậu xử lý, và xuất submission.

Toàn bộ dự án được tổ chức theo dạng module hóa nhằm dễ bảo trì và chạy cả trên **Kaggle** lẫn **local machine**.

---

## 1. Cấu trúc thư mục dự án

```
project/
│
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_self.py
│   ├── feature_pair.py
│   ├── preprocess.py
│   ├── infer.py
│   ├── postprocess.py
│   └── run_pipeline.py
│
├── external/
│   ├── self_features.py
│   ├── pair_features.py
│   └── robustify.py
│
├── models/
│   └── results/ (copy từ Kaggle starter kit)
│
└── working/
    ├── self_features/
    ├── pair_features/
    └── submission.csv
```

---

## 2. Yêu cầu môi trường

* Python **3.10+** (khuyến nghị)
* pip, venv
* Cài thư viện trong `requirements.txt`
* Dữ liệu từ Kaggle competition

---

## 3. Cài đặt

### Cài cho Kaggle Notebook

```
pip install -r requirements.txt --no-index --find-links=/kaggle/input/mabe-package
```

### Cài trên local (MacOS / Linux / Windows)

```
pip install -r requirements.txt
```

*(file này bạn có thể tạo để tránh lỗi polars/xgboost do version Python)*

---

## 4. Mô tả tác dụng từng file trong thư mục `src/`

### 📌 `config.py`

Chứa hằng số toàn cục:

* đường dẫn thư mục input
* danh sách behavior (self, pair)
* danh sách body parts
* danh sách cột index

→ Mục đích: tất cả file khác đều import thông tin từ đây, giúp tránh trùng logic.

---

### 📌 `data_loader.py`

Tác dụng:

* Đọc file `test.csv` và annotation
* Parse dữ liệu thành dạng phân nhóm theo (lab, video, agent, target)
* Load tracking `.parquet`

→ Đây là bước **tiền xử lý input** của pipeline.

---

### 📌 `feature_self.py`

Tác dụng:

* Gọi hàm `make_self_features()` từ `external/self_features.py`
* Sinh đặc trưng cho các hành vi tự thân (SELF)
  → Ví dụ: run, freeze, selfgroom.

### 📌 `feature_pair.py`

Tác dụng:

* Gọi hàm `make_pair_features()` từ `external/pair_features.py`
* Sinh đặc trưng giữa 2 con chuột (PAIR)
  → Ví dụ: chase, attack, sniff.

→ Cả hai file chịu trách nhiệm **feature engineering**.

---

### 📌 `preprocess.py`

Tác dụng:

* Chuẩn hóa dữ liệu trước khi đưa vào mô hình
* Tách feature / index
* Gom nhóm theo behavior

→ Giúp mô hình nhận đúng input và đúng định dạng.

---

### 📌 `infer.py`

Tác dụng:

* Load toàn bộ model XGBoost theo từng behavior
* Predict theo từng fold
* Áp dụng threshold → ra nhãn hành vi
* Kết hợp kết quả nhiều fold

→ Đây là nơi thực hiện **inference logic**.

---

### 📌 `postprocess.py`

Tác dụng:

* Merge các frame liên tiếp thành segment hành vi
* Loại bỏ nhiễu
* Gọi hàm `robustify()` từ starter kit
* Chuẩn hóa theo format submission yêu cầu

→ Đây là bước **làm sạch kết quả và chuẩn hóa**.

---

### 📌 `run_pipeline.py`

Tác dụng:

* Chạy toàn bộ pipeline theo đúng thứ tự:
  **load → feature → preprocess → infer → postprocess → export**
* Tự động tạo các thư mục cần thiết
* Xuất file `submission.csv`

→ Đây là file chính để chạy toàn bộ hệ thống.

---

## 5. Luồng chạy tổng quan (Pipeline Flow)

```
1. data_loader.py      → Load dữ liệu + phân nhóm video / agent / target
2. feature_self/pair   → Sinh feature cho từng nhóm
3. preprocess.py       → Chuẩn hóa dữ liệu
4. infer.py            → Dự đoán bằng XGBoost + threshold
5. postprocess.py      → Gộp đoạn, dọn nhiễu, robustify
6. run_pipeline.py     → Xuất submission.csv
```

Sơ đồ đơn giản:

```
Raw Data → Feature Extraction → Preprocess → Model Inference → Postprocess → Submission
```

---

## 6. Chạy pipeline

```
python src/run_pipeline.py
```

File kết quả xuất ra:

```
working/submission.csv
```

---

## 7. Cải tiến tương lai

### 🔧 Feature Engineering

* Thêm tốc độ, gia tốc
* Góc quay cơ thể / orientation
* Smoothing bằng rolling window

### 🔧 Mô hình

* LightGBM hoặc CatBoost
* Ensemble theo multi-time-window

### 🔧 Hậu xử lý

* Loại bỏ segment quá ngắn (< 5 frames)
* Dùng HMM để làm mượt chuỗi nhãn

### 🔧 Tối ưu hiệu năng

* Dùng LazyFrame của Polars triệt để
* Chạy song song theo từng video

---

