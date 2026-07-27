# LAB 03 - Regression & Classification

## รายวิชา
Machine Learning (04-624-201)

## ผู้จัดทำ
- ชื่อ : มานัติพันธิ์ เมืองนาค
- รหัสนักศึกษา : 116710462013-9

---

# รายละเอียดโครงการ

โครงการนี้เป็นส่วนหนึ่งของรายวิชา Machine Learning เพื่อศึกษาและเปรียบเทียบเทคนิคการเรียนรู้แบบ Supervised Learning ได้แก่

- Linear Regression
- Multiple Linear Regression
- Logistic Regression
- Principal Component Analysis (PCA)

โดยใช้ข้อมูลภาพใบหน้าจากชุดข้อมูล **UTKFace Dataset** เพื่อ

- ทำนายอายุ (Age Prediction)
- จำแนกเพศ (Gender Classification)

---

# Dataset

ชุดข้อมูลที่ใช้

**UTKFace Dataset (Age, Gender and Ethnicity Face Data)**

Kaggle

https://www.kaggle.com/datasets/jangedoo/utkface-new

ภายในโปรเจกต์ใช้ไฟล์

```
utkface_sortAge.csv
```

ประกอบด้วยข้อมูล

- age
- gender
- ethnicity
- image (Pixel Values)

จำนวนข้อมูลทั้งหมด

```
23,708 Images
```

---

# Libraries

```python
pandas
numpy
matplotlib
scikit-learn
```

สามารถติดตั้งได้ด้วย

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# ขั้นตอนการทำงาน

## 1. Data Preprocessing

- อ่านข้อมูลจาก CSV
- แปลง Pixel String เป็น Array
- StandardScaler
- PCA ลดมิติข้อมูล

จาก

```
7500 Features
```

เหลือ

```
100 Features
```

---

## 2. Regression

ใช้

- Simple Linear Regression
- Multiple Linear Regression

เพื่อทำนาย

```
Age Prediction
```

ประเมินผลด้วย

- MAE
- R² Score

---

## 3. Classification

ใช้

```
Logistic Regression
```

เพื่อจำแนก

```
Gender Prediction
```

ประเมินผลด้วย

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC

---

## 4. Decision Boundary

ใช้ข้อมูล PCA 2 Components

เพื่อแสดงพื้นที่การจำแนกของ Logistic Regression

---

# ผลลัพธ์ตัวอย่าง

Regression

```
Simple Regression

MAE = 15.39
R² = 0.023
```

```
Multiple Regression

MAE = 11.51
R² = 0.439
```

Classification

```
Accuracy = 84.80%
```

```
Precision = 0.85

Recall = 0.85

F1-score = 0.85
```

---

# โครงสร้างโปรเจกต์

```
LAB-03/
│
├── phyton.py
├── utkface_sortAge.csv
├── README.md
└── .venv/
```

---

# วิธีใช้งาน

Clone Repository

```bash
git clone https://github.com/USERNAME/LAB-03.git
```

เข้าโฟลเดอร์

```bash
cd LAB-03
```

ติดตั้ง Library

```bash
pip install pandas numpy matplotlib scikit-learn
```

รันโปรแกรม

```bash
python phyton.py
```

---

# ผลการเรียนรู้

- เข้าใจหลักการ Regression
- เข้าใจหลักการ Classification
- เรียนรู้การลดมิติข้อมูลด้วย PCA
- เปรียบเทียบประสิทธิภาพของ Regression และ Classification
- วิเคราะห์ผลด้วยตัวชี้วัดทาง Machine Learning

---

# อ้างอิง

1. UTKFace Dataset

https://www.kaggle.com/datasets/jangedoo/utkface-new

2. Scikit-learn Documentation

https://scikit-learn.org/stable/

3. Pandas Documentation

https://pandas.pydata.org/

4. NumPy Documentation

https://numpy.org/

5. Matplotlib Documentation

https://matplotlib.org/
หมายเหตุ: เนื่องจากไฟล์ utkface_sortAge.csv มีขนาดใหญ่และข้อจำกัดของ GitHub หรือพื้นที่จัดเก็บ จึงไม่ได้อัปโหลดไฟล์ดังกล่าวไว้ใน Repository ผู้ใช้งานสามารถดาวน์โหลดชุดข้อมูล UTKFace Dataset จาก Kaggle และเตรียมไฟล์ utkface_sortAge.csv ไว้ในโฟลเดอร์โปรเจกต์ก่อนรันโปรแกรม
