# 🤖 LAB 03 – Regression & Classification

> **Machine Learning (04-624-201)**  
> Faculty of Engineering  
> Rajamangala University of Technology Thanyaburi

---

## 👨‍🎓 Author

**Name:** มานัติพันธิ์ เมืองนาค 
**Student ID:** 116710462013-9

---

# 📖 Project Overview

This project was developed as part of the **Machine Learning** course to study and compare several **Supervised Learning** techniques.

The implemented algorithms include:

- 📈 Simple Linear Regression
- 📊 Multiple Linear Regression
- 🤖 Logistic Regression
- 🧠 Principal Component Analysis (PCA)

The project uses the **UTKFace Dataset** to perform:

- 🎂 **Age Prediction**
- 👤 **Gender Classification**

---

# 📂 Dataset

### UTKFace Dataset (Age, Gender and Ethnicity Face Data)

🔗 https://www.kaggle.com/datasets/jangedoo/utkface-new

### Dataset Features

| Feature | Description |
|---------|-------------|
| Age | Age of the person |
| Gender | Male / Female |
| Ethnicity | Ethnicity category |
| Image | Face image stored as pixel values |

**Total Images:** **23,708**

---

# 🛠 Libraries

This project uses the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn

Install them with:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# ⚙️ Workflow

## 1️⃣ Data Preprocessing

✔ Load CSV Dataset

✔ Convert pixel strings into NumPy arrays

✔ Standardize data using **StandardScaler**

✔ Reduce dimensionality using **PCA**

```
7500 Features
        │
        ▼
100 Principal Components
```

---

## 2️⃣ Regression

Models:

- Simple Linear Regression
- Multiple Linear Regression

Target:

🎂 **Age Prediction**

Evaluation Metrics

- MAE
- R² Score

---

## 3️⃣ Classification

Model:

- Logistic Regression

Target:

👤 **Gender Classification**

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 4️⃣ Decision Boundary

Visualize the classification boundary using

**PCA (2 Components)**

---

# 📊 Sample Results

## Regression

| Model | MAE | R² Score |
|------|----:|---------:|
| Simple Linear Regression | **15.39** | **0.023** |
| Multiple Linear Regression | **11.51** | **0.439** |

---

## Classification

| Metric | Score |
|--------|------:|
| Accuracy | **84.80%** |
| Precision | **0.85** |
| Recall | **0.85** |
| F1-score | **0.85** |

---

# 📁 Project Structure

```text
LAB-03/
│
├── phyton.py
├── README.md
└── .gitignore
```

---

# 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/USERNAME/LAB-03.git
```

### Enter Project Directory

```bash
cd LAB-03
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Run the Program

```bash
python phyton.py
```

---

# 🎯 Learning Outcomes

After completing this laboratory, the following concepts were learned:

- ✅ Linear Regression
- ✅ Multiple Linear Regression
- ✅ Logistic Regression
- ✅ Principal Component Analysis (PCA)
- ✅ Regression & Classification Comparison
- ✅ Machine Learning Model Evaluation

---

# 📚 References

- **UTKFace Dataset**  
  https://www.kaggle.com/datasets/jangedoo/utkface-new

- **Scikit-learn Documentation**  
  https://scikit-learn.org/stable/

- **Pandas Documentation**  
  https://pandas.pydata.org/

- **NumPy Documentation**  
  https://numpy.org/

- **Matplotlib Documentation**  
  https://matplotlib.org/

---

# ⚠️ Important Note

> **The `utkface_sortAge.csv` dataset is not included in this repository because of its large file size and GitHub upload limitations.**
>
> Please download the **UTKFace Dataset** from Kaggle and place the `utkface_sortAge.csv` file in the project directory before running the program.

---

## ⭐ Machine Learning Laboratory

**Faculty of Engineering**  
**Rajamangala University of Technology Thanyaburi**
