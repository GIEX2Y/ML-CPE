# LAB 2 : Data Preprocessing

## Course
**Machine Learning (04-624-201)**

Faculty of Engineering  
Rajamangala University of Technology Thanyaburi

---

## Objective

The objective of this lab is to learn the data preprocessing process before applying Machine Learning models. This includes:

- Dataset Exploration
- Data Visualization
- Data Cleaning
- Feature Engineering

---

## Dataset

Dataset: **Apple Products Pricing Dataset (2020–2026)**

Contains information about Apple products sold on different platforms, including:

- Product Category
- Model Name
- Launch Price
- Current Price
- Discount
- Rating
- Reviews
- Stock Status
- Sale Event

Total Records: **80,000**

Total Features: **14**

---

## Libraries Used

```python
pandas
matplotlib
seaborn
scikit-learn
```

Install

```bash
pip install pandas matplotlib seaborn scikit-learn
```

---

## Project Structure

```
LAB-02/
│── LAB2.py
│── apple_products_pricing.csv
│── apple_products_clean.csv
│── README.md
```

---

## Tasks

### LAB 1 : Dataset Exploration

- Load Dataset
- Display Shape
- Display Data Types
- Summary Statistics
- Missing Values
- Duplicate Records
- Class Distribution

---

### LAB 2 : Data Visualization

- Histogram
- Correlation Heatmap

---

### Part 3 : Data Cleaning

- Handle Missing Values
- Remove Duplicate Records
- Correct Incorrect Data
- Convert Data Types
- Compare Mean and Median

---

### Part 4 : Feature Engineering

- Label Encoding
- One-Hot Encoding

---

## Results

### Dataset Information

| Item | Value |
|------|------:|
| Rows | 80,000 |
| Columns | 14 |
| Missing Values | Sale_Event (73,351) |
| Duplicate Rows | 0 |

---

## Output

After preprocessing, the cleaned dataset is saved as:

```
apple_products_clean.csv
```

---

## Author

**Name : MANATTIPAN MUANGNAK**

**Student ID : 1167104620139**

Machine Learning Laboratory