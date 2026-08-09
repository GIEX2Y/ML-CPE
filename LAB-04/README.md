# LAB 04 - K-Nearest Neighbors (KNN) Classification

## Machine Learning (04-624-201)

### Student Information

* **Name:** มานัติพันธิ์ เมืองนาค
* **Student ID:** 116710462013-9
* **Course:** Machine Learning (04-624-201)
* **Faculty:** Faculty of Engineering
* **Department:** Computer Engineering
* **University:** Rajamangala University of Technology Thanyaburi

---

# Project Overview

This laboratory assignment demonstrates the implementation of the **K-Nearest Neighbors (KNN)** algorithm for classification using the **Student Performance Dataset**. The project explores data preprocessing, feature scaling, model training, and performance evaluation using different values of **K**.

The objective is to determine the optimal number of neighbors by comparing classification accuracy and evaluating the model's performance.

---

# Objectives

* Load and explore the dataset.
* Perform data preprocessing.
* Standardize numerical features.
* Train KNN models with different values of **K (3, 5, and 7)**.
* Evaluate model performance using Accuracy.
* Compare the performance of different K values.
* Identify the best-performing model.

---

# Dataset

**Dataset:** Student Performance Dataset

The dataset contains student academic information used to classify student grades.

## Features

| Feature                 | Description               |
| ----------------------- | ------------------------- |
| student_id              | Student identifier        |
| weekly_self_study_hours | Weekly self-study hours   |
| attendance_percentage   | Attendance percentage     |
| class_participation     | Class participation score |
| total_score             | Overall academic score    |
| grade                   | Target class              |

Target Variable:

* **grade**

---

# Project Structure

```text
LAB-04/
│
├── LAB-04.py
├── student_performance.csv
├── README.md
---

# Libraries Used

* pandas
* matplotlib
* scikit-learn

Install dependencies:

```bash
pip install pandas matplotlib scikit-learn
```

or

```bash
pip install -r requirements.txt
```

---

# Machine Learning Workflow

1. Load the dataset
2. Explore the dataset
3. Handle missing values
4. Select features and target
5. Split training and testing data
6. Standardize the features using StandardScaler
7. Train KNN classifiers with:

   * K = 3
   * K = 5
   * K = 7
8. Evaluate Accuracy
9. Generate Confusion Matrix
10. Compare model performance

---

# Model Evaluation

The following evaluation metrics are used:

* Accuracy
* Classification Report
* Confusion Matrix

The model with the highest test accuracy is selected as the best classifier.

---

# Output

The program produces:

* Dataset overview
* Missing value analysis
* Accuracy for each K value
* Classification Report
* Confusion Matrix
* Accuracy Comparison Graph
* Best K value

---

# Experimental Results

Example output:

```text
K = 3
Accuracy = 0.9875

K = 5
Accuracy = 0.9910

K = 7
Accuracy = 0.9894

Best K = 5
Best Accuracy = 0.9910
```

*(Actual results may vary depending on the dataset split and random sampling.)*

---

# Conclusion

The K-Nearest Neighbors algorithm successfully classified student performance using the selected features. Feature standardization significantly improved classification performance. By comparing different values of K, the experiment identified the optimal number of neighbors that achieved the highest classification accuracy.

This laboratory demonstrates the effectiveness of KNN for supervised classification problems and highlights the importance of selecting an appropriate K value.

---

# References

* Scikit-learn Documentation: https://scikit-learn.org/stable/
* Pandas Documentation: https://pandas.pydata.org/
* Matplotlib Documentation: https://matplotlib.org/
* Student Performance Dataset (Kaggle)
