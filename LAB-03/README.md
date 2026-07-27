LAB 03 - Regression & Classification
Course
Machine Learning (04-624-201)
Faculty of Engineering
Rajamangala University of Technology Thanyaburi
Author
Name: Manattipan Muangnak
Student ID: 116710462013-9
Project Overview
This project is part of the Machine Learning course and aims to study and compare several Supervised Learning techniques, including:
Simple Linear Regression
Multiple Linear Regression
Logistic Regression
Principal Component Analysis (PCA)
The project uses the UTKFace Dataset to perform the following tasks:
Age Prediction
Gender Classification
Dataset
Dataset: UTKFace Dataset (Age, Gender, and Ethnicity Face Data)
Source: Kaggle
https://www.kaggle.com/datasets/jangedoo/utkface-new
The project uses the following dataset file:
utkface_sortAge.csv
The dataset contains the following features:
Age
Gender
Ethnicity
Image (Pixel Values)
Total Images: 23,708
Libraries Used
pandas
numpy
matplotlib
scikit-learn
Install the required libraries using:
pip install pandas numpy matplotlib scikit-learn
Workflow
1. Data Preprocessing
Load the dataset from CSV
Convert pixel strings into numerical arrays
Apply StandardScaler
Reduce dimensionality using PCA
The number of features is reduced from:
7,500 Features
to
100 Principal Components
2. Regression
Apply the following regression models:
Simple Linear Regression
Multiple Linear Regression
Target:
Age Prediction
Evaluation Metrics:
Mean Absolute Error (MAE)
R² Score
3. Classification
Apply:
Logistic Regression
Target:
Gender Classification
Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
ROC Curve
Area Under the Curve (AUC)
4. Decision Boundary Visualization
Use the first 2 PCA components to visualize the decision boundary of the Logistic Regression classifier.
Sample Results
Regression
Simple Linear Regression
Metric	Value
MAE	15.39
R² Score	0.023
Multiple Linear Regression
Metric	Value
MAE	11.51
R² Score	0.439
Classification
Metric	Value
Accuracy	84.80%
Precision	0.85
Recall	0.85
F1-score	0.85
Project Structure
LAB-03/
│
├── phyton.py
├── README.md
└── .gitignore
Note: The dataset file (utkface_sortAge.csv) is not included in this repository.
How to Run
1. Clone the repository
git clone https://github.com/USERNAME/LAB-03.git
2. Navigate to the project folder
cd LAB-03
3. Install the required libraries
pip install pandas numpy matplotlib scikit-learn
4. Run the program
python phyton.py
Learning Outcomes
After completing this project, the following concepts were learned:
Understanding Linear Regression
Understanding Multiple Linear Regression
Understanding Logistic Regression
Applying Principal Component Analysis (PCA)
Comparing Regression and Classification models
Evaluating Machine Learning models using standard performance metrics
References
UTKFace Dataset (Age, Gender, and Ethnicity Face Data)
https://www.kaggle.com/datasets/jangedoo/utkface-new
Scikit-learn Documentation
https://scikit-learn.org/stable/
Pandas Documentation
https://pandas.pydata.org/
NumPy Documentation
https://numpy.org/
Matplotlib Documentation
https://matplotlib.org/
Note
The utkface_sortAge.csv file is not included in this repository due to its large file size and GitHub upload limitations. Please download the UTKFace Dataset from Kaggle and place the utkface_sortAge.csv file in the project directory before running the program.
