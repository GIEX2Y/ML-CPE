import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# โหลด Dataset
df = pd.read_csv("LAB-02/apple_products_pricing.csv")

# แสดงข้อมูล 5 แถวแรก
print(df.head())

# Shape
print("\nShape:")
print(df.shape)

# Data Types
print("\nData Types:")
print(df.dtypes)

# Summary Statistics
print("\nSummary Statistics:")
print(df.describe())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Class Distribution
print("\nProduct Category:")
print(df["Product_Category"].value_counts())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["Current_Price_USD"], bins=30)
plt.title("Current Price Distribution")
plt.xlabel("Current Price (USD)")
plt.ylabel("Frequency")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Data Cleaning
df = df.drop_duplicates()

# เติม Missing Value ของข้อมูลตัวเลขด้วยค่าเฉลี่ย
df = df.fillna(df.select_dtypes(include="number").mean())

# แปลงชนิดข้อมูล Date
df["Date"] = pd.to_datetime(df["Date"])

# เปรียบเทียบ Mean และ Median
print("\nMean:", df["Current_Price_USD"].mean())
print("Median:", df["Current_Price_USD"].median())

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Stock_Status_Label"] = le.fit_transform(df["Stock_Status"])

# One-Hot Encoding
df = pd.get_dummies(df, columns=["Product_Category"])

# บันทึกข้อมูล
df.to_csv("LAB-02/apple_products_clean.csv", index=False)

print("\nData Preprocessing Complete!")