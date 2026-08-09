# ==========================================
# LAB 04 - K-Nearest Neighbor (KNN)
# Machine Learning (04-624-201)
# Student Performance Dataset
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("student_performance.csv")

print("="*50)
print("First 5 Rows")
print(df.head())

print("\nDataset Shape :", df.shape)

print("\nDataset Information")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# 2. Sampling (Reduce training time)
# ==========================================

df = df.sample(n=100000, random_state=42)

print("\nDataset After Sampling")
print(df.shape)

# ==========================================
# 3. Features & Target
# ==========================================

X = df.drop(columns=["student_id", "grade"])

y = df["grade"]

# ==========================================
# 4. Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. Standardization
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# 6. Train KNN Models
# ==========================================

k_values = [3, 5, 7]

accuracy_scores = []

best_model = None
best_accuracy = 0
best_k = 0

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    accuracy_scores.append(acc)

    print("="*50)
    print(f"K = {k}")
    print(f"Accuracy = {acc:.4f}")

    print(classification_report(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k
        best_model = model

# ==========================================
# 7. Best Model
# ==========================================

print("="*50)
print("Best K :", best_k)
print("Best Accuracy :", round(best_accuracy,4))

# ==========================================
# 8. Confusion Matrix
# ==========================================

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot(cmap="Blues")

plt.title(f"Confusion Matrix (K={best_k})")

plt.show()

# ==========================================
# 9. Accuracy Comparison
# ==========================================

plt.figure(figsize=(6,4))

plt.plot(
    k_values,
    accuracy_scores,
    marker="o",
    linewidth=2
)

plt.xticks(k_values)

plt.xlabel("K Value")

plt.ylabel("Accuracy")

plt.title("Accuracy Comparison")

plt.grid(True)

plt.show()

# ==========================================
# 10. Summary
# ==========================================

print("="*50)

for k, acc in zip(k_values, accuracy_scores):
    print(f"K={k} Accuracy={acc:.4f}")

print("\nBest K =", best_k)
print("Best Accuracy =", round(best_accuracy,4))