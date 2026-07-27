import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# -------------------------------
# 1. โหลดข้อมูล
# -------------------------------
df = pd.read_csv("utkface_sortAge.csv")

print(df.head())
print(df.info())

# -------------------------------
# 2. แปลงข้อมูลภาพ
# -------------------------------
X = np.array(df["image"].apply(lambda x: np.fromstring(x, sep=' ')).tolist())

print("Shape :", X.shape)

# -------------------------------
# 3. Scaling + PCA
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)

print("After PCA :", X_pca.shape)

# =====================================================
# LAB 1 : Regression (Age Prediction)
# =====================================================

print("\n========== Regression ==========\n")

y_age = df["age"]

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y_age,
    test_size=0.2,
    random_state=42
)

# --- Simple Regression ---
reg_simple = LinearRegression()
reg_simple.fit(X_train[:, :1], y_train)
pred_simple = reg_simple.predict(X_test[:, :1])

mae_simple = mean_absolute_error(y_test, pred_simple)
r2_simple = r2_score(y_test, pred_simple)

print("Simple Regression")
print("MAE =", mae_simple)
print("R² =", r2_simple)

# --- Multiple Regression ---
reg_multi = LinearRegression()
reg_multi.fit(X_train, y_train)
pred_multi = reg_multi.predict(X_test)

mae_multi = mean_absolute_error(y_test, pred_multi)
r2_multi = r2_score(y_test, pred_multi)

print("\nMultiple Regression")
print("MAE =", mae_multi)
print("R² =", r2_multi)

# --- Training vs Testing ---
train_pred = reg_multi.predict(X_train)
mae_train = mean_absolute_error(y_train, train_pred)
mae_test = mean_absolute_error(y_test, pred_multi)

print("\nTraining MAE =", mae_train)
print("Testing MAE =", mae_test)

# =====================================================
# LAB 2 : Classification (Gender Prediction)
# =====================================================

print("\n========== Classification ==========\n")

y_gender = df["gender"]

X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y_gender,
    test_size=0.2,
    random_state=42
)

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

pred_gender = clf.predict(X_test)

acc = accuracy_score(y_test, pred_gender)

print("Accuracy =", acc)

print("\nClassification Report\n")
print(classification_report(y_test, pred_gender))

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, pred_gender)

print("\nConfusion Matrix\n")
print(cm)

ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -------------------------------
# ROC Curve & AUC
# -------------------------------
y_prob = clf.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# ***** เพิ่มบรรทัดนี้ *****
print("\nAUC =", roc_auc)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# Decision Boundary Visualization
# -------------------------------
X_vis = X_pca[:, :2]

X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
    X_vis,
    y_gender,
    test_size=0.2,
    random_state=42
)

clf_vis = LogisticRegression(max_iter=1000, class_weight="balanced")
clf_vis.fit(X_train_vis, y_train_vis)

x_min, x_max = X_train_vis[:,0].min()-1, X_train_vis[:,0].max()+1
y_min, y_max = X_train_vis[:,1].min()-1, X_train_vis[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = clf_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(
    X_train_vis[:,0],
    X_train_vis[:,1],
    c=y_train_vis,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=20
)

plt.title("Decision Boundary (Gender Classification)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()