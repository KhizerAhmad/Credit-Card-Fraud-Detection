import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv("creditcard.csv")
print(f"Dataset Shape: {dataset.shape}")
pd.set_option("display.max_columns", 35)
print(dataset.head(5))

print("=== DATASET BEFORE PREPROCESSING ===")
print(dataset.head(10))
print(f"Shape: {dataset.shape}")
print(f"Missing Values:\n{dataset.isnull().sum()}")
print(f"Duplicates: {dataset.duplicated().sum()}")
dataset = dataset.drop_duplicates()
print(f"Shape after removing duplicates: {dataset.shape}")

dataset = dataset.fillna(dataset.mean())
print("Missing values after handling:", dataset.isnull().sum().sum())

print("No categorical columns to encode in this dataset.")

print("=== SUMMARY STATISTICS ===")
print(dataset.describe())

labels = dataset["Class"].values
features = dataset.drop("Class", axis=1).values
print(f"Total Length: {len(labels)}")
print(f"Total Features: {features.shape[1]}")

missing_values = dataset.isnull().sum().sum()
print(f"Missing values in dataset: {missing_values}")

unique, counts = np.unique(labels, return_counts=True)
print("Class Distribution:")
for value, count in zip(unique, counts):
    if value == 0:
        print(f"Normal Transactions: {count}")
    else:
        print(f"Fraud Transactions: {count}")

scaler = StandardScaler()
features = scaler.fit_transform(features)
print(f"Features scaled to standard normal variance distributions.")

print("=== DATASET AFTER PREPROCESSING ===")
print(pd.DataFrame(features).head(10))
print(f"Shape: {features.shape}")
print(f"Missing Values: {pd.DataFrame(features).isnull().sum().sum()}")
print(f"Duplicates: {pd.DataFrame(features).duplicated().sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)

print(f"Total Length of train data of labels: {len(y_train)}")
print(f"Total Length of train data of features: {len(X_train)}")
print(f"Total Length of test data of labels: {len(y_test)}")
print(f"Total Length of test data of features: {len(X_test)}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predict_knn = knn.predict(X_test)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
predict_lr = lr.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predict_dt = dt.predict(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
predict_rf = rf.predict(X_test)

print("KNN Results:")
print("Accuracy:", accuracy_score(y_test, predict_knn))
print("Precision:", precision_score(y_test, predict_knn))
print("Recall:", recall_score(y_test, predict_knn))
print("F1-Score:", f1_score(y_test, predict_knn))

print("Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, predict_lr))
print("Precision:", precision_score(y_test, predict_lr))
print("Recall:", recall_score(y_test, predict_lr))
print("F1-Score:", f1_score(y_test, predict_lr))

print("Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, predict_dt))
print("Precision:", precision_score(y_test, predict_dt))
print("Recall:", recall_score(y_test, predict_dt))
print("F1-Score:", f1_score(y_test, predict_dt))

print("Random Forest Results:")
print("Accuracy:", accuracy_score(y_test, predict_rf))
print("Precision:", precision_score(y_test, predict_rf))
print("Recall:", recall_score(y_test, predict_rf))
print("F1-Score:", f1_score(y_test, predict_rf))

cm_knn = confusion_matrix(y_test, predict_knn)
cm_lr = confusion_matrix(y_test, predict_lr)
cm_dt = confusion_matrix(y_test, predict_dt)
cm_rf = confusion_matrix(y_test, predict_rf)

plt.figure(figsize=(8, 5))
plt.bar(["Normal", "Fraud"], counts)
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.yscale("log")
plt.show()

plt.figure(figsize=(8, 5))
correlation = dataset.corr()
plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()
plt.title("Correlation Heatmap")
plt.show()

algorithms = ["KNN", "Logistic Regression", "Decision Tree", "Random Forest"]
accuracies = [
    accuracy_score(y_test, predict_knn),
    accuracy_score(y_test, predict_lr),
    accuracy_score(y_test, predict_dt),
    accuracy_score(y_test, predict_rf),
]

plt.figure(figsize=(8, 5))
plt.bar(algorithms, accuracies)
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_knn, cmap="Blues")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm_knn[i, j],
            ha="center",
            va="center",
            color="white" if cm_knn[i, j] > cm_knn.max() / 2 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_lr, cmap="Greens")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm_lr[i, j],
            ha="center",
            va="center",
            color="white" if cm_lr[i, j] > cm_lr.max() / 2 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_dt, cmap="Oranges")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm_dt[i, j],
            ha="center",
            va="center",
            color="white" if cm_dt[i, j] > cm_dt.max() / 2 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(cm_rf, cmap="Purples")
plt.colorbar()
for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm_rf[i, j],
            ha="center",
            va="center",
            color="white" if cm_rf[i, j] > cm_rf.max() / 2 else "black",
        )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest Confusion Matrix")
plt.show()