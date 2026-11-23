
# ============================================================
# PROJECT 2: DATA CLASSIFICATION USING ARTIFICIAL INTELLIGENCE
# Dataset: Iris Flower Dataset
# Algorithm: K-Nearest Neighbors (KNN)
# ============================================================

# 1. Import required libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# 2. Create output directory

os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("       IRIS FLOWER CLASSIFICATION USING KNN")
print("=" * 60)


# 3. Load the Iris Dataset

iris = load_iris()

X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = [target_names[i] for i in y]

print("\n[1] DATASET INFORMATION")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Description:")
print(df.describe())

print("\nClass Distribution:")
print(df["species"].value_counts())


# 4. Dataset Visualization

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="species",
    hue="species",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Iris Flower Species")
plt.xlabel("Flower Species")
plt.ylabel("Number of Samples")
plt.tight_layout()

plt.savefig("outputs/class_distribution.png")
plt.show()


# 5. Separate Features and Target

X = df[feature_names]
y = iris.target

print("\n[2] FEATURES AND TARGET")

print("Features:", feature_names)
print("Target Classes:", target_names)


# 6. Split Dataset into Training and Testing Sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n[3] TRAIN-TEST SPLIT")

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# 7. Feature Scaling using StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n[4] FEATURE SCALING COMPLETED")

print("StandardScaler applied successfully.")


# 8. Build KNN Classification Model

k = 5

model = KNeighborsClassifier(n_neighbors=k)

print("\n[5] MODEL TRAINING")

model.fit(X_train_scaled, y_train)

print("KNN Model trained successfully!")
print("Number of Neighbors (K):", k)


# 9. Make Predictions

y_pred = model.predict(X_test_scaled)

print("\n[6] MODEL PREDICTIONS")

print("Actual Values:    ", y_test)
print("Predicted Values: ", y_pred)


# 10. Evaluate Model Performance

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

print("\n[7] MODEL PERFORMANCE")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")


# 11. Classification Report

print("\n[8] CLASSIFICATION REPORT")

report = classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    zero_division=0
)

print(report)


# 12. Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()

plt.savefig("outputs/confusion_matrix.png")
plt.show()


# 13. Save Performance Metrics

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

metrics.to_csv("outputs/model_metrics.csv", index=False)

print("\n[9] METRICS SAVED")
print(metrics)


# 14. Test Model with a New Flower Sample

print("\n[10] NEW FLOWER PREDICTION")

# [Sepal Length, Sepal Width, Petal Length, Petal Width]
new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

predicted_species = target_names[prediction[0]]

print("Input Measurements:", new_flower[0])
print("Predicted Flower Species:", predicted_species)


# 15. Final Summary

print("\n" + "=" * 60)
print("                 PROJECT COMPLETED")
print("=" * 60)

print(f"Algorithm Used: K-Nearest Neighbors (K={k})")
print(f"Dataset Samples: {len(df)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Final Accuracy: {accuracy * 100:.2f}%")
print(f"Final F1 Score: {f1:.4f}")

print("\nGenerated Files:")
print("- outputs/class_distribution.png")
print("- outputs/confusion_matrix.png")
print("- outputs/model_metrics.csv")

print("\nExecution completed successfully!")
