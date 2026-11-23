# Iris Flower Classification Using Machine Learning

## Project 2 – Artificial Intelligence | DecodeLabs Industrial Training

### 1. Project Overview

This project implements a supervised machine learning classification model to predict the species of an Iris flower based on its physical measurements.

The project uses the **K-Nearest Neighbors (KNN)** algorithm to classify flowers into three categories: Setosa, Versicolor, and Virginica.

It demonstrates the fundamental machine learning workflow, including dataset exploration, preprocessing, feature scaling, model training, prediction, and performance evaluation.

### 2. Objectives

- Understand and explore a real-world dataset.
- Perform data preprocessing and feature scaling.
- Divide the dataset into training and testing sets.
- Implement a K-Nearest Neighbors classification algorithm.
- Evaluate model performance using standard classification metrics.
- Predict the species of new flower samples.

### 3. Dataset Description

The project uses the Iris dataset provided by Scikit-learn.

| Parameter | Description |
|---|---|
| Total Samples | 150 |
| Total Features | 4 |
| Target Classes | 3 |
| Training Data | 80% (120 samples) |
| Testing Data | 20% (30 samples) |

**Input Features:**
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

**Output Classes:**
- Iris Setosa
- Iris Versicolor
- Iris Virginica

### 4. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

### 5. Machine Learning Algorithm

**K-Nearest Neighbors (KNN)**

KNN is a supervised machine learning algorithm that classifies a new data point based on the majority class among its nearest neighbors.

In this project:
- Algorithm: K-Nearest Neighbors
- Number of Neighbors (K): 5
- Distance Metric: Euclidean Distance (default)
- Feature Scaling: StandardScaler

The model learns from the training dataset and predicts the class of unseen flower samples.

### 6. Project Workflow

1. Import the required Python libraries.
2. Load the Iris dataset.
3. Explore dataset features and class distribution.
4. Separate input features and target labels.
5. Split the dataset into training and testing sets.
6. Apply StandardScaler for feature normalization.
7. Train the KNN classification model.
8. Generate predictions using test data.
9. Evaluate performance using accuracy, precision, recall, and F1-score.
10. Generate a confusion matrix and save visualizations.

### 7. Model Evaluation

The model is evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified samples.
- **Precision:** Measures how many positive predictions are correct.
- **Recall:** Measures how many actual class samples are correctly identified.
- **F1-Score:** Represents the harmonic mean of precision and recall.
- **Confusion Matrix:** Displays correct and incorrect classifications across the three species.

All evaluation metrics are calculated using the testing dataset.

### 8. Project Structure

```text
AI_Project_2_Iris_Classification/
│
├── main.py
├── requirements.txt
├── README.md
│
├── outputs/
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   └── model_metrics.csv
│
└── venv/
```

### 9. Installation and Execution

**Step 1: Clone or download the project.**

**Step 2: Install the required dependencies.**

```bash
pip install -r requirements.txt
```

**Step 3: Run the Python program.**

```bash
python main.py
```

The program will train the KNN model, generate predictions, display evaluation metrics, and save the output visualizations inside the `outputs` directory.

### 10. Results and Output

The program generates:

- Dataset distribution visualization.
- Confusion matrix heatmap.
- Classification report.
- Accuracy, precision, recall, and F1-score.
- Prediction for a new Iris flower sample.
- CSV file containing model performance metrics.

Actual performance values are generated during execution.

### 11. Future Improvements

- Compare KNN with Decision Tree, Random Forest, and Support Vector Machine algorithms.
- Experiment with different K values.
- Implement hyperparameter tuning using GridSearchCV.
- Develop an interactive web application for flower classification.
- Test the model with additional flower measurements.

### 12. Conclusion

This project demonstrates the implementation of a supervised machine learning classification system using the Iris dataset and K-Nearest Neighbors algorithm.

It provides practical experience with data preprocessing, feature scaling, model training, prediction, and evaluation. The project establishes a foundation for understanding classification techniques and developing more advanced artificial intelligence applications.

---

**Project:** Data Classification Using AI  
**Organization:** DecodeLabs  
**Domain:** Artificial Intelligence and Machine Learning  
**Algorithm:** K-Nearest Neighbors (KNN)  
**Language:** Python