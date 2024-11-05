# Supervised Learning: Breast Cancer Classification
This project implements several classification algorithms on the breast cancer dataset from sklearn to predict whether a tumor is malignant or benign. The following steps are covered in the analysis:

### 1. Loading and Preprocessing the Dataset
### 2. Implementation of Five Classification Algorithms
### 3. Comparison of Model Performance
## Objective
The main objective of this project is to demonstrate an understanding of supervised learning techniques and apply them to a real-world dataset (the breast cancer dataset) to classify tumors as either malignant or benign.

## Dataset
The dataset used for this project is the Breast Cancer Wisconsin (Diagnostic) dataset, available through the sklearn.datasets module. This dataset contains 30 features extracted from breast cancer cell images, such as radius, texture, smoothness, etc. The target variable indicates whether a tumor is benign (0) or malignant (1).

## Preprocessing Steps
### Loading the Dataset: 
The dataset is loaded using sklearn.datasets.load_breast_cancer(). The dataset consists of 30 feature columns and a target column indicating the class of the tumor.

### Handling Missing Values: 
The breast cancer dataset from sklearn does not have missing values, so no imputation was necessary.

### Feature Scaling:
The features are standardized using StandardScaler to ensure that each feature contributes equally to the classification models. Feature scaling is particularly important for models like Logistic Regression, SVM, and k-NN, which are sensitive to the scale of the input data.

### Data Split:
The dataset is split into a training set (80%) and a test set (20%) using train_test_split from sklearn.model_selection. This ensures that the model can be trained and evaluated effectively.

## Classification Algorithms Implemented
### Logistic Regression:
A linear model that outputs the probability of a class label using the sigmoid function. Suitable for problems where the relationship between features and the target is approximately linear.

### Decision Tree Classifier: 
A non-linear model that creates a tree-like structure to make decisions. It is interpretable but can easily overfit without proper regularization.

### Random Forest Classifier:
An ensemble method that aggregates the predictions from multiple decision trees. It is less prone to overfitting compared to individual decision trees and works well on complex datasets.

### Support Vector Machine (SVM):
A classifier that finds the hyperplane that best separates the data into classes. It is effective for both linear and non-linear problems, particularly in high-dimensional spaces.

### k-Nearest Neighbors (k-NN): 
A non-parametric algorithm that classifies a data point based on the majority class of its nearest neighbors. It works well for small datasets and when the decision boundary is not linear.

