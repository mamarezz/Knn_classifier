# K-Nearest Neighbors (KNN) on Custom and Iris Datasets
This project demonstrates the implementation of the K-Nearest Neighbors (KNN) algorithm from scratch in Python. The project is divided into two parts:

1- Finding the Optimal K using LOOCV: A custom implementation of KNN is used to determine the optimal value of K using Leave-One-Out Cross-Validation (LOOCV) on a synthetic dataset.

2- KNN Classification on the Iris Dataset: The optimal K is then applied to classify the Iris dataset, a well-known dataset in machine learning.

Overview

Part 1: Finding the Optimal K

The first part of this project focuses on selecting the optimal value of K for the KNN algorithm. The loocv_knn function performs Leave-One-Out Cross-Validation (LOOCV) on a synthetic dataset, which consists of two classes. The function iterates over a range of K values, trains the KNN model, and evaluates its accuracy to determine the best K.

KNN Class: A simple KNN implementation that computes Euclidean distance to determine the closest neighbors.
LOOCV: This method evaluates the model’s performance by training on all data points except one and testing on the excluded point. This is repeated for each point in the dataset.

Part 2: KNN Classification on the Iris Dataset

In the second part, the optimal K value identified in the first part is used to classify the Iris dataset. The Iris dataset is loaded using sklearn.datasets, and the KNN classifier is applied to classify the species of iris flowers based on their features.

Data Visualization: 
The Iris dataset is visualized using a scatter plot.

Model Training and Testing: 
The dataset is split into training and testing sets, and the KNN classifier is trained on the training set.

Model Evaluation: The accuracy of the KNN classifier is evaluated on the test set.
# Code Structure
1- KNN Implementation: A custom implementation of the K-Nearest Neighbors algorithm with methods for fitting the model and making predictions.

2- LOOCV Function: A function that performs Leave-One-Out Cross-Validation to find the optimal K value.

3- Data Visualization: A simple visualization of the Iris dataset using matplotlib.

4- Model Evaluation: The accuracy of the model is computed and printed.
