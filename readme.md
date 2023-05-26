# Iris Dataset Web App Documentation

## Overview

The Iris Dataset Web App is a simple web application that allows users to enter values for the attributes of an iris flower and predicts its class using the K-Means clustering algorithm and the K-Nearest Neighbors (KNN) classification algorithm. The application is built using Python, Flask, and scikit-learn library.

## Installation

1. Clone the repository from GitHub:

```bash
git clone https://github.com/ahmedmgelwan/iris-kmeans-knn
```

2. Navigate to the project directory:

```bash
cd iris-kmeans-knn
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment:

- For Windows:

```bash
venv\Scripts\activate
```

- For Linux/Mac:

```bash
source venv/bin/activate
```

5. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask web server:

```bash
flask run
```

1. Access the web application in your browser at `http://localhost:5000`.
2. Enter the values for the attributes of an iris flower in the input fields provided.
3. Click on the "Predict" button to submit the values and view the predicted class of the iris flower.

## Algorithms

### K-Means Clustering

The K-Means clustering algorithm is used to group the iris flowers into clusters based on their attribute values. It aims to partition the data points into K distinct non-overlapping clusters, where each data point belongs to the cluster with the nearest mean value.

The steps involved in the K-Means algorithm are as follows:

1. Initialize K cluster centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update the centroids by computing the mean of the data points assigned to each cluster.
4. Repeat steps 2 and 3 until convergence (when the centroids no longer change significantly).

### K-Nearest Neighbors (KNN) Classification

The K-Nearest Neighbors (KNN) algorithm is used for classification based on the attributes of the iris flowers. It assigns a class label to a new data point based on the majority vote of its K nearest neighbors in the training dataset.

The steps involved in the KNN algorithm are as follows:

1. Choose the number of nearest neighbors K.
2. Calculate the distance between the new data point and all other data points in the training dataset using a distance metric (e.g., Euclidean distance).
3. Sort the distances in ascending order and select the top K data points with the shortest distances.
4. Determine the class labels of the selected K data points.
5. Assign the class label to the new data point based on the majority vote of the class labels of the K nearest neighbors.
6. Repeat the process for each new data point to be classified.

Note: It's important to choose an appropriate value for K, as a smaller K may lead to overfitting and a larger K may lead to underfitting.

## Conclusion

The Iris Prediction Web App provides a user-friendly interface to predict the class of an iris flower based on its attribute values. It leverages the K-Means clustering algorithm to group the iris flowers into clusters and the KNN classification algorithm to assign class labels to new data points. The application is easy to install and use, making it accessible for users who want to explore and understand the iris dataset.