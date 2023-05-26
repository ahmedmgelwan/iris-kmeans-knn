import pickle
import pandas as pd

# Load K-Means Model
with open('k_means.pkl', 'rb') as k_means_model:
    k_means = pickle.load(k_means_model)

# Load KNN Model
with open('knn.pkl', 'rb') as knn_model:
    knn = pickle.load(knn_model)

def process_input(w,x,y,z):
    data = pd.DataFrame([[x,y,z]])
    return data

def kmeans_pred(data):
    pred = k_means.predict(data)
    pred_scalar = pred[0]  # Access the first element of the NumPy array
    pred_dict ={
        0: 'virginica',
        1: 'setosa',
        2: 'versicolor',
    }
    return pred_dict[pred_scalar]

def knn_pred(data):
    pred = knn.predict(data)
    pred_scalar = pred[0]  # Access the first element of the NumPy array

    pred_dict = {
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    }
    return pred_dict[pred_scalar]

if __name__ == '__main__':
    # Create a sample data point
    data_point = pd.DataFrame([[ 2, 2, 1]])

    # Get predictions using the kmeans_pred and knn_pred functions
    kmeans_prediction = kmeans_pred(data_point)
    knn_prediction = knn_pred(data_point)

    print('K-Means Prediction:', kmeans_prediction)
    print('KNN Prediction:', knn_prediction)
