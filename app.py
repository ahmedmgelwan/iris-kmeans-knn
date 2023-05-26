from flask import Flask, render_template,request
import pickle
from make_pred import *
# import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal-length'])
        sepal_width = float(request.form['sepal-width'])
        petal_length = float(request.form['petal-length'])
        petal_width = float(request.form['petal-width'])
        # row = np.array([[sepal_width, petal_length, petal_width]])
        data_point = process_input(sepal_length, sepal_width,petal_length, petal_width)
        kmeans_prediction = kmeans_pred(data_point)
        knn_prediction = knn_pred(data_point)
        return render_template('home.html', prediction=(kmeans_prediction, knn_prediction))
    return render_template('home.html')



if __name__ == '__main__':
    app.run(debug=True)