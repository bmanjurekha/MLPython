import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Test"

@app.route('/predict',methods=["POST"])
def predict():

    json = request.json
    query_dataframe = pd.DataFrame(json)
    prediction = model.predict(query_dataframe)

    output = round(prediction[3], 2)
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)