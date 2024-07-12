# app.py
from flask import Flask, request, jsonify, render_template
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('model/iris_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = [float(data['SepalLengthCm']),
                    float(data['SepalWidthCm']),
                    float(data['PetalLengthCm']),
                    float(data['PetalWidthCm'])]
        prediction = model.predict([features])[0]
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
