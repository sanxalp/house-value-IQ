from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = load_model('house_price_model.keras') 
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

@app.route('/')
def index():
    return render_template("C:\\Users\\SANKALP\Desktop\\ml-project\housing\\templates\\index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array([[data['crim'], data['zn'], data['indus'], data['chas'], 
                          data['nox'], data['rm'], data['age'], data['dis'], 
                          data['rad'], data['tax'], data['ptratio'], data['b'], 
                          data['lstat']]])
    
    # Scale the features
    features_scaled = scaler.transform(features)  # Use transform instead of fit_transform
    features_lstm = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))

    # Make prediction
    prediction = model.predict(features_lstm)
    return jsonify({'predicted_price': prediction[0][0]})

if __name__ == '__main__':
    app.run(debug=True)
