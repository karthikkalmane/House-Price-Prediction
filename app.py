import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model, scaler, and features
model = joblib.load('house_price_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# Convert model_features to a list if it's not already
if not isinstance(model_features, list):
    model_features = model_features.tolist()

# Load the dataset to get unique locations for the form
df = pd.read_csv('bengaluru_house_prices1.csv')
locations = sorted(df['location'].dropna().unique())

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    form_values = [x for x in request.form.values()]
    location = form_values[0]
    total_sqft = float(form_values[1])
    other_features = form_values[2:]

    # Prepare the input features
    input_features = [0] * len(model_features)
    other_features_index = 0  # Index to iterate over other_features
    for i, feature in enumerate(model_features):
        if feature == f'location_{location}':
            input_features[i] = 1
        elif feature == 'total_sqft':
            input_features[i] = total_sqft
        elif feature in ['bath', 'balcony', 'bhk']:
            input_features[i] = float(other_features[other_features_index])
            other_features_index += 1

    # Scale the numerical features
    numerical_indices = [model_features.index(f) for f in ['total_sqft', 'bath', 'balcony', 'bhk']]
    numerical_features = [input_features[idx] for idx in numerical_indices]
    numerical_features_scaled = scaler.transform([numerical_features])[0]
    for idx, value in zip(numerical_indices, numerical_features_scaled):
        input_features[idx] = value

    # Make prediction
    prediction_per_sqft = abs(model.predict([input_features])[0])
    prediction_total = prediction_per_sqft * total_sqft
    prediction_total_in_rupees = prediction_total * 85  

    # Round to the nearest thousand
    output = int(round(prediction_total_in_rupees, -3))
    
    # Convert to crore if greater than 50 million rupees
    #if output > 50000000:
     #   output = 50000000

    return jsonify(prediction=f'Predicted House Price: ₹{output}')

if __name__ == "__main__":
    app.run(debug=True)
