import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load the dataset
df = pd.read_csv('bengaluru_house_prices.csv')

# Data Cleaning
df = df.drop(columns=['area_type', 'society', 'availability'], axis=1)

# Handle missing values
df['bath'] = df['bath'].fillna(df['bath'].median())
df['balcony'] = df['balcony'].fillna(df['balcony'].median())
df = df.dropna(subset=['size', 'total_sqft', 'location'])

# Feature Engineering
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]) if pd.notnull(x) else None)
df = df.drop(columns=['size'])

def convert_sqft_to_num(x):
    try:
        return float(x)
    except:
        if '-' in x:
            parts = x.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
df = df.dropna(subset=['total_sqft'])

# One-Hot Encoding for 'location'
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Feature Scaling
scaler = StandardScaler()
numerical_features = ['total_sqft', 'bath', 'balcony', 'bhk']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Split the data
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
#print(f"Root Mean Squared Error: {rmse}")

# Save the model and scaler
joblib.dump(model, 'house_price_prediction_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'model_features.pkl')
