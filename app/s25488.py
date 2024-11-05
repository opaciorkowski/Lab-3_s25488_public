import requests
import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from flask import Flask, request, jsonify

MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "scaler.pkl"

path = "CollegeDistance.csv"
url = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv"
with open(path, 'wb') as f:
    f.write(requests.get(url).content)

data = pd.read_csv(path)
data = data.drop(columns=data.columns[0])

categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

X = data_encoded.drop(columns=["score"])
y = data_encoded['score']

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model_xgb = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded from file.")
else:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    start_time = time.time()
    param_grid_xgb = {
        'n_estimators': [50, 100],
        'learning_rate': [0.1],
        'max_depth': [3, 5],
    }
    model_xgb = XGBRegressor(random_state=42, verbosity=0)
    grid_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=3, scoring='r2', n_jobs=-1)
    grid_xgb.fit(X_train, y_train)
    xgb_time = time.time() - start_time
    model_xgb = grid_xgb.best_estimator_

    joblib.dump(model_xgb, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model trained and saved to '{MODEL_PATH}'.")
    print(f"Scaler saved to '{SCALER_PATH}'.")

app = Flask(__name__)
@app.route('/')
def home():
    return "Use the /predict endpoint to make predictions"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            data = pd.DataFrame([data], columns=X.columns)
        else:
            data = pd.read_csv(request.files['file'], header=None)
            data.columns = X.columns
        
        # Encode categorical features as per training
        data_encoded = pd.get_dummies(data, drop_first=True)
        data_encoded = data_encoded.reindex(columns=X.columns, fill_value=0)
        
        # Scale data and make predictions
        data_scaled = scaler.transform(data_encoded.values)
        predictions = model_xgb.predict(data_scaled)
        
        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
