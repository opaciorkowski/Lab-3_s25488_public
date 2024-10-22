import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import time

#Pobranie danych
path = "CollegeDistance.csv"
url = "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv"
with open(path, 'wb') as f:
    f.write(requests.get(url).content)

data = pd.read_csv(path)

#Usunięcie pierwszej kolumny
data = data.drop(columns=data.columns[0])

#Zamiana wartości tekstowych na liczbowe
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

#Usunięcie score z X, gdyż to go będziemy chcieli przewidywać
X = data_encoded.drop(columns=["score"])

y = data_encoded['score']

#Normalizacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Podział 80:20 na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Linear Regression
start_time = time.time()
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
lr_time = time.time() - start_time

#Random Forest
start_time = time.time()
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
}
model_rf = RandomForestRegressor(random_state=42)
grid_rf = GridSearchCV(model_rf, param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
y_pred_rf = grid_rf.best_estimator_.predict(X_test)
rf_time = time.time() - start_time

#XGBoost
start_time = time.time()
param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.1],
    'max_depth': [3, 5],      
}
model_xgb = XGBRegressor(random_state=42, verbosity=0)
grid_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=3, scoring='r2', n_jobs=-1)
grid_xgb.fit(X_train, y_train)
y_pred_xgb = grid_xgb.best_estimator_.predict(X_test)
xgb_time = time.time() - start_time

#Ewaluacja
def evaluate_model(y_true, y_pred, model_name, exec_time):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    success_percentage = r2 * 100
    print(f"\nModel: {model_name}")
    print(f"Execution Time: {exec_time:.2f} seconds")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    print(f"Model Success Percentage: {success_percentage}%")
    return mse, mae, r2, success_percentage, exec_time, model_name, y_pred

results_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression", lr_time)
results_rf = evaluate_model(y_test, y_pred_rf, "Random Forest Regressor", rf_time)
results_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost Regressor", xgb_time)

#Wybranie najlepszego modelu na podstawie R²
if results_lr[2] > results_rf[2] and results_lr[2] > results_xgb[2]:
    best_model = results_lr
elif results_rf[2] > results_xgb[2]:
    best_model = results_rf
else:
    best_model = results_xgb

best_model_name = best_model[5]
best_model_r2 = best_model[2]
y_pred_best = best_model[6]

print(f"\nBest Model: {best_model_name} (R² Score: {best_model_r2})")

#Porównanie predykcji ze stanem faktycznym w najlepszym modelu
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.title(f'Actual vs Predicted Scores ({best_model_name})')
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.savefig(f'actual_vs_predicted_{best_model_name.lower().replace(" ", "_")}.png')
plt.show()


residuals = y_test - y_pred_best
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_best, residuals, alpha=0.5)
plt.hlines(0, y_pred_best.min(), y_pred_best.max(), colors='red', linestyles='dashed')
plt.title(f'Residual Plot ({best_model_name})')
plt.xlabel('Predicted Scores')
plt.ylabel('Residuals')
plt.savefig(f'residual_plot_{best_model_name.lower().replace(" ", "_")}.png')
plt.show()


plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title(f'Distribution of Residuals ({best_model_name})')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig(f'residual_distribution_{best_model_name.lower().replace(" ", "_")}.png')
plt.show()

#Znaczenie cech
if best_model_name in ["Random Forest Regressor", "XGBoost Regressor"]:
    if best_model_name == "Random Forest Regressor":
        model = grid_rf.best_estimator_
    else:
        model = grid_xgb.best_estimator_

    plt.figure(figsize=(10, 8))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns[indices]

    plt.title(f'Feature Importance ({best_model_name})')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()
    plt.savefig(f'feature_importance_{best_model_name.lower().replace(" ", "_")}.png')
    plt.show()

def save_summary_to_txt(results_lr, results_rf, results_xgb, best_model_name, best_model_r2):
    with open('report.txt', 'w') as file:
        #Linear Regression
        file.write("Linear Regression:\n")
        file.write(f"Execution Time: {results_lr[4]:.2f} seconds\n")
        file.write(f"Mean Squared Error (MSE): {results_lr[0]}\n")
        file.write(f"Mean Absolute Error (MAE): {results_lr[1]}\n")
        file.write(f"R² Score: {results_lr[2]}\n")
        file.write(f"Model Success Percentage: {results_lr[3]}%\n\n")

        #Random Forest
        file.write("Random Forest:\n")
        file.write(f"Execution Time: {results_rf[4]:.2f} seconds\n")
        file.write(f"Mean Squared Error (MSE): {results_rf[0]}\n")
        file.write(f"Mean Absolute Error (MAE): {results_rf[1]}\n")
        file.write(f"R² Score: {results_rf[2]}\n")
        file.write(f"Model Success Percentage: {results_rf[3]}%\n\n")

        #XGBoost
        file.write("XGBoost:\n")
        file.write(f"Execution Time: {results_xgb[4]:.2f} seconds\n")
        file.write(f"Mean Squared Error (MSE): {results_xgb[0]}\n")
        file.write(f"Mean Absolute Error (MAE): {results_xgb[1]}\n")
        file.write(f"R² Score: {results_xgb[2]}\n")
        file.write(f"Model Success Percentage: {results_xgb[3]}%\n\n")

        #Podsmuowanie
        file.write("Best Model:\n")
        file.write(f"Model Name: {best_model_name}\n")
        file.write(f"R² Score: {best_model_r2}\n\n")