# ===============================================
# 🔥 COMPLETE PROFESSAL ML PIPELINE
# ===============================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ===============================================
# 1️⃣ LOAD DATA
# ===============================================

df = pd.read_csv("round1_L2_1.csv")

print("======================================")
print("Dataset Shape:", df.shape)
print("======================================")

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

print("\nCorrelation Matrix:")
print(df.corr())

# ===============================================
# 2️⃣ DATA CLEANING
# ===============================================

df = df.dropna()

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print("\nTarget Column:", df.columns[-1])

# ===============================================
# 3️⃣ FEATURE ENGINEERING (Example)
# ===============================================

# You can modify based on dataset
for col in X.columns:
    X[col + "_squared"] = X[col] ** 2

# ===============================================
# 4️⃣ TRAIN TEST SPLIT
# ===============================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================================
# 5️⃣ MODEL COMPARISON
# ===============================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

print("\n======================================")
print("Model Comparison")
print("======================================")

best_model = None
best_score = -999

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)

    print(f"{name} R2 Score: {r2}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print("\nBest Model:", best_name)

# ===============================================
# 6️⃣ HYPERPARAMETER TUNING (Random Forest)
# ===============================================

param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid.fit(X_train, y_train)

tuned_model = grid.best_estimator_

print("\nBest Parameters After Tuning:", grid.best_params_)

# ===============================================
# 7️⃣ FINAL EVALUATION
# ===============================================

pred = tuned_model.predict(X_test)

r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n======================================")
print("Final Model Performance")
print("======================================")
print("R2 Score:", r2)
print("RMSE:", rmse)

# ===============================================
# 8️⃣ CROSS VALIDATION
# ===============================================

cv_scores = cross_val_score(tuned_model, X, y, cv=5, scoring='r2')

print("\nCross Validation R2:", cv_scores.mean())

# ===============================================
# 9️⃣ FEATURE IMPORTANCE
# ===============================================

if hasattr(tuned_model, "feature_importances_"):
    importances = tuned_model.feature_importances_
    features = X.columns

    plt.figure()
    plt.bar(features, importances)
    plt.title("Feature Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# ===============================================
# 🔟 SAVE MODEL
# ===============================================

joblib.dump(tuned_model, "final_model.pkl")

print("\nModel saved as final_model.pkl")
print("======================================")
