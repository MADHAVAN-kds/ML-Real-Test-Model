# ============================================
# Professional Regression ML Pipeline
# Hackathon-Ready | Optimized | Production Style
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --------------------------------------------
# 1. Data Loading
# --------------------------------------------

df = pd.read_csv("round1_L2_1.csv")
print("Dataset Shape:", df.shape)

# --------------------------------------------
# 2. Basic EDA
# --------------------------------------------

print("\nHead:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation Matrix:")
print(df.corr(numeric_only=True))

# --------------------------------------------
# 3. Data Cleaning
# --------------------------------------------

# Drop ID-like columns (no predictive value)
id_columns = ["sla_id", "route_id"]
df = df.drop(columns=[col for col in id_columns if col in df.columns])

# Clean inconsistent categorical values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

# Convert 'sla_hours' if contains text like "240.0 min"
if "sla_hours" in df.columns:
    df["sla_hours"] = df["sla_hours"].str.replace("min", "", regex=False)
    df["sla_hours"] = pd.to_numeric(df["sla_hours"], errors="coerce")

# --------------------------------------------
# 4. Feature Engineering
# --------------------------------------------

# Derived feature: numeric feature sum
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numeric_cols.remove(df.columns[-1])  # remove target

df["feature_sum"] = df[numeric_cols].sum(axis=1)

# --------------------------------------------
# 5. Split Features & Target
# --------------------------------------------

X = df.iloc[:, :-1]
y_raw = df.iloc[:, -1]

# Encode target if categorical
if y_raw.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
else:
    y = y_raw

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include="object").columns.tolist()

# --------------------------------------------
# 6. Preprocessing Pipeline
# --------------------------------------------

numeric_transformer = SimpleImputer(strategy="median")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# --------------------------------------------
# 7. Train-Test Split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------
# 8. Model Comparison
# --------------------------------------------

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

print("\nModel Comparison Results:")

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    
    print(f"\n{name}")
    print("R2:", round(r2_score(y_test, preds), 4))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_test, preds)), 4))
    print("5-Fold CV R2:", round(np.mean(cross_val_score(pipe, X, y, cv=5)), 4))

# --------------------------------------------
# 9. Hyperparameter Tuning (Random Forest)
# --------------------------------------------

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [None, 10]
}

grid = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters:", grid.best_params_)

# --------------------------------------------
# 10. Final Evaluation
# --------------------------------------------

final_preds = best_model.predict(X_test)

print("\nFinal Model Performance:")
print("R2:", round(r2_score(y_test, final_preds), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, final_preds)), 4))
print("5-Fold CV R2:", round(np.mean(cross_val_score(best_model, X, y, cv=5)), 4))

# --------------------------------------------
# 11. Feature Importance
# --------------------------------------------

rf_model = best_model.named_steps["model"]

encoded_cat = best_model.named_steps["preprocessor"]\
    .named_transformers_["cat"]\
    .named_steps["encoder"]\
    .get_feature_names_out(categorical_features)

all_features = numeric_features + list(encoded_cat)

importance_df = pd.DataFrame({
    "Feature": all_features,
    "Importance": rf_model.feature_importances_
}).sort_values(by="Importance", ascending=False).head(20)

plt.figure()
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.xticks(rotation=90)
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 12. Save Final Model
# --------------------------------------------

with open("final_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\nModel saved as final_model.pkl")
