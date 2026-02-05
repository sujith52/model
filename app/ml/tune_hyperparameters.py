# ==========================================================
# FILE PATH:
# insurance_fraud_detection/app/ml/tune_hyperparameters.py
# ==========================================================

import pandas as pd
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

# ==========================================================
# ARGUMENT PARSER
# ==========================================================
parser = argparse.ArgumentParser(description='Tune XGBoost hyperparameters for fraud detection.')
parser.add_argument('--dataset', type=str, default='data/insuranceFraud_Dataset (1).csv', help='Path to the dataset CSV file.')
args = parser.parse_args()

# ==========================================================
# PATHS
# ==========================================================
DATASET_PATH = args.dataset
MODEL_PATH = "app/models/fraud_xgb_model.pkl"

# ==========================================================
# LOAD AND PREPARE DATA (Same as train_model.py)
# ==========================================================
if not os.path.exists(DATASET_PATH):
    print(f"Error: The dataset file was not found at: {DATASET_PATH}")
    exit()

df = pd.read_csv(DATASET_PATH)
TARGET_COL = "fraud_reported"

if TARGET_COL not in df.columns:
    print(f"Error: Target column '{TARGET_COL}' not found.")
    exit()

df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================================
# HYPERPARAMETER GRID
# ==========================================================
# Define a grid of hyperparameters to search. This is a focused search 
# around commonly tuned parameters for XGBoost.
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of trees
    'max_depth': [4, 6, 8],               # Maximum depth of a tree
    'learning_rate': [0.05, 0.1, 0.2],    # Step size shrinkage
    'subsample': [0.7, 0.8],              # Fraction of samples used for fitting
    'colsample_bytree': [0.7, 0.8]        # Fraction of features used for fitting
}

# ==========================================================
# GRID SEARCH WITH CROSS-VALIDATION
# ==========================================================
# We use F1-score as the scoring metric because it is a good balance
# between precision and recall, which is crucial for imbalanced datasets.
f1_scorer = make_scorer(f1_score)

xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# GridSearchCV will exhaustively search through the param_grid and find the best combination.
# cv=3 means 3-fold cross-validation.
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring=f1_scorer, cv=3, verbose=1, n_jobs=-1)

print("Starting hyperparameter tuning...")
grid_search.fit(X_train, y_train)

print("Hyperparameter tuning completed.")

# ==========================================================
# SAVE THE BEST MODEL
# ==========================================================
# The best estimator found by GridSearchCV is now our new, improved model.
best_model = grid_search.best_estimator_

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": best_model,
        "encoders": label_encoders,
        "columns": X.columns.tolist()
    }, f)

print("\n=========================================")
print("       Hyperparameter Tuning Results     ")
print("=========================================")
print(f"  Best F1-Score (on CV): {grid_search.best_score_:.4f}")
print("\n  Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"    {param}: {value}")
print("\n✅ Best XGBoost model saved successfully!")
print("=========================================")

# Optionally, evaluate on the test set to see the final performance
final_f1_score = f1_score(y_test, best_model.predict(X_test))
print(f"\n  F1-Score on Test Set: {final_f1_score:.4f}")
print("=========================================")
