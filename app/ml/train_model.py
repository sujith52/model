# ==========================================================
# FILE PATH:
# insurance_fraud_detection/app/ml/train_model.py
# ==========================================================

import pandas as pd
import pickle
import os
import argparse

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE  # <-- Import SMOTE

# ==========================================================
# ARGUMENT PARSER
# ==========================================================
parser = argparse.ArgumentParser(description='Train an XGBoost model for fraud detection.')
parser.add_argument('--dataset', type=str, default='data/insuranceFraud_Dataset (1).csv', help='Path to the training dataset CSV file.')
args = parser.parse_args()

# ==========================================================
# PATHS
# ==========================================================
DATASET_PATH = args.dataset
MODEL_PATH = "app/models/fraud_xgb_model.pkl"

# ==========================================================
# LOAD DATA
# ==========================================================
if not os.path.exists(DATASET_PATH):
    print(f"Error: The dataset file was not found at: {DATASET_PATH}")
    exit()

df = pd.read_csv(DATASET_PATH)

TARGET_COL = "fraud_reported"

if TARGET_COL not in df.columns:
    print(f"Error: The target column '{TARGET_COL}' was not found in the dataset.")
    exit()

df[TARGET_COL] = df[TARGET_COL].map({"Y": 1, "N": 0})

# ==========================================================
# FEATURE / LABEL SPLIT
# ==========================================================
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# ==========================================================
# HANDLE CATEGORICAL FEATURES
# ==========================================================
label_encoders = {}
original_columns = X.columns.tolist() #<-- Store original columns

for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# APPLY SMOTE TO THE TRAINING DATA
# ==========================================================
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training set shape: {X_train.shape}")
print(f"Resampled training set shape: {X_train_resampled.shape}")
print("SMOTE applied successfully.")

# ==========================================================
# LOAD BEST PARAMETERS (FROM TUNING)
# ==========================================================

# Use the best parameters found during the hyperparameter tuning step
best_params = {
    'colsample_bytree': 0.7,
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 200,
    'subsample': 0.7,
    'eval_metric': "logloss",
    'random_state': 42
}

model = XGBClassifier(**best_params)

# Train the model on the RESAMPLED data
model.fit(X_train_resampled, y_train_resampled)

# ==========================================================
# SAVE MODEL + ENCODERS + ORIGINAL COLUMNS
# ==========================================================
os.makedirs("app/models", exist_ok=True)

with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": label_encoders,
        "columns": original_columns  # <-- Save the original column order
    }, f)

print(f"\n✅ XGBoost Fraud Model (Trained with SMOTE) on '{DATASET_PATH}' & Saved Successfully")
