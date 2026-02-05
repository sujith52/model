# ==========================================================
# FILE PATH:
# insurance_fraud_detection/app/ml/report_metrics.py
# ==========================================================

import pandas as pd
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================================
# ARGUMENT PARSER
# ==========================================================
# Set up an argument parser to allow specifying the evaluation dataset file path from the command line.
parser = argparse.ArgumentParser(description='Evaluate the XGBoost fraud detection model.')
parser.add_argument('--dataset', type=str, default='data/insuranceFraud_Dataset (1).csv', help='Path to the evaluation dataset CSV file.')
args = parser.parse_args()

# ==========================================================
# PATHS
# ==========================================================
DATASET_PATH = args.dataset
MODEL_PATH = "app/models/fraud_xgb_model.pkl"

# ==========================================================
# LOAD MODEL AND DATA
# ==========================================================
# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first.")
    exit()

# Load the trained model, encoders, and column names
with open(MODEL_PATH, "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
label_encoders = saved_objects["encoders"]
trained_columns = saved_objects["columns"]

# Check if the dataset file exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: The dataset file was not found at: {DATASET_PATH}")
    exit()

df = pd.read_csv(DATASET_PATH)

# ==========================================================
# PREPARE DATA (IDENTICAL TO TRAINING)
# ==========================================================
TARGET_COL = "fraud_reported"

if TARGET_COL not in df.columns:
    print(f"Error: Target column '{TARGET_COL}' not in the dataset. Cannot evaluate performance.")
    exit()

# Encode target variable
df[TARGET_COL] = df[TARGET_COL].map({'Y': 1, 'N': 0})

# Separate features and labels
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode categorical features using the *same* encoders from training
for col, le in label_encoders.items():
    if col in X.columns:
        # Filter out unseen labels in the test data before transforming
        unseen_labels = set(X[col].astype(str)) - set(le.classes_)
        if unseen_labels:
            print(f"Warning: Unseen labels in column '{col}' will be ignored: {unseen_labels}")
            X = X[~X[col].isin(unseen_labels)]
            y = y[X.index] # Align labels with features

        X[col] = le.transform(X[col].astype(str))

# Ensure column order is the same as during training
X = X[trained_columns]

# ==========================================================
# MAKE PREDICTIONS
# ==========================================================

# Use the same train-test split to get a consistent evaluation set
_, X_test, _, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

y_pred = model.predict(X_test)

# ==========================================================
# CALCULATE AND DISPLAY METRICS
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=========================================")
print("      Model Performance Metrics          ")
print("=========================================")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print("-----------------------------------------")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Not Fraud (N)', 'Fraud (Y)']))
print("=========================================")

