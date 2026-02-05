# ==========================================================
# FILE PATH:
# insurance_fraud_detection/app/ml/predict.py
# ==========================================================

import pandas as pd
import pickle


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

MODEL_PATH = "app/models/fraud_xgb_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved_objects = pickle.load(f)

model = saved_objects["model"]
label_encoders = saved_objects["encoders"]
trained_columns = saved_objects["columns"]


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_fraud(input_csv_path: str, output_csv_path: str):
    """
    Reads input CSV (WITHOUT fraud_reported),
    predicts fraud,
    saves output CSV with fraud_prediction column
    """

    # Load input data
    df = pd.read_csv(input_csv_path)

    # Keep original data for output
    output_df = df.copy()
    
    # --- Start of fix ---
    # Check for columns that the model was trained on but are missing in the input file.
    missing_cols = set(trained_columns) - set(df.columns)
    if missing_cols:
        # If there are missing columns, we cannot proceed with the prediction.
        # Raise a ValueError with a clear message about which columns are missing.
        raise ValueError(f"The uploaded file is missing the following required columns: {', '.join(missing_cols)}")

    # ======================================================
    # ENCODE CATEGORICAL FEATURES (SAME AS TRAINING)
    # ======================================================

    for col, le in label_encoders.items():
        if col in df.columns:
            # Check for new, unseen categorical values that were not present in the training data.
            unseen_labels = set(df[col].astype(str)) - set(le.classes_)
            if unseen_labels:
                # If there are unseen labels, the label encoder will fail.
                # Raise a ValueError with a clear message.
                raise ValueError(
                    f"The column '{col}' contains new values that the model hasn't seen before: "
                    f"{', '.join(list(unseen_labels)[:5])}" # Show first 5 unseen
                    f"{', ...' if len(unseen_labels) > 5 else ''}. "
                    "Please ensure the input data matches the training data format."
                )
            # If all values are known, transform the column.
            df[col] = le.transform(df[col].astype(str))
    # --- End of fix ---


    # Ensure column order matches training
    df = df[trained_columns]

    # ======================================================
    # MODEL PREDICTION
    # ======================================================

    predictions = model.predict(df)

    # Convert back to labels
    output_df["fraud_prediction"] = ["Y" if p == 1 else "N" for p in predictions]

    # Save output CSV
    output_df.to_csv(output_csv_path, index=False)

    return output_df
