
# AI-Driven Insurance Fraud Detection System

This project is a full-stack Machine Learning application designed to detect fraudulent insurance claims using **FastAPI** and **XGBoost**.

> **Data Source:** The core dataset is sourced from the open-source **Kaggle** community: [Auto Insurance Claims Data](https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data).

---

## 🚀 Getting Started

### 1. Installation
Ensure you have Python 3.11+ installed. Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt

```


### 2. Run the Application

Start the FastAPI server with the following command:

```bash
uvicorn app.main:app --reload

```

Once started, open your browser to `http://127.0.0.1:8000` to access the UI.

---

## 🧠 Model Training and Optimization

The model logic is located in the `app/ml/` directory. Use these scripts to build and refine your detector.

### 1. Train the Model with SMOTE

This script trains the XGBoost model and applies **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance, ensuring the model better identifies rare fraud cases.

```bash
python app/ml/train_model.py

```

* **Output:** Saves the trained model to `app/models/fraud_xgb_model.pkl`.

### 2. Tune Hyperparameters (Optional)

To find the optimal configuration for your XGBoost model, run the tuning script. This uses Grid Search to find the best parameters and automatically updates the saved model.

```bash
python app/ml/tune_hyperparameters.py

```

### 3. Report Performance Metrics

Evaluate how well the model is performing on unseen data. This script prints **Accuracy, Precision, Recall, and F1-Score**.

```bash
python app/ml/report_metrics.py

```

---

## 🛠 Tech Stack

* **Backend:** FastAPI (Python)
* **Machine Learning:** XGBoost, Scikit-learn, Pandas, Numpy
* **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
* **Environment:** Google IDX / Nix

---

## 📋 Project Structure

* `app/api/`: API endpoints and routing logic.
* `app/ml/`: Training, prediction, and metric reporting scripts.
* `app/models/`: Directory for stored `.pkl` model files.
* `app/templates/`: Web interface components.
* `data/`: Storage for raw and processed CSV datasets.

---

