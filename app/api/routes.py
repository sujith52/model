# FILE PATH: insurance_fraud_detection/app/api/routes.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import os
import uuid
import pandas as pd
from app.ml.predict import predict_fraud

router = APIRouter(prefix="/api", tags=["Fraud Detection"])

DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        return JSONResponse(status_code=400, content={"error": "Only CSV files are supported"})
    
    file_id = str(uuid.uuid4())
    input_path = os.path.join(DATA_DIR, f"{file_id}_input.csv")
    output_filename = f"{file_id}_output.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        # Run XGBoost Prediction logic
        output_df = predict_fraud(input_path, output_path)
        
        return {
            "total_rows": len(output_df),
            "fraud_cases": int((output_df["fraud_prediction"] == "Y").sum()),
            "safe_cases": int((output_df["fraud_prediction"] == "N").sum()),
            "output_file_id": output_filename
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/download/{file_id}")
async def download_file(file_id: str):
    file_path = os.path.join(OUTPUT_DIR, file_id)
    return FileResponse(path=file_path, filename="fraud_analysis_results.csv")