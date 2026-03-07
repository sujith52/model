import os
import threading
import webbrowser
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.ml.cleaning.cleaner import clean_and_restore_data # Import our cleaner
import pandas as pd
from fastapi.responses import HTMLResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
import joblib
# Import API router
from app.api.routes import router as api_router

# 1. INITIALIZE THE APP FIRST (Fixes NameError)
app = FastAPI(
    title="AI-Driven Insurance Fraud Detection",
    description="Insurance Fraud Detection using Machine Learning (SVM & XGBoost)",
    version="1.0.0"
)

# 2. SETUP PATHS
# Use absolute paths to ensure Render finds the folders correctly
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static")
template_dir = os.path.join(BASE_DIR, "templates")

# 3. CONFIGURATION
templates = Jinja2Templates(directory=template_dir)
templates = Jinja2Templates(directory="app/templates")

# Only mount if the directory actually exists
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"⚠️ Warning: Static directory NOT found at: {static_dir}")

# 4. INCLUDE ROUTES
app.include_router(api_router)

# 5. HOME ROUTE
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

# 6. AUTO-OPEN BROWSER (Local Only)
def open_browser():
    # This will silently do nothing on Render, which is fine
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except:
        pass

@app.on_event("startup")
def startup_event():
    threading.Timer(1.5, open_browser).start()

@app.get("/clean", response_class=HTMLResponse)
async def get_clean_page(request: Request):
    return templates.TemplateResponse("data.html", {"request": request})

@app.post("/api/clean-process")
async def process_cleaning(file: UploadFile = File(...)):
    # Load the uploaded file
    df = pd.read_csv(file.file)
    
    # Run our cleaning logic
    cleaned_df = clean_and_restore_data(df)
    
    # Save to output folder
    output_path = "output/cleaned_data.csv"
    os.makedirs("output", exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    
    return FileResponse(path=output_path, filename="cleaned_insurance_data.csv")





# 7. RUN LOGIC
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )