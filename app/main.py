# ==========================================================
# FILE PATH (TOP PRIORITY)
# insurance_fraud_detection/app/main.py
# ==========================================================

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import threading
import webbrowser

# Import API router (we will create this next)
from app.api.routes import router as api_router
import os
from fastapi.staticfiles import StaticFiles

# 1. Find where this current file (main.py) is located
current_file_path = os.path.abspath(__file__)
# 2. Find the directory containing main.py
current_dir = os.path.dirname(current_file_path)
# 3. Create a reliable path to the static folder
static_dir = os.path.join(current_dir, "static")

# Only mount if the directory actually exists to avoid the RuntimeError
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
else:
    print(f"❌ Error: Static directory NOT found at: {static_dir}")

# ==========================================================
# FASTAPI APP INITIALIZATION
# ==========================================================

app = FastAPI(
    title="AI-Driven Insurance Fraud Detection",
    description="Insurance Fraud Detection using Machine Learning (SVM & XGBoost)",
    version="1.0.0"
)


# ==========================================================
# TEMPLATE & STATIC FILE CONFIG
# ==========================================================

templates = Jinja2Templates(directory="app/templates")

app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static"
)


# ==========================================================
# INCLUDE API ROUTES
# ==========================================================

app.include_router(api_router)


# ==========================================================
# HOME ROUTE → LOAD index.html
# ==========================================================

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ==========================================================
# AUTO-OPEN BROWSER (DEMO FRIENDLY)
# ==========================================================

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

@app.on_event("startup")
def startup_event():
    threading.Timer(1.5, open_browser).start()


# ==========================================================
# RUN USING PYTHON (OPTIONAL)
# ==========================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
