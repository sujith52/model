import os
import threading
import webbrowser
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

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

# 7. RUN LOGIC
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )