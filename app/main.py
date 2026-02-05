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
