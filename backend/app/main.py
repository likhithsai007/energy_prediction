from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes.forecast import router as forecast_router
from .routes.models import router as models_router
from .services.model_selection import ModelManager

app = FastAPI(
    title="Smart Energy Consumption Forecasting API",
    description="Backend API for Energy Forecasting, Peak Detection, K-Means Clustering, and Model Evaluation.",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(forecast_router)
app.include_router(models_router)

@app.on_event("startup")
def startup_event():
    print("Initializing Smart Energy Backend & Model Manager...")
    ModelManager.get_instance()
    print("Models and datasets loaded successfully.")

@app.get("/api/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "Smart Energy Consumption Forecasting API",
        "version": "1.0.0"
    }
