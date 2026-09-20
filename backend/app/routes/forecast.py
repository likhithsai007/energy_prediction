from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ..services.forecasting import generate_forecast

router = APIRouter(prefix="/api", tags=["Forecast"])

class ForecastRequest(BaseModel):
    date: str = Field(..., example="2026-09-22", description="Forecast target date in YYYY-MM-DD format")
    start_time: str = Field(default="00:00", example="00:00", description="Start time in HH:MM format")
    end_time: str = Field(default="23:59", example="23:59", description="End time in HH:MM format")

@router.post("/forecast")
def create_forecast(request: ForecastRequest):
    try:
        result = generate_forecast(
            date_str=request.date,
            start_time_str=request.start_time,
            end_time_str=request.end_time
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {str(e)}")
