from fastapi import APIRouter, HTTPException
from ..services.model_selection import ModelManager

router = APIRouter(prefix="/api", tags=["Model Analysis"])

@router.get("/models/performance")
def get_performance():
    try:
        manager = ModelManager.get_instance()
        return manager.get_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch model performance: {str(e)}")

@router.get("/patterns")
def get_patterns():
    try:
        manager = ModelManager.get_instance()
        return manager.get_patterns()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch K-Means patterns: {str(e)}")

@router.get("/models/feature-importance")
def get_feature_importance():
    try:
        manager = ModelManager.get_instance()
        return manager.get_feature_importance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feature importance: {str(e)}")
