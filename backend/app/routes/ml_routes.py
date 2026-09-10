import asyncio
from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..schemas import AdHocPredictRequest, AdHocPredictResponse
from ...ml.predictor import LandslidePredictor
from ...ml.train import train_and_save_models, FEATURE_COLUMNS

router = APIRouter(prefix="/model", tags=["Machine Learning"])

@router.post("/predict", response_model=AdHocPredictResponse)
def predict_adhoc_risk(payload: AdHocPredictRequest):
    predictor = LandslidePredictor.get_instance()
    try:
        features = payload.model_dump()
        result = predictor.predict(features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/metrics")
def get_model_metrics():
    predictor = LandslidePredictor.get_instance()
    metrics = predictor.get_metrics()
    if not metrics:
        raise HTTPException(status_code=404, detail="Model metrics not found.")
    return metrics


@router.get("/features")
def get_features_info():
    return {
        "features": [
            {
                "name": "rainfall_rate",
                "unit": "mm/hr",
                "description": "Instantaneous precipitation intensity recorded by tipping bucket rain gauge.",
                "typical_range": "0 - 150 mm/hr"
            },
            {
                "name": "cumulative_rainfall_24h",
                "unit": "mm",
                "description": "Antecedent 24-hour aggregate precipitation causing regolith hydration.",
                "typical_range": "0 - 500 mm"
            },
            {
                "name": "soil_moisture",
                "unit": "%",
                "description": "Volumetric water content of mountain slope subsoil.",
                "typical_range": "10% - 100%"
            },
            {
                "name": "slope_angle",
                "unit": "degrees (°)",
                "description": "Real-time inclinometer reading reflecting slope inclination and shear creep.",
                "typical_range": "15° - 65°"
            },
            {
                "name": "vibration_frequency",
                "unit": "Hz",
                "description": "Geophone micro-tremor and seismic ground acceleration frequency.",
                "typical_range": "0.05 - 5.5 Hz"
            },
            {
                "name": "pore_water_pressure",
                "unit": "kPa",
                "description": "Vibrating wire piezometer pressure measuring hydraulic uplift in failure plane.",
                "typical_range": "0 - 100 kPa"
            },
            {
                "name": "temperature",
                "unit": "°C",
                "description": "Ambient environmental air and topsoil temperature.",
                "typical_range": "2°C - 42°C"
            }
        ]
    }


def background_retrain():
    try:
        print("Starting model retraining in background...")
        train_and_save_models()
        predictor = LandslidePredictor.get_instance()
        predictor.reload()
        print("Background retraining and reload complete.")
    except Exception as e:
        print(f"Background retraining failed: {e}")


@router.post("/train")
def trigger_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_retrain)
    return {
        "status": "initiated",
        "message": "Model retraining initiated in the background. Metrics will update automatically once finished."
    }
