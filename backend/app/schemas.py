from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# --- Station Schemas ---
class StationBase(BaseModel):
    code: str
    name: str
    region: str
    latitude: float
    longitude: float
    elevation: float
    base_slope: float
    soil_type: str = "Colluvial Silt-Gravel"
    warning_threshold: float = 55.0
    critical_threshold: float = 80.0
    is_active: bool = True

class StationThresholdUpdate(BaseModel):
    warning_threshold: float = Field(..., ge=10.0, le=95.0)
    critical_threshold: float = Field(..., ge=20.0, le=100.0)

class StationResponse(StationBase):
    id: int
    created_at: datetime
    current_reading: Optional[Dict[str, Any]] = None
    current_prediction: Optional[Dict[str, Any]] = None
    active_alert: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


# --- Sensor Reading Schemas ---
class SensorReadingBase(BaseModel):
    rainfall_rate: float
    cumulative_rainfall_24h: float
    soil_moisture: float
    slope_angle: float
    vibration_frequency: float
    pore_water_pressure: float
    temperature: float

class SensorReadingCreate(SensorReadingBase):
    station_id: int
    timestamp: Optional[datetime] = None

class SensorReadingResponse(SensorReadingBase):
    id: int
    station_id: int
    timestamp: datetime

    class Config:
        from_attributes = True


# --- Prediction Schemas ---
class PredictionResponse(BaseModel):
    id: int
    reading_id: int
    station_id: int
    timestamp: datetime
    risk_score: float
    risk_level: str
    confidence: float
    contributing_factors: List[str]

    class Config:
        from_attributes = True

class AdHocPredictRequest(SensorReadingBase):
    pass

class AdHocPredictResponse(BaseModel):
    risk_score: float
    risk_level: str
    confidence: float
    class_probabilities: Dict[str, float]
    contributing_factors: List[str]


# --- Alert Schemas ---
class NotificationLogResponse(BaseModel):
    id: int
    alert_id: int
    channel: str
    recipient: str
    message: str
    status: str
    timestamp: datetime

    class Config:
        from_attributes = True

class AlertLogResponse(BaseModel):
    id: int
    station_id: int
    station_name: Optional[str] = None
    station_code: Optional[str] = None
    timestamp: datetime
    risk_level: str
    risk_score: float
    message: str
    contributing_factors: List[str]
    suggested_action: str
    status: str
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notifications: List[NotificationLogResponse] = []

    class Config:
        from_attributes = True

class AlertStatusUpdate(BaseModel):
    status: str  # ACKNOWLEDGED, RESOLVED


# --- Simulation Schemas ---
class ScenarioTriggerRequest(BaseModel):
    scenario_type: str  # "normal", "cloudburst", "pore_surge", "seismic", "escalating_demo"
    station_id: Optional[int] = None  # None for default or all
    duration_seconds: Optional[int] = 120
    intensity: Optional[float] = 1.0

class SimulationStatusResponse(BaseModel):
    is_running: bool
    interval_seconds: float
    active_scenarios: Dict[str, Any]
    current_tick: int
