from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ...database.session import get_db
from ...database.models import Station, SensorReading, Prediction, AlertLog
from ..schemas import StationResponse, StationThresholdUpdate, SensorReadingResponse

router = APIRouter(prefix="/stations", tags=["Stations"])

@router.get("", response_model=List[StationResponse])
def get_all_stations(db: Session = Depends(get_db)):
    stations = db.query(Station).filter(Station.is_active == True).all()
    results = []

    for s in stations:
        # Get latest reading
        latest_reading = (
            db.query(SensorReading)
            .filter(SensorReading.station_id == s.id)
            .order_by(SensorReading.timestamp.desc())
            .first()
        )
        # Get latest prediction
        latest_pred = (
            db.query(Prediction)
            .filter(Prediction.station_id == s.id)
            .order_by(Prediction.timestamp.desc())
            .first()
        )
        # Get active alert if any
        active_alert = (
            db.query(AlertLog)
            .filter(AlertLog.station_id == s.id, AlertLog.status.in_(["ACTIVE", "ACKNOWLEDGED"]))
            .order_by(AlertLog.timestamp.desc())
            .first()
        )

        reading_dict = None
        if latest_reading:
            reading_dict = {
                "id": latest_reading.id,
                "timestamp": latest_reading.timestamp.isoformat(),
                "rainfall_rate": latest_reading.rainfall_rate,
                "cumulative_rainfall_24h": latest_reading.cumulative_rainfall_24h,
                "soil_moisture": latest_reading.soil_moisture,
                "slope_angle": latest_reading.slope_angle,
                "vibration_frequency": latest_reading.vibration_frequency,
                "pore_water_pressure": latest_reading.pore_water_pressure,
                "temperature": latest_reading.temperature
            }

        pred_dict = None
        if latest_pred:
            pred_dict = {
                "id": latest_pred.id,
                "timestamp": latest_pred.timestamp.isoformat(),
                "risk_score": latest_pred.risk_score,
                "risk_level": latest_pred.risk_level,
                "confidence": latest_pred.confidence,
                "contributing_factors": latest_pred.contributing_factors or []
            }

        alert_dict = None
        if active_alert:
            alert_dict = {
                "id": active_alert.id,
                "risk_level": active_alert.risk_level,
                "risk_score": active_alert.risk_score,
                "message": active_alert.message,
                "status": active_alert.status,
                "timestamp": active_alert.timestamp.isoformat(),
                "suggested_action": active_alert.suggested_action
            }

        station_dict = {
            "id": s.id,
            "code": s.code,
            "name": s.name,
            "region": s.region,
            "latitude": s.latitude,
            "longitude": s.longitude,
            "elevation": s.elevation,
            "base_slope": s.base_slope,
            "soil_type": s.soil_type,
            "warning_threshold": s.warning_threshold,
            "critical_threshold": s.critical_threshold,
            "is_active": s.is_active,
            "created_at": s.created_at,
            "current_reading": reading_dict,
            "current_prediction": pred_dict,
            "active_alert": alert_dict
        }
        results.append(station_dict)

    return results


@router.get("/{station_id}", response_model=StationResponse)
def get_station_details(station_id: int, db: Session = Depends(get_db)):
    station = db.query(Station).filter(Station.id == station_id).first()
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")

    latest_reading = (
        db.query(SensorReading)
        .filter(SensorReading.station_id == station.id)
        .order_by(SensorReading.timestamp.desc())
        .first()
    )
    latest_pred = (
        db.query(Prediction)
        .filter(Prediction.station_id == station.id)
        .order_by(Prediction.timestamp.desc())
        .first()
    )
    active_alert = (
        db.query(AlertLog)
        .filter(AlertLog.station_id == station.id, AlertLog.status.in_(["ACTIVE", "ACKNOWLEDGED"]))
        .order_by(AlertLog.timestamp.desc())
        .first()
    )

    return {
        "id": station.id,
        "code": station.code,
        "name": station.name,
        "region": station.region,
        "latitude": station.latitude,
        "longitude": station.longitude,
        "elevation": station.elevation,
        "base_slope": station.base_slope,
        "soil_type": station.soil_type,
        "warning_threshold": station.warning_threshold,
        "critical_threshold": station.critical_threshold,
        "is_active": station.is_active,
        "created_at": station.created_at,
        "current_reading": {
            "id": latest_reading.id,
            "timestamp": latest_reading.timestamp.isoformat(),
            "rainfall_rate": latest_reading.rainfall_rate,
            "cumulative_rainfall_24h": latest_reading.cumulative_rainfall_24h,
            "soil_moisture": latest_reading.soil_moisture,
            "slope_angle": latest_reading.slope_angle,
            "vibration_frequency": latest_reading.vibration_frequency,
            "pore_water_pressure": latest_reading.pore_water_pressure,
            "temperature": latest_reading.temperature
        } if latest_reading else None,
        "current_prediction": {
            "id": latest_pred.id,
            "timestamp": latest_pred.timestamp.isoformat(),
            "risk_score": latest_pred.risk_score,
            "risk_level": latest_pred.risk_level,
            "confidence": latest_pred.confidence,
            "contributing_factors": latest_pred.contributing_factors or []
        } if latest_pred else None,
        "active_alert": {
            "id": active_alert.id,
            "risk_level": active_alert.risk_level,
            "risk_score": active_alert.risk_score,
            "message": active_alert.message,
            "status": active_alert.status,
            "timestamp": active_alert.timestamp.isoformat(),
            "suggested_action": active_alert.suggested_action
        } if active_alert else None
    }


@router.get("/{station_id}/readings")
def get_station_readings(
    station_id: int,
    limit: int = Query(default=60, le=500),
    db: Session = Depends(get_db)
):
    """
    Returns time series readings merged with their ML predictions for charts.
    """
    station = db.query(Station).filter(Station.id == station_id).first()
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")

    readings = (
        db.query(SensorReading)
        .filter(SensorReading.station_id == station_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(limit)
        .all()
    )

    # Return chronological order (oldest to newest) for chart plotting
    readings = list(reversed(readings))

    chart_data = []
    for r in readings:
        pred = db.query(Prediction).filter(Prediction.reading_id == r.id).first()
        chart_data.append({
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "time_label": r.timestamp.strftime("%H:%M:%S"),
            "rainfall_rate": r.rainfall_rate,
            "cumulative_rainfall_24h": r.cumulative_rainfall_24h,
            "soil_moisture": r.soil_moisture,
            "slope_angle": r.slope_angle,
            "vibration_frequency": r.vibration_frequency,
            "pore_water_pressure": r.pore_water_pressure,
            "temperature": r.temperature,
            "risk_score": pred.risk_score if pred else 0.0,
            "risk_level": pred.risk_level if pred else "Safe"
        })

    return chart_data


@router.post("/{station_id}/thresholds")
def update_station_thresholds(
    station_id: int,
    payload: StationThresholdUpdate,
    db: Session = Depends(get_db)
):
    station = db.query(Station).filter(Station.id == station_id).first()
    if not station:
        raise HTTPException(status_code=404, detail="Station not found")

    if payload.warning_threshold >= payload.critical_threshold:
        raise HTTPException(
            status_code=400,
            detail="Warning threshold must be strictly less than critical threshold."
        )

    station.warning_threshold = payload.warning_threshold
    station.critical_threshold = payload.critical_threshold
    db.commit()

    return {
        "status": "success",
        "station_id": station.id,
        "warning_threshold": station.warning_threshold,
        "critical_threshold": station.critical_threshold
    }
