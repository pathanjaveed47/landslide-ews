import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from ...database.session import get_db
from ...database.models import AlertLog, NotificationLog, Station
from ..schemas import AlertLogResponse, NotificationLogResponse, AlertStatusUpdate

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertLogResponse])
def get_alerts(
    status: Optional[str] = Query(default=None, description="Filter by status: ACTIVE, ACKNOWLEDGED, RESOLVED"),
    risk_level: Optional[str] = Query(default=None, description="Filter by risk_level: Warning, Critical"),
    station_id: Optional[int] = Query(default=None),
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    query = db.query(AlertLog)

    if status:
        query = query.filter(AlertLog.status == status.upper())
    if risk_level:
        query = query.filter(AlertLog.risk_level == risk_level.capitalize())
    if station_id:
        query = query.filter(AlertLog.station_id == station_id)

    alerts = query.order_by(AlertLog.timestamp.desc()).limit(limit).all()

    results = []
    for a in alerts:
        station = db.query(Station).filter(Station.id == a.station_id).first()
        results.append({
            "id": a.id,
            "station_id": a.station_id,
            "station_name": station.name if station else "Unknown Station",
            "station_code": station.code if station else "N/A",
            "timestamp": a.timestamp,
            "risk_level": a.risk_level,
            "risk_score": a.risk_score,
            "message": a.message,
            "contributing_factors": a.contributing_factors or [],
            "suggested_action": a.suggested_action,
            "status": a.status,
            "acknowledged_at": a.acknowledged_at,
            "resolved_at": a.resolved_at,
            "notifications": [
                {
                    "id": n.id,
                    "alert_id": n.alert_id,
                    "channel": n.channel,
                    "recipient": n.recipient,
                    "message": n.message,
                    "status": n.status,
                    "timestamp": n.timestamp
                }
                for n in a.notifications
            ]
        })

    return results


@router.post("/{alert_id}/acknowledge")
def acknowledge_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(AlertLog).filter(AlertLog.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "ACKNOWLEDGED"
    alert.acknowledged_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()

    return {
        "status": "success",
        "alert_id": alert.id,
        "new_status": alert.status,
        "acknowledged_at": alert.acknowledged_at.isoformat()
    }


@router.post("/{alert_id}/resolve")
def resolve_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = db.query(AlertLog).filter(AlertLog.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "RESOLVED"
    alert.resolved_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()

    return {
        "status": "success",
        "alert_id": alert.id,
        "new_status": alert.status,
        "resolved_at": alert.resolved_at.isoformat()
    }


@router.get("/notifications", response_model=List[NotificationLogResponse])
def get_notification_logs(
    limit: int = Query(default=50, le=100),
    db: Session = Depends(get_db)
):
    notifs = (
        db.query(NotificationLog)
        .order_by(NotificationLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return notifs
