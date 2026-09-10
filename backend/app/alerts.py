import datetime
import logging
from sqlalchemy.orm import Session
from ..database.models import Station, SensorReading, Prediction, AlertLog, NotificationLog

logger = logging.getLogger("landslide_sentinel.alerts")

# ============================================================================
# MOCK NOTIFICATION DISPATCHERS (Ready to swap for Twilio & SendGrid)
# ============================================================================

def send_mock_sms(to_phone: str, message: str, alert_id: int, db: Session) -> NotificationLog:
    """
    Simulates dispatch of an emergency SMS broadcast.
    To swap for Twilio:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(body=message, from_=TWILIO_NUMBER, to=to_phone)
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    print("\n" + "="*70)
    print(f"📱 [MOCK SMS DISPATCHED] -> {to_phone}")
    print(f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Message: {message}")
    print("="*70 + "\n")

    notif = NotificationLog(
        alert_id=alert_id,
        channel="SMS",
        recipient=to_phone,
        message=message,
        status="DELIVERED",
        timestamp=timestamp
    )
    db.add(notif)
    return notif


def send_mock_email(to_email: str, subject: str, body: str, alert_id: int, db: Session) -> NotificationLog:
    """
    Simulates dispatch of an emergency email dispatch.
    To swap for SendGrid:
        import sendgrid
        from sendgrid.helpers.mail import Mail
        sg = sendgrid.SendGridAPIClient(api_key=SENDGRID_API_KEY)
        mail = Mail(from_email='alerts@sentinel.org', to_emails=to_email, subject=subject, html_content=body)
        sg.send(mail)
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    print("\n" + "#"*70)
    print(f"📧 [MOCK EMAIL DISPATCHED] -> {to_email}")
    print(f"Subject: {subject}")
    print(f"Body: {body[:150]}...")
    print("#"*70 + "\n")

    notif = NotificationLog(
        alert_id=alert_id,
        channel="EMAIL",
        recipient=to_email,
        message=f"{subject}\n\n{body}",
        status="DELIVERED",
        timestamp=timestamp
    )
    db.add(notif)
    return notif


# ============================================================================
# ALERT EVALUATION & ESCALATION ENGINE
# ============================================================================

def determine_suggested_action(risk_level: str, factors: list[str], station: Station) -> str:
    if risk_level == "Critical":
        return (
            f"EVACUATION PROTOCOL LEVEL 3: Immediate evacuation of downslope communities in {station.region}. "
            f"Close arterial roadways. Dispatch NDRF/SDRF rescue teams immediately."
        )
    elif risk_level == "Warning":
        return (
            f"ADVISORY LEVEL 2: Restrict vehicular transit along {station.name} slopes. "
            f"Activate local emergency response shelters and inspect drainage culverts."
        )
    else:
        return f"MONITORING LEVEL 1: Maintain heightened telemetry surveillance across {station.name} sector."


def evaluate_station_risk(
    station: Station,
    reading: SensorReading,
    prediction: Prediction,
    db: Session
) -> tuple[AlertLog | None, bool]:
    """
    Evaluates whether an alert should be triggered or escalated.
    Returns (alert_instance, is_newly_triggered).
    """
    risk_score = prediction.risk_score
    risk_level = prediction.risk_level

    # Check against station-specific configured thresholds
    is_critical = risk_score >= station.critical_threshold or risk_level == "Critical"
    is_warning = (risk_score >= station.warning_threshold or risk_level == "Warning") and not is_critical

    if not (is_critical or is_warning):
        return None, False

    target_level = "Critical" if is_critical else "Warning"

    # Query latest active alert for this station
    latest_active = (
        db.query(AlertLog)
        .filter(AlertLog.station_id == station.id, AlertLog.status.in_(["ACTIVE", "ACKNOWLEDGED"]))
        .order_by(AlertLog.timestamp.desc())
        .first()
    )

    now = datetime.datetime.now(datetime.timezone.utc)

    # Condition 1: No active alert -> Trigger brand new alert
    # Condition 2: Active alert exists at Warning, but now elevated to Critical -> Trigger Escalated alert
    should_trigger_new = False
    if not latest_active:
        should_trigger_new = True
    elif latest_active.risk_level == "Warning" and target_level == "Critical":
        # Escalate previous warning to acknowledged/superseded and fire new critical alert
        latest_active.status = "ACKNOWLEDGED"
        should_trigger_new = True
    else:
        # Update existing active alert's current score and factors without spamming notifications
        latest_active.risk_score = risk_score
        latest_active.contributing_factors = prediction.contributing_factors
        db.flush()
        return latest_active, False

    if should_trigger_new:
        suggested_action = determine_suggested_action(target_level, prediction.contributing_factors, station)
        message = f"[{target_level.upper()}] Landslide Risk Threshold Exceeded at {station.name} ({station.code})"

        alert = AlertLog(
            station_id=station.id,
            timestamp=now,
            risk_level=target_level,
            risk_score=risk_score,
            message=message,
            contributing_factors=prediction.contributing_factors,
            suggested_action=suggested_action,
            status="ACTIVE"
        )
        db.add(alert)
        db.flush()

        # Dispatch automated mock notifications
        sms_recipients = [
            "+91-9876543210 (SDMA Disaster Control)",
            "+91-9447123456 (Emergency Ops Chief)"
        ]
        for phone in sms_recipients:
            sms_body = (
                f"[SENTINEL ALERT - {target_level.upper()}]\n"
                f"Station: {station.name} ({station.region})\n"
                f"Risk Score: {risk_score:.1f}/100\n"
                f"Action: {suggested_action[:100]}..."
            )
            send_mock_sms(phone, sms_body, alert.id, db)

        email_recipients = [
            "disaster.response@sdma.gov.in",
            "district.magistrate@hazardcontrol.org"
        ]
        email_body = (
            f"LandSlide Sentinel Early Warning Automated Dispatch\n"
            f"Station: {station.name} [{station.code}]\n"
            f"Region: {station.region}\n"
            f"Coordinates: {station.latitude:.4f}, {station.longitude:.4f}\n"
            f"Elevation: {station.elevation}m | Slope: {station.base_slope}°\n"
            f"Risk Level: {target_level}\n"
            f"Risk Score: {risk_score:.1f} / 100\n"
            f"Confidence: {prediction.confidence * 100:.1f}%\n\n"
            f"PRIMARY CONTRIBUTING FACTORS:\n" +
            "\n".join([f" • {f}" for f in prediction.contributing_factors]) +
            f"\n\nREQUIRED IMMEDIATE ACTION:\n{suggested_action}\n"
        )
        for email in email_recipients:
            send_mock_email(email, f"URGENT: {target_level} Landslide Threat - {station.name}", email_body, alert.id, db)

        db.commit()
        return alert, True

    return None, False
