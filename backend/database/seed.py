import datetime
import random
from sqlalchemy.orm import Session
from .session import Base, engine, SessionLocal
from .models import Station, SensorReading, Prediction, AlertLog, NotificationLog
from ..ml.predictor import LandslidePredictor

SEED_STATIONS = [
    {
        "code": "STN-01",
        "name": "Meppadi Hill Slope",
        "region": "Western Ghats, Wayanad",
        "latitude": 11.5518,
        "longitude": 76.1265,
        "elevation": 940.0,
        "base_slope": 37.5,
        "soil_type": "Lateritic Colluvium",
        "warning_threshold": 55.0,
        "critical_threshold": 80.0,
        "base_rainfall": 12.0,
        "base_moisture": 62.0,
        "base_pore": 28.0,
        "base_vibe": 0.4
    },
    {
        "code": "STN-02",
        "name": "Joshimath Sector 4",
        "region": "Garhwal Himalayas, Chamoli",
        "latitude": 30.5562,
        "longitude": 79.5667,
        "elevation": 1890.0,
        "base_slope": 41.0,
        "soil_type": "Glacio-fluvial Moraine",
        "warning_threshold": 55.0,
        "critical_threshold": 78.0,
        "base_rainfall": 6.0,
        "base_moisture": 52.0,
        "base_pore": 24.0,
        "base_vibe": 0.5
    },
    {
        "code": "STN-03",
        "name": "Shimla Ridge Bypass",
        "region": "Himachal Shivalik Hills",
        "latitude": 31.1048,
        "longitude": 77.1734,
        "elevation": 2205.0,
        "base_slope": 34.0,
        "soil_type": "Fissured Schist & Silt",
        "warning_threshold": 55.0,
        "critical_threshold": 80.0,
        "base_rainfall": 4.0,
        "base_moisture": 45.0,
        "base_pore": 18.0,
        "base_vibe": 0.3
    },
    {
        "code": "STN-04",
        "name": "Darjeeling North Spur",
        "region": "Eastern Himalayas, West Bengal",
        "latitude": 27.0410,
        "longitude": 88.2663,
        "elevation": 2042.0,
        "base_slope": 39.0,
        "soil_type": "Weathered Phyllite & Clay",
        "warning_threshold": 55.0,
        "critical_threshold": 82.0,
        "base_rainfall": 24.0,
        "base_moisture": 76.0,
        "base_pore": 42.0,
        "base_vibe": 0.7
    },
    {
        "code": "STN-05",
        "name": "Munnar Gap Road Pass",
        "region": "Idukki Western Ghats",
        "latitude": 10.0889,
        "longitude": 77.0595,
        "elevation": 1530.0,
        "base_slope": 36.0,
        "soil_type": "Bouldery Regolith & Kaolin",
        "warning_threshold": 55.0,
        "critical_threshold": 80.0,
        "base_rainfall": 8.0,
        "base_moisture": 50.0,
        "base_pore": 21.0,
        "base_vibe": 0.35
    },
    {
        "code": "STN-06",
        "name": "Rishikesh-Badrinath NH Cut",
        "region": "Uttarakhand Alaknanda Valley",
        "latitude": 30.2854,
        "longitude": 78.9812,
        "elevation": 820.0,
        "base_slope": 44.5,
        "soil_type": "Quartzite Scree & Silt",
        "warning_threshold": 50.0,
        "critical_threshold": 75.0,
        "base_rainfall": 15.0,
        "base_moisture": 68.0,
        "base_pore": 34.0,
        "base_vibe": 0.8
    }
]

def seed_database(db: Session = None):
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        existing_count = db.query(Station).count()
        if existing_count > 0:
            print(f"Database already seeded with {existing_count} stations.")
            return

        print("Seeding stations, sensor telemetry, and ML predictions...")
        predictor = LandslidePredictor.get_instance()
        now = datetime.datetime.utcnow()

        created_stations = []
        for s_data in SEED_STATIONS:
            station = Station(
                code=s_data["code"],
                name=s_data["name"],
                region=s_data["region"],
                latitude=s_data["latitude"],
                longitude=s_data["longitude"],
                elevation=s_data["elevation"],
                base_slope=s_data["base_slope"],
                soil_type=s_data["soil_type"],
                warning_threshold=s_data["warning_threshold"],
                critical_threshold=s_data["critical_threshold"],
                is_active=True,
                created_at=now - datetime.timedelta(hours=4)
            )
            db.add(station)
            created_stations.append((station, s_data))
        
        db.flush()  # assign IDs

        # Generate 45 historical reading intervals (past 90 minutes, every 2 minutes)
        for station, s_data in created_stations:
            base_rain = s_data["base_rainfall"]
            base_moist = s_data["base_moisture"]
            base_pore = s_data["base_pore"]
            base_vibe = s_data["base_vibe"]
            slope = s_data["base_slope"]

            cum_rain = base_rain * 3.5

            for i in range(45, 0, -1):
                reading_time = now - datetime.timedelta(minutes=i * 2)
                
                # Introduce slight time-series drift and noise
                drift = (45 - i) * 0.1
                rain = max(0.0, base_rain + drift * 0.5 + random.gauss(0, 1.2))
                cum_rain += (rain * (2 / 60))  # 2 minute accumulation
                moisture = max(10.0, min(99.0, base_moist + drift * 0.3 + random.gauss(0, 0.8)))
                pore = max(0.0, base_pore + drift * 0.25 + random.gauss(0, 0.7))
                vibe = max(0.05, base_vibe + random.gauss(0, 0.05))
                inst_slope = max(15.0, slope + random.gauss(0, 0.15))
                temp = max(5.0, 22.0 - (station.elevation / 200.0) + random.gauss(0, 0.4))

                reading = SensorReading(
                    station_id=station.id,
                    timestamp=reading_time,
                    rainfall_rate=round(rain, 2),
                    cumulative_rainfall_24h=round(cum_rain, 2),
                    soil_moisture=round(moisture, 2),
                    slope_angle=round(inst_slope, 2),
                    vibration_frequency=round(vibe, 2),
                    pore_water_pressure=round(pore, 2),
                    temperature=round(temp, 2)
                )
                db.add(reading)
                db.flush()

                # Predict risk
                feat_dict = {
                    "rainfall_rate": reading.rainfall_rate,
                    "cumulative_rainfall_24h": reading.cumulative_rainfall_24h,
                    "soil_moisture": reading.soil_moisture,
                    "slope_angle": reading.slope_angle,
                    "vibration_frequency": reading.vibration_frequency,
                    "pore_water_pressure": reading.pore_water_pressure,
                    "temperature": reading.temperature
                }
                pred_result = predictor.predict(feat_dict)

                prediction = Prediction(
                    reading_id=reading.id,
                    station_id=station.id,
                    timestamp=reading_time,
                    risk_score=pred_result["risk_score"],
                    risk_level=pred_result["risk_level"],
                    confidence=pred_result["confidence"],
                    contributing_factors=pred_result["contributing_factors"]
                )
                db.add(prediction)

                # If last reading or higher risk, seed alert for testing
                if i == 1 and pred_result["risk_level"] in ["Warning", "Critical"]:
                    alert = AlertLog(
                        station_id=station.id,
                        timestamp=reading_time,
                        risk_level=pred_result["risk_level"],
                        risk_score=pred_result["risk_score"],
                        message=f"{pred_result['risk_level'].upper()}: High landslide potential at {station.name}",
                        contributing_factors=pred_result["contributing_factors"],
                        suggested_action="Deploy emergency reconnaissance team and notify district disaster authority.",
                        status="ACTIVE"
                    )
                    db.add(alert)
                    db.flush()

                    notif = NotificationLog(
                        alert_id=alert.id,
                        channel="SMS",
                        recipient="+91-9876543210 (SDMA Disaster Cell)",
                        message=f"[LandSlide Sentinel] {alert.message} - Action: {alert.suggested_action}",
                        status="DELIVERED",
                        timestamp=reading_time
                    )
                    db.add(notif)

        # Let's also add one historical resolved alert for Darjeeling North Spur
        darjeeling = [s for s, _ in created_stations if s.code == "STN-04"][0]
        hist_alert = AlertLog(
            station_id=darjeeling.id,
            timestamp=now - datetime.timedelta(hours=3),
            risk_level="Warning",
            risk_score=68.5,
            message="WARNING: Rain-induced slope saturation at Darjeeling North Spur",
            contributing_factors=["Intense precipitation: 32.5 mm/hr", "Elevated soil moisture content: 77.2%"],
            suggested_action="Issue yellow advisory to NH-10 traffic control and slope residents.",
            status="RESOLVED",
            acknowledged_at=now - datetime.timedelta(hours=2, minutes=45),
            resolved_at=now - datetime.timedelta(hours=1, minutes=30)
        )
        db.add(hist_alert)
        db.flush()

        hist_notif = NotificationLog(
            alert_id=hist_alert.id,
            channel="EMAIL",
            recipient="district.collector@darjeeling.gov.in",
            message=f"[LandSlide Sentinel ALERT] {hist_alert.message}",
            status="DELIVERED",
            timestamp=now - datetime.timedelta(hours=3)
        )
        db.add(hist_notif)

        db.commit()
        print("Database seeded successfully with 6 stations and telemetry history.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
        raise e
    finally:
        if close_db:
            db.close()

if __name__ == "__main__":
    seed_database()
