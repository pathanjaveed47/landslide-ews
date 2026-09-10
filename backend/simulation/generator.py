import asyncio
import datetime
import logging
import random
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from ..database.session import SessionLocal
from ..database.models import Station, SensorReading, Prediction
from ..ml.predictor import LandslidePredictor
from ..app.alerts import evaluate_station_risk
from ..app.websocket_manager import ws_manager

logger = logging.getLogger("landslide_sentinel.simulation")

class SensorSimulator:
    def __init__(self, interval_seconds: float = 2.5):
        self.interval_seconds = interval_seconds
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.tick_counter = 0
        self.station_states: Dict[int, Dict[str, float]] = {}
        self.active_scenarios: Dict[int, Dict[str, Any]] = {}
        self.predictor = LandslidePredictor.get_instance()

    def _init_station_states(self, db: Session):
        stations = db.query(Station).filter(Station.is_active == True).all()
        for s in stations:
            # Query the latest reading if available
            latest_reading = (
                db.query(SensorReading)
                .filter(SensorReading.station_id == s.id)
                .order_by(SensorReading.timestamp.desc())
                .first()
            )
            if latest_reading:
                self.station_states[s.id] = {
                    "rainfall_rate": latest_reading.rainfall_rate,
                    "cumulative_rainfall_24h": latest_reading.cumulative_rainfall_24h,
                    "soil_moisture": latest_reading.soil_moisture,
                    "slope_angle": latest_reading.slope_angle,
                    "vibration_frequency": latest_reading.vibration_frequency,
                    "pore_water_pressure": latest_reading.pore_water_pressure,
                    "temperature": latest_reading.temperature,
                    "base_slope": s.base_slope
                }
            else:
                self.station_states[s.id] = {
                    "rainfall_rate": 8.0,
                    "cumulative_rainfall_24h": 35.0,
                    "soil_moisture": 50.0,
                    "slope_angle": s.base_slope,
                    "vibration_frequency": 0.35,
                    "pore_water_pressure": 22.0,
                    "temperature": 18.0,
                    "base_slope": s.base_slope
                }

    def trigger_scenario(self, station_id: Optional[int], scenario_type: str, duration_seconds: int = 180, intensity: float = 1.0):
        """
        Injects a scenario into a single station or all stations.
        scenario_type: 'normal', 'cloudburst', 'pore_surge', 'seismic', 'escalating_demo'
        """
        db = SessionLocal()
        try:
            target_ids = [station_id] if station_id is not None else [s.id for s in db.query(Station).all()]
            for s_id in target_ids:
                if scenario_type == "normal":
                    if s_id in self.active_scenarios:
                        del self.active_scenarios[s_id]
                    # Reset state towards nominal
                    if s_id in self.station_states:
                        state = self.station_states[s_id]
                        state["rainfall_rate"] = 6.0
                        state["soil_moisture"] = 48.0
                        state["pore_water_pressure"] = 20.0
                        state["vibration_frequency"] = 0.3
                else:
                    self.active_scenarios[s_id] = {
                        "type": scenario_type,
                        "elapsed": 0,
                        "duration": duration_seconds,
                        "intensity": intensity
                    }
            logger.info(f"Triggered scenario '{scenario_type}' on stations: {target_ids}")
        finally:
            db.close()

    def _apply_step(self, station_id: int, s_info: Station) -> Dict[str, float]:
        state = self.station_states[station_id]
        base_slope = state["base_slope"]
        scenario = self.active_scenarios.get(station_id)

        if scenario:
            scenario["elapsed"] += self.interval_seconds
            s_type = scenario["type"]
            elapsed = scenario["elapsed"]
            intensity = scenario.get("intensity", 1.0)

            if s_type == "cloudburst":
                # Torrential downpour
                state["rainfall_rate"] = min(125.0, state["rainfall_rate"] + 8.0 * intensity + random.gauss(0, 1.5))
                state["cumulative_rainfall_24h"] += (state["rainfall_rate"] * (self.interval_seconds / 3600.0))
                state["soil_moisture"] = min(99.0, state["soil_moisture"] + 1.2 * intensity)
                state["pore_water_pressure"] = min(85.0, state["pore_water_pressure"] + 1.5 * intensity)
                state["vibration_frequency"] = max(0.1, 0.4 + random.gauss(0, 0.1))

            elif s_type == "pore_surge":
                # Subsurface water entrapment
                state["pore_water_pressure"] = min(90.0, state["pore_water_pressure"] + 2.2 * intensity)
                state["soil_moisture"] = min(95.0, state["soil_moisture"] + 0.8 * intensity)
                state["rainfall_rate"] = max(5.0, min(35.0, state["rainfall_rate"] + random.gauss(0, 0.5)))
                state["cumulative_rainfall_24h"] += (state["rainfall_rate"] * (self.interval_seconds / 3600.0))

            elif s_type == "seismic":
                # Micro-tremor seismic excitation
                state["vibration_frequency"] = min(5.0, max(1.8, 2.8 * intensity + random.gauss(0, 0.4)))
                state["slope_angle"] = base_slope + random.uniform(0.5, 1.8) * intensity

            elif s_type == "escalating_demo":
                # Programmed disaster demonstration cycle: Safe -> Watch -> Warning -> Critical
                if elapsed < 25:
                    # Phase 1: Safe baseline
                    target_rain = 6.0
                    target_moist = 44.0
                    target_pore = 18.0
                    target_vibe = 0.3
                elif elapsed < 65:
                    # Phase 2: Moderate worsening -> Watch
                    target_rain = 34.0
                    target_moist = 64.0
                    target_pore = 32.0
                    target_vibe = 0.6
                elif elapsed < 115:
                    # Phase 3: Heavy rain & saturation -> Warning
                    target_rain = 68.0
                    target_moist = 81.0
                    target_pore = 49.0
                    target_vibe = 1.2
                elif elapsed < 175:
                    # Phase 4: Extreme cloudburst & tremors -> Critical
                    target_rain = 108.0
                    target_moist = 96.0
                    target_pore = 72.0
                    target_vibe = 2.9
                else:
                    # Phase 5: Calming down
                    target_rain = 12.0
                    target_moist = 55.0
                    target_pore = 26.0
                    target_vibe = 0.4

                # Smoothly interpolate towards targets
                state["rainfall_rate"] += (target_rain - state["rainfall_rate"]) * 0.15 + random.gauss(0, 0.5)
                state["soil_moisture"] += (target_moist - state["soil_moisture"]) * 0.12 + random.gauss(0, 0.3)
                state["pore_water_pressure"] += (target_pore - state["pore_water_pressure"]) * 0.12 + random.gauss(0, 0.4)
                state["vibration_frequency"] += (target_vibe - state["vibration_frequency"]) * 0.15 + random.gauss(0, 0.05)
                state["cumulative_rainfall_24h"] += (max(0.0, state["rainfall_rate"]) * (self.interval_seconds / 3600.0))

            # Auto-expire scenario when duration reached
            if elapsed >= scenario.get("duration", 180):
                logger.info(f"Scenario {s_type} ended on station {station_id}")
                del self.active_scenarios[station_id]

        else:
            # Normal environmental random walk with mean-reversion
            rain_target = 8.0
            moist_target = 52.0
            pore_target = 22.0
            vibe_target = 0.35

            state["rainfall_rate"] = max(0.0, state["rainfall_rate"] + (rain_target - state["rainfall_rate"]) * 0.04 + random.gauss(0, 0.6))
            state["cumulative_rainfall_24h"] += (state["rainfall_rate"] * (self.interval_seconds / 3600.0))
            state["soil_moisture"] = max(15.0, min(92.0, state["soil_moisture"] + (moist_target - state["soil_moisture"]) * 0.03 + random.gauss(0, 0.4)))
            state["pore_water_pressure"] = max(5.0, min(75.0, state["pore_water_pressure"] + (pore_target - state["pore_water_pressure"]) * 0.03 + random.gauss(0, 0.3)))
            state["vibration_frequency"] = max(0.05, min(3.0, state["vibration_frequency"] + (vibe_target - state["vibration_frequency"]) * 0.08 + random.gauss(0, 0.04)))
            state["slope_angle"] = max(10.0, min(65.0, base_slope + random.gauss(0, 0.08)))

        # Bounds safety clamping
        state["rainfall_rate"] = max(0.0, min(140.0, round(state["rainfall_rate"], 2)))
        state["cumulative_rainfall_24h"] = max(0.0, min(500.0, round(state["cumulative_rainfall_24h"], 2)))
        state["soil_moisture"] = max(5.0, min(100.0, round(state["soil_moisture"], 2)))
        state["pore_water_pressure"] = max(0.0, min(100.0, round(state["pore_water_pressure"], 2)))
        state["vibration_frequency"] = max(0.02, min(5.5, round(state["vibration_frequency"], 2)))
        state["slope_angle"] = max(10.0, min(70.0, round(state["slope_angle"], 2)))
        state["temperature"] = round(max(4.0, 24.0 - (s_info.elevation / 200.0) + random.gauss(0, 0.3)), 2)

        return state

    async def _simulation_loop(self):
        logger.info(f"Simulation loop started (Tick interval: {self.interval_seconds}s)")
        while self.is_running:
            try:
                await self._process_tick()
            except Exception as e:
                logger.error(f"Error in simulation tick: {e}", exc_info=True)

            await asyncio.sleep(self.interval_seconds)

    async def _process_tick(self):
        self.tick_counter += 1
        db = SessionLocal()
        try:
            if not self.station_states:
                self._init_station_states(db)

            stations = db.query(Station).filter(Station.is_active == True).all()
            now = datetime.datetime.now(datetime.timezone.utc)
            updates_batch = []

            for s in stations:
                if s.id not in self.station_states:
                    continue

                new_state = self._apply_step(s.id, s)

                # Persist reading
                reading = SensorReading(
                    station_id=s.id,
                    timestamp=now,
                    rainfall_rate=new_state["rainfall_rate"],
                    cumulative_rainfall_24h=new_state["cumulative_rainfall_24h"],
                    soil_moisture=new_state["soil_moisture"],
                    slope_angle=new_state["slope_angle"],
                    vibration_frequency=new_state["vibration_frequency"],
                    pore_water_pressure=new_state["pore_water_pressure"],
                    temperature=new_state["temperature"]
                )
                db.add(reading)
                db.flush()

                # ML Inference
                feat_dict = {
                    "rainfall_rate": reading.rainfall_rate,
                    "cumulative_rainfall_24h": reading.cumulative_rainfall_24h,
                    "soil_moisture": reading.soil_moisture,
                    "slope_angle": reading.slope_angle,
                    "vibration_frequency": reading.vibration_frequency,
                    "pore_water_pressure": reading.pore_water_pressure,
                    "temperature": reading.temperature
                }
                pred = self.predictor.predict(feat_dict)

                prediction = Prediction(
                    reading_id=reading.id,
                    station_id=s.id,
                    timestamp=now,
                    risk_score=pred["risk_score"],
                    risk_level=pred["risk_level"],
                    confidence=pred["confidence"],
                    contributing_factors=pred["contributing_factors"]
                )
                db.add(prediction)
                db.flush()

                # Alert Evaluation
                alert, is_new_alert = evaluate_station_risk(s, reading, prediction, db)

                alert_dict = None
                if alert:
                    alert_dict = {
                        "id": alert.id,
                        "station_id": s.id,
                        "station_code": s.code,
                        "station_name": s.name,
                        "risk_level": alert.risk_level,
                        "risk_score": alert.risk_score,
                        "message": alert.message,
                        "suggested_action": alert.suggested_action,
                        "contributing_factors": alert.contributing_factors,
                        "status": alert.status,
                        "timestamp": alert.timestamp.isoformat(),
                        "is_new": is_new_alert
                    }

                updates_batch.append({
                    "station_id": s.id,
                    "station_code": s.code,
                    "station_name": s.name,
                    "region": s.region,
                    "latitude": s.latitude,
                    "longitude": s.longitude,
                    "elevation": s.elevation,
                    "reading": {
                        "id": reading.id,
                        "timestamp": now.isoformat(),
                        "rainfall_rate": reading.rainfall_rate,
                        "cumulative_rainfall_24h": reading.cumulative_rainfall_24h,
                        "soil_moisture": reading.soil_moisture,
                        "slope_angle": reading.slope_angle,
                        "vibration_frequency": reading.vibration_frequency,
                        "pore_water_pressure": reading.pore_water_pressure,
                        "temperature": reading.temperature
                    },
                    "prediction": {
                        "id": prediction.id,
                        "risk_score": prediction.risk_score,
                        "risk_level": prediction.risk_level,
                        "confidence": prediction.confidence,
                        "class_probabilities": pred.get("class_probabilities", {}),
                        "contributing_factors": prediction.contributing_factors
                    },
                    "alert": alert_dict
                })

            db.commit()

            # Broadcast batch telemetry update over WebSocket
            if updates_batch:
                active_scenario_summary = {
                    str(s_id): {
                        "type": info["type"],
                        "elapsed": round(info["elapsed"], 1),
                        "duration": info["duration"]
                    }
                    for s_id, info in self.active_scenarios.items()
                }

                await ws_manager.broadcast({
                    "type": "telemetry_batch",
                    "tick": self.tick_counter,
                    "timestamp": now.isoformat(),
                    "stations": updates_batch,
                    "active_scenarios": active_scenario_summary
                })

        except Exception as e:
            db.rollback()
            logger.error(f"Failed in _process_tick: {e}", exc_info=True)
        finally:
            db.close()

    def start(self):
        if not self.is_running:
            self.is_running = True
            self._task = asyncio.create_task(self._simulation_loop())
            logger.info("Sensor simulator background task scheduled.")

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self._task:
                self._task.cancel()
            logger.info("Sensor simulator background task stopped.")

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "interval_seconds": self.interval_seconds,
            "tick_counter": self.tick_counter,
            "active_scenarios": {
                str(s_id): {
                    "type": info["type"],
                    "elapsed": round(info["elapsed"], 1),
                    "duration": info["duration"]
                }
                for s_id, info in self.active_scenarios.items()
            }
        }

# Global singleton simulator
simulator = SensorSimulator(interval_seconds=2.5)
