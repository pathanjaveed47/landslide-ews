import os
import json
import joblib
import numpy as np

ML_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ML_DIR, "landslide_model.joblib")
METRICS_PATH = os.path.join(ML_DIR, "metrics.json")

class LandslidePredictor:
    _instance = None
    _bundle = None

    def __init__(self):
        self.load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LandslidePredictor()
        return cls._instance

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            # Train if model does not exist yet
            from .train import train_and_save_models
            print("Model file not found. Auto-training initial model...")
            train_and_save_models(save_dir=ML_DIR)
        
        self._bundle = joblib.load(MODEL_PATH)
        self.clf = self._bundle["classifier"]
        self.reg = self._bundle["regressor"]
        self.scaler = self._bundle["scaler"]
        self.feature_columns = self._bundle["feature_columns"]
        self.label_map = self._bundle["label_map"]
        self.metrics = self._bundle.get("metrics", {})

    def reload(self):
        """Reloads the model from disk (useful after retraining)."""
        self.load_model()

    def identify_contributing_factors(self, data: dict) -> list[str]:
        factors = []
        
        rainfall = data.get("rainfall_rate", 0.0)
        cum_rain = data.get("cumulative_rainfall_24h", 0.0)
        moisture = data.get("soil_moisture", 0.0)
        slope = data.get("slope_angle", 0.0)
        pore_pressure = data.get("pore_water_pressure", 0.0)
        vibration = data.get("vibration_frequency", 0.0)
        
        if rainfall > 45.0:
            factors.append(f"Intense precipitation: {rainfall:.1f} mm/hr (Extreme)")
        elif rainfall > 25.0:
            factors.append(f"Elevated rainfall rate: {rainfall:.1f} mm/hr")

        if cum_rain > 160.0:
            factors.append(f"High 24h antecedent rainfall: {cum_rain:.1f} mm (Deep saturation)")
        elif cum_rain > 90.0:
            factors.append(f"Significant cumulative rainfall: {cum_rain:.1f} mm")

        if pore_pressure > 50.0:
            factors.append(f"Severe pore water pressure: {pore_pressure:.1f} kPa (Hydraulic uplift hazard)")
        elif pore_pressure > 32.0:
            factors.append(f"Pore water pressure above baseline: {pore_pressure:.1f} kPa")

        if moisture > 85.0:
            factors.append(f"Near-saturated regolith: soil moisture {moisture:.1f}%")
        elif moisture > 70.0:
            factors.append(f"Elevated soil moisture content: {moisture:.1f}%")

        if slope > 36.0:
            factors.append(f"Steep topographical slope: {slope:.1f}°")

        if vibration > 2.0:
            factors.append(f"Severe ground vibration / tremor detected: {vibration:.2f} Hz")
        elif vibration > 1.0:
            factors.append(f"Elevated micro-seismic activity: {vibration:.2f} Hz")

        if not factors:
            factors.append("Geotechnical parameters within normal nominal bounds.")
            
        return factors

    def predict(self, features: dict) -> dict:
        """
        Takes dictionary of features and returns prediction summary:
        risk_score (0-100), risk_level, confidence, class_probabilities, and contributing_factors.
        """
        raw_vals = [float(features.get(col, 0.0)) for col in self.feature_columns]
        X = np.array([raw_vals])
        X_scaled = self.scaler.transform(X)

        # Regressor continuous score (clamped between 0 and 100)
        raw_score = float(self.reg.predict(X_scaled)[0])
        risk_score = round(max(0.0, min(100.0, raw_score)), 1)

        # Classifier predicted class and probabilities
        class_idx = int(self.clf.predict(X_scaled)[0])
        probs = self.clf.predict_proba(X_scaled)[0]
        confidence = round(float(np.max(probs)), 3)

        # Harmonize continuous score with categorical classification
        if risk_score >= 80.0:
            risk_level = "Critical"
        elif risk_score >= 60.0:
            risk_level = "Warning"
        elif risk_score >= 30.0:
            risk_level = "Watch"
        else:
            risk_level = "Safe"

        class_probabilities = {
            self.label_map[i]: round(float(p), 3) for i, p in enumerate(probs)
        }

        factors = self.identify_contributing_factors(features)

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "contributing_factors": factors
        }

    def get_metrics(self) -> dict:
        if not self.metrics and os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                self.metrics = json.load(f)
        return self.metrics
