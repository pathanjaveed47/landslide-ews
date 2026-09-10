import numpy as np
import pandas as pd

def compute_physics_risk(
    rainfall_rate: np.ndarray,
    cumulative_rainfall_24h: np.ndarray,
    soil_moisture: np.ndarray,
    slope_angle: np.ndarray,
    vibration_frequency: np.ndarray,
    pore_water_pressure: np.ndarray,
    temperature: np.ndarray,
    noise_std: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes geotechnical risk score (0-100) based on limit equilibrium
    and hydrological triggers. Returns (risk_score, risk_class_index).
    """
    n = len(rainfall_rate)
    
    # 1. Slope vulnerability (steeper slopes > 30 deg rapidly increase shear stress)
    # Baseline stability angle of typical weathered mountain soils is ~28-32 deg
    slope_factor = np.clip((slope_angle - 20.0) / 35.0, 0.0, 1.3)
    
    # 2. Moisture & Pore pressure factor (destabilizes effective normal stress)
    # Soil moisture above 65% starts saturation; pore water pressure above 30 kPa is hazardous
    moisture_factor = np.clip((soil_moisture - 40.0) / 50.0, 0.0, 1.2)
    pore_factor = np.clip((pore_water_pressure - 15.0) / 45.0, 0.0, 1.4)
    hydraulic_factor = 0.45 * moisture_factor + 0.55 * pore_factor
    
    # 3. Precipitation trigger (instantaneous intensity + antecedent moisture saturation)
    # 30 mm/hr is torrential; 150mm cumulative saturates regolith
    rain_rate_factor = np.clip(rainfall_rate / 60.0, 0.0, 1.3)
    cum_rain_factor = np.clip(cumulative_rainfall_24h / 180.0, 0.0, 1.2)
    precip_factor = 0.5 * rain_rate_factor + 0.5 * cum_rain_factor
    
    # 4. Vibration / Seismic perturbation
    vibe_factor = np.clip((vibration_frequency - 0.5) / 2.5, 0.0, 1.2)
    
    # Core Geotechnical Hazard interaction term:
    # High hydraulic saturation ON a steep slope creates severe structural destabilization
    coupled_shear = slope_factor * (0.65 * hydraulic_factor + 0.35 * precip_factor)
    
    # Base risk formulation (0 to 100 scale)
    base_score = (
        coupled_shear * 52.0 +
        precip_factor * 20.0 +
        pore_factor * 16.0 +
        vibe_factor * 12.0
    )
    
    # Add realistic environmental variance
    noise = np.random.normal(0, noise_std, size=n)
    risk_score = np.clip(base_score + noise, 0.0, 100.0)
    
    # Discretize into 4 tiered categories:
    # 0: Safe (0-30), 1: Watch (30-60), 2: Warning (60-80), 3: Critical (80-100)
    risk_class = np.zeros(n, dtype=int)
    risk_class[risk_score >= 30.0] = 1
    risk_class[risk_score >= 60.0] = 2
    risk_class[risk_score >= 80.0] = 3
    
    return risk_score, risk_class


def generate_synthetic_dataset(num_samples: int = 15000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a rich, balanced multi-variate synthetic dataset modeling
    various environmental states from drought to catastrophic monsoonal failure.
    """
    np.random.seed(random_seed)
    
    # We sample across distinct atmospheric & geotechnical regimes
    # Regime 1: Dry / Calm baseline (40%)
    n_calm = int(num_samples * 0.40)
    rain_calm = np.random.exponential(scale=2.5, size=n_calm)
    cum_rain_calm = np.random.uniform(0.0, 25.0, size=n_calm)
    moist_calm = np.random.normal(35.0, 8.0, size=n_calm)
    slope_calm = np.random.normal(28.0, 7.0, size=n_calm)
    vibe_calm = np.random.uniform(0.1, 0.6, size=n_calm)
    pore_calm = np.random.normal(10.0, 4.0, size=n_calm)
    temp_calm = np.random.normal(24.0, 4.0, size=n_calm)
    
    # Regime 2: Moderate Rain & Watch conditions (30%)
    n_mod = int(num_samples * 0.30)
    rain_mod = np.random.gamma(shape=3.0, scale=8.0, size=n_mod)
    cum_rain_mod = np.random.uniform(30.0, 120.0, size=n_mod)
    moist_mod = np.random.normal(62.0, 9.0, size=n_mod)
    slope_mod = np.random.normal(33.0, 6.0, size=n_mod)
    vibe_mod = np.random.uniform(0.3, 1.2, size=n_mod)
    pore_mod = np.random.normal(28.0, 7.0, size=n_mod)
    temp_mod = np.random.normal(19.0, 3.0, size=n_mod)
    
    # Regime 3: Heavy Rain & High Warning conditions (18%)
    n_warn = int(num_samples * 0.18)
    rain_warn = np.random.gamma(shape=5.0, scale=10.0, size=n_warn)
    cum_rain_warn = np.random.uniform(110.0, 220.0, size=n_warn)
    moist_warn = np.random.normal(78.0, 6.0, size=n_warn)
    slope_warn = np.random.normal(38.0, 6.0, size=n_warn)
    vibe_warn = np.random.uniform(0.8, 2.0, size=n_warn)
    pore_warn = np.random.normal(45.0, 9.0, size=n_warn)
    temp_warn = np.random.normal(16.0, 3.0, size=n_warn)
    
    # Regime 4: Extreme Cloudburst / Seismic Destabilization / Critical (12%)
    n_crit = num_samples - (n_calm + n_mod + n_warn)
    rain_crit = np.random.uniform(50.0, 130.0, size=n_crit)
    cum_rain_crit = np.random.uniform(190.0, 380.0, size=n_crit)
    moist_crit = np.random.uniform(85.0, 99.0, size=n_crit)
    slope_crit = np.random.uniform(36.0, 58.0, size=n_crit)
    vibe_crit = np.random.uniform(1.8, 4.8, size=n_crit)
    pore_crit = np.random.uniform(52.0, 92.0, size=n_crit)
    temp_crit = np.random.normal(14.0, 3.0, size=n_crit)
    
    # Concatenate all regimes
    rainfall_rate = np.clip(np.concatenate([rain_calm, rain_mod, rain_warn, rain_crit]), 0.0, 150.0)
    cumulative_rainfall_24h = np.clip(np.concatenate([cum_rain_calm, cum_rain_mod, cum_rain_warn, cum_rain_crit]), 0.0, 450.0)
    soil_moisture = np.clip(np.concatenate([moist_calm, moist_mod, moist_warn, moist_crit]), 5.0, 100.0)
    slope_angle = np.clip(np.concatenate([slope_calm, slope_mod, slope_warn, slope_crit]), 10.0, 65.0)
    vibration_frequency = np.clip(np.concatenate([vibe_calm, vibe_mod, vibe_warn, vibe_crit]), 0.05, 5.5)
    pore_water_pressure = np.clip(np.concatenate([pore_calm, pore_mod, pore_warn, pore_crit]), 0.0, 105.0)
    temperature = np.clip(np.concatenate([temp_calm, temp_mod, temp_warn, temp_crit]), 2.0, 42.0)
    
    risk_score, risk_class = compute_physics_risk(
        rainfall_rate,
        cumulative_rainfall_24h,
        soil_moisture,
        slope_angle,
        vibration_frequency,
        pore_water_pressure,
        temperature
    )
    
    label_map = {0: "Safe", 1: "Watch", 2: "Warning", 3: "Critical"}
    risk_label = [label_map[c] for c in risk_class]
    
    df = pd.DataFrame({
        "rainfall_rate": np.round(rainfall_rate, 2),
        "cumulative_rainfall_24h": np.round(cumulative_rainfall_24h, 2),
        "soil_moisture": np.round(soil_moisture, 2),
        "slope_angle": np.round(slope_angle, 2),
        "vibration_frequency": np.round(vibration_frequency, 2),
        "pore_water_pressure": np.round(pore_water_pressure, 2),
        "temperature": np.round(temperature, 2),
        "risk_score": np.round(risk_score, 1),
        "risk_class": risk_class,
        "risk_label": risk_label
    })
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_synthetic_dataset(15000)
    print(f"Generated synthetic dataset with {len(df)} samples.")
    print("Class distribution:")
    print(df["risk_label"].value_counts())
    print(df.head())
