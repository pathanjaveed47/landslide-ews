import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_generator import generate_synthetic_dataset
except ImportError:
    from dataset_generator import generate_synthetic_dataset

FEATURE_COLUMNS = [
    "rainfall_rate",
    "cumulative_rainfall_24h",
    "soil_moisture",
    "slope_angle",
    "vibration_frequency",
    "pore_water_pressure",
    "temperature"
]

LABEL_MAP = {0: "Safe", 1: "Watch", 2: "Warning", 3: "Critical"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def train_and_save_models(save_dir: str = None, num_samples: int = 15000):
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Generating {num_samples} synthetic geotechnical records...")
    df = generate_synthetic_dataset(num_samples=num_samples, random_seed=42)
    
    # Save a copy of training data CSV for reference
    data_csv_path = os.path.join(save_dir, "synthetic_training_data.csv")
    df.to_csv(data_csv_path, index=False)
    print(f"Saved dataset snapshot to {data_csv_path}")
    
    X = df[FEATURE_COLUMNS].values
    y_class = df["risk_class"].values
    y_score = df["risk_score"].values
    
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X, y_class, y_score, test_size=0.2, random_state=42, stratify=y_class
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest Classifier for tiered category prediction...")
    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=14,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_class_train)
    
    print("Training Gradient Boosting Regressor for continuous risk score (0-100)...")
    reg = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    reg.fit(X_train_scaled, y_score_train)
    
    # Evaluations
    y_class_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_class_test, y_class_pred)
    cm = confusion_matrix(y_class_test, y_class_pred).tolist()
    report = classification_report(y_class_test, y_class_pred, target_names=["Safe", "Watch", "Warning", "Critical"], output_dict=True)
    
    y_score_pred = reg.predict(X_test_scaled)
    r2 = r2_score(y_score_test, y_score_pred)
    rmse = float(np.sqrt(mean_squared_error(y_score_test, y_score_pred)))
    
    # Feature importances (normalized average of both models)
    clf_importances = clf.feature_importances_
    reg_importances = reg.feature_importances_
    combined_importances = 0.6 * clf_importances + 0.4 * reg_importances
    
    feature_importance_list = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in sorted(zip(FEATURE_COLUMNS, combined_importances), key=lambda x: x[1], reverse=True)
    ]
    
    metrics = {
        "accuracy": round(float(acc), 4),
        "r2_score": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "confusion_matrix": cm,
        "class_labels": ["Safe", "Watch", "Warning", "Critical"],
        "classification_report": report,
        "feature_importances": feature_importance_list,
        "total_training_samples": len(df),
        "test_samples": len(X_test)
    }
    
    # Serialize model bundle
    bundle = {
        "classifier": clf,
        "regressor": reg,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "label_map": LABEL_MAP,
        "metrics": metrics
    }
    
    model_path = os.path.join(save_dir, "landslide_model.joblib")
    joblib.dump(bundle, model_path, compress=3)
    print(f"Saved trained model bundle to {model_path}")
    
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")
    print(f"Model Accuracy: {acc * 100:.2f}%, Regressor R2: {r2:.4f}, RMSE: {rmse:.2f}")
    
    return bundle, metrics

if __name__ == "__main__":
    train_and_save_models()
