# 🏔️ LandSlide Sentinel — AI-Based Early Warning & Landslide Risk Monitoring System

**LandSlide Sentinel** is a complete, end-to-end full-stack web application for real-time landslide risk assessment and early warning. It simulates multi-parametric IoT telemetry streams across mountain slopes, feeds them into a calibrated scikit-learn machine learning model, visualizes live risk on interactive GIS maps and multi-series telemetry charts, and triggers tiered alerts (Safe, Watch, Warning, Critical) with simulated SMS/Email dispatch and sirens.

---

## 📸 System Capabilities & Features

1. **Simulated Real-Time IoT Telemetry Engine**:
   - 6 geo-tagged monitoring stations in landslide-prone Indian mountain sectors (Himalayas & Western Ghats: Wayanad, Joshimath, Shimla, Darjeeling, Munnar, Rishikesh-Badrinath).
   - Real-time time series generated every 2.5 seconds:
     - **Rainfall Rate** (mm/hr)
     - **24-Hour Cumulative Rainfall** (mm)
     - **Soil Moisture Content** (%)
     - **Slope Angle Creep** (degrees °)
     - **Ground Vibration & Micro-Tremors** (Hz)
     - **Pore Water Pressure** (kPa)
     - **Ambient Temperature** (°C)
   - Dynamic **Scenario Injection**: *Torrential Cloudburst*, *Pore Water Surge*, *Micro-Seismic Tremors*, and a **Programmed 3-Minute Escalating Disaster Demo** (Safe → Watch → Warning → Critical).

2. **Geotechnical Machine Learning Risk Model**:
   - **Physics-Informed Training**: Trained on 15,000 synthetic records coupling hydrological driving forces, limit equilibrium shear stress, and pore pressure destabilization.
   - **Dual Model Architecture**:
     - **Random Forest Classifier**: Categorical tier classification (`Safe`, `Watch`, `Warning`, `Critical`) with 95.8% test accuracy.
     - **Gradient Boosting Regressor**: Continuous Risk Index (`0 - 100`) with \(R^2 = 0.9917\) and \(RMSE = 2.87\).
   - **Explainability**: Identifies primary contributing hazard factors in real time (e.g. *"Near-saturated regolith: soil moisture 94.2%"*, *"Severe pore water pressure: 68.5 kPa (Hydraulic uplift hazard)"*).
   - Saved and loaded via `joblib` with `/api/model/predict` and `/api/model/train` endpoints.

3. **Live Mission Control Dashboard (React + Vite + Tailwind CSS)**:
   - **Overview Dashboard**: High-level regional KPIs, peak hazard points, network health, and color-coded station cards with radial risk gauges and sensor grids.
   - **Interactive GIS Map (Leaflet.js)**: Topographic dark-matter mountain map with custom color-coded pins, pulsing radar circles on active warnings, and detail popups.
   - **Telemetry Charts (Recharts)**: Synchronized time series for Risk Index, Rainfall Dynamics, Subsurface Moisture & Pore Pressure, and Slope Stability.
   - **Tiered Incident Alert Feed**: Live incident alerts with contributing factors, emergency suggested actions, and "Acknowledge" / "Resolve" buttons.
   - **Mock SMS / Email Dispatch Inspector**: Logs automated outbound dispatches to disaster management authorities with clear hooks ready to swap for Twilio and SendGrid.
   - **Admin & Calibration Console**: One-click scenario injector, per-station threshold calibration sliders, ML feature importance ranking, and an interactive ad-hoc risk calculator.
   - **Audio Alert Synthesizer**: Web Audio API sirens for Critical alerts (can be muted via top navbar).

---

## 🏗️ Architecture & Project Structure

```
landslide-ews/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app, lifespan, CORS, static SPA serving
│   │   ├── config.py            # Global configuration settings
│   │   ├── schemas.py           # Pydantic validation schemas
│   │   ├── alerts.py            # Tiered alert logic & mock SMS/Email dispatcher
│   │   ├── websocket_manager.py # Duplex WebSocket connection manager
│   │   └── routes/
│   │       ├── stations.py      # Station metadata, telemetry history & threshold config
│   │       ├── alerts.py        # Active alert management and dispatch logs
│   │       ├── ml_routes.py     # Model inference, metrics & retraining trigger
│   │       └── simulation.py    # Scenario injection & simulator controls
│   ├── database/
│   │   ├── __init__.py
│   │   ├── session.py           # SQLite connection with WAL mode
│   │   ├── models.py            # SQLAlchemy models (Station, Reading, Prediction, Alert, Notification)
│   │   └── seed.py              # Initial stations and 2 hours of synthetic historical readings
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── dataset_generator.py # Physics-informed synthetic dataset generator
│   │   ├── train.py             # Model training & evaluation pipeline
│   │   ├── predictor.py         # Singleton inference engine & factor explainability
│   │   ├── landslide_model.joblib # Serialized model bundle
│   │   └── metrics.json         # Evaluated metrics snapshot
│   └── simulation/
│       ├── __init__.py
│       └── generator.py         # Async background time-series simulation engine
├── frontend/
│   ├── index.html               # SPA Entrypoint
│   ├── vite.config.js           # Vite dev server with proxy to backend
│   ├── tailwind.config.js       # Dark emergency response styling palette
│   └── src/
│       ├── App.jsx              # Main app state, tabs, and WebSocket sync
│       ├── main.jsx             # React DOM root
│       ├── index.css            # Tailwind + Leaflet styles
│       ├── services/
│       │   └── api.js           # REST API client & WebSocket handler
│       ├── utils/
│       │   └── audio.js         # Web Audio API emergency alert sound synthesizer
│       └── components/
│           ├── Navbar.jsx       # Top navigation, status indicator & quick demo launch
│           ├── AlertBanner.jsx  # Floating emergency alert banner for active hazards
│           ├── Dashboard.jsx    # Overview KPI cards, demo hero showcase, station grid
│           ├── StationCard.jsx  # Radial gauge, sensor pills & quick actions
│           ├── MapView.jsx      # Leaflet GIS map with animated threat markers
│           ├── SensorCharts.jsx # Recharts multi-series telemetry graphs
│           ├── AlertFeed.jsx    # Live alerts feed and simulated SMS/Email logs
│           └── AdminPanel.jsx   # Scenario injector, threshold sliders, ML metrics
├── test_e2e.py                  # Automated end-to-end verification test suite
└── README.md
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.13)
- **Node.js 18+** & **npm**

### 2. Backend Setup
Navigate to `landslide-ews` and install the Python dependencies:

```bash
cd c:\SIH\landslide-ews
pip install fastapi uvicorn sqlalchemy websockets scikit-learn numpy pandas joblib pydantic
```

*(Optional) Retrain the machine learning model from scratch:*
```bash
python backend/ml/train.py
```

*(Optional) Seed the database with initial stations and telemetry history:*
```bash
python -m backend.database.seed
```
*(Note: If the database does not exist, FastAPI will automatically seed it upon first startup).*

### 3. Frontend Setup
Navigate to the `frontend` directory and install npm packages:

```bash
cd c:\SIH\landslide-ews\frontend
npm install
npm run build
```

---

## 💻 Running the Application

### Option A: Unified Single-Server Mode (Recommended)
Since the production frontend is built into `frontend/dist`, FastAPI will serve the full web application directly:

```bash
cd c:\SIH\landslide-ews
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
Open your browser at:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### Option B: Interactive Developer Mode (Hot-Reloading)
Run backend and frontend concurrently in two terminals:

**Terminal 1 (Backend API & WebSocket):**
```bash
cd c:\SIH\landslide-ews
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 (Frontend with Vite HMR):**
```bash
cd c:\SIH\landslide-ews\frontend
npm run dev
```
Open your browser at:
👉 **[http://localhost:5173](http://localhost:5173)** *(API and WebSockets are automatically proxied to port 8000)*.

---

## 🧪 Testing the Live Escalation Demo Scenario

To demonstrate the full capability of the system from nominal baseline to emergency evacuation:

1. Click the **"Demo Escalation"** button in the top navigation bar or the **"Launch Escalation Demo"** hero banner on the Overview page.
2. The simulation targets **Station STN-03 (Shimla Ridge Bypass)** and cycles through 4 phases over 3 minutes:
   - **Phase 1 (0–25s) — Safe**: Rainfall ~6 mm/hr, Moisture ~44%, Pore pressure ~18 kPa, Risk Score ~12/100 (Green).
   - **Phase 2 (25–65s) — Watch**: Rain increases to ~34 mm/hr, Moisture climbs to ~64%, Risk Score rises to ~45/100 (Amber).
   - **Phase 3 (65–115s) — Warning**: Torrential rain reaches ~68 mm/hr, Moisture hits ~81%, Pore pressure exceeds ~49 kPa. Risk Score crosses 65/100 (Orange).
     - Automated **Warning Alert** triggers.
     - Warning chime sounds.
     - Simulated SMS & Email alerts dispatched to disaster cell.
   - **Phase 4 (115–175s) — Critical**: Cloudburst peaks at ~108 mm/hr, Moisture saturates to ~96%, Pore pressure reaches ~72 kPa, Ground vibration reaches ~2.9 Hz. Risk Score crosses 85/100 (Red).
     - Automated **Critical Evacuation Alert** triggers.
     - High-urgency siren sounds.
     - Map pin for STN-03 pulses with red animated radar waves.
     - Emergency Level-3 Evacuation SMS broadcast is dispatched.
3. Switch to the **Alerts** tab to view the generated incident, read the contributing factor breakdown, inspect the simulated outbound SMS/Email logs, and click **"Acknowledge"** or **"Resolve"**.
4. Switch to the **Telemetry** tab to observe the live multi-series curves rising across thresholds in real time.

---

## ⚡ Automated End-to-End Test Suite

An automated end-to-end verification script is included to test all subsystems:

```bash
cd c:\SIH\landslide-ews
python test_e2e.py
```

This verifies:
1. API health and simulation status.
2. Stations registry and telemetry loading.
3. 60-point historical time-series retrieval.
4. ML model accuracy (>95%) and catastrophic risk prediction.
5. Dynamic threshold calibration.
6. Real-time WebSocket streaming.
7. Scenario injection & automated alert generation.
8. Mock SMS/Email emergency dispatch logging.

---

## 🔌 Swapping Mock Notifications for Production Twilio / SendGrid

In `backend/app/alerts.py`, the notification dispatchers are isolated and clearly commented:

### Twilio SMS:
```python
from twilio.rest import Client

def send_mock_sms(to_phone: str, message: str, alert_id: int, db: Session):
    client = Client(os.environ["TWILIO_ACCOUNT_SID"], os.environ["TWILIO_AUTH_TOKEN"])
    client.messages.create(body=message, from_=os.environ["TWILIO_NUMBER"], to=to_phone)
```

### SendGrid Email:
```python
import sendgrid
from sendgrid.helpers.mail import Mail

def send_mock_email(to_email: str, subject: str, body: str, alert_id: int, db: Session):
    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(from_email="alerts@sentinel.org", to_emails=to_email, subject=subject, html_content=body)
    sg.send(mail)
```
# New-AI-Based-early-warning-and-landslide-Risk-Monitoring-System
"# landslide-ews"  
