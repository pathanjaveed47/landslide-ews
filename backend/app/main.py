import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from ..database.session import Base, engine, SessionLocal
from ..database.seed import seed_database
from ..database.models import Station
from ..ml.predictor import LandslidePredictor
from ..simulation.generator import simulator
from .websocket_manager import ws_manager

from .routes.stations import router as stations_router
from .routes.alerts import router as alerts_router
from .routes.ml_routes import router as ml_router
from .routes.simulation import router as simulation_router
from .routes.maptiler import router as maptiler_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("landslide_sentinel")

FRONTEND_DIST = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend", "dist")
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Ensure DB tables and seed data
    logger.info("Initializing LandSlide Sentinel database and models...")
    Base.metadata.create_all(bind=engine)
    seed_database()

    # 2. Warm up ML Predictor
    logger.info("Warming up ML Predictor...")
    LandslidePredictor.get_instance()

    # 3. Start sensor telemetry simulation
    logger.info("Starting real-time sensor simulator background loop...")
    simulator.start()

    yield

    # Shutdown
    logger.info("Shutting down sensor simulator...")
    simulator.stop()


app = FastAPI(
    title="LandSlide Sentinel - AI Early Warning System API",
    description="Real-time multi-sensor telemetry simulation, ML slope failure risk assessment, automated alerts, and WebSockets.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(stations_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(ml_router, prefix="/api")
app.include_router(simulation_router, prefix="/api")
app.include_router(maptiler_router, prefix="/api")


@app.get("/api/health")
def api_health():
    return {
        "system": "LandSlide Sentinel API",
        "status": "operational",
        "version": "1.0.0",
        "simulation": simulator.get_status(),
        "endpoints": {
            "stations": "/api/stations",
            "alerts": "/api/alerts",
            "ml_metrics": "/api/model/metrics",
            "predict": "/api/model/predict",
            "simulation": "/api/simulation/status",
            "websocket": "/ws/telemetry"
        }
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        # Send an immediate greeting and system snapshot
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to LandSlide Sentinel real-time telemetry stream.",
            "status": simulator.get_status()
        })
        while True:
            # Keep receiving any client messages/pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WebSocket connection error: {e}")
        ws_manager.disconnect(websocket)


# Mount Static frontend build if present
if os.path.exists(FRONTEND_DIST):
    logger.info(f"Mounting compiled frontend from: {FRONTEND_DIST}")
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
else:
    @app.get("/")
    def root():
        return api_health()
