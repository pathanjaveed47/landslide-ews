from fastapi import APIRouter, HTTPException
from ..schemas import ScenarioTriggerRequest, SimulationStatusResponse
from ...simulation.generator import simulator

router = APIRouter(prefix="/simulation", tags=["Simulation"])

@router.post("/scenario")
def trigger_scenario(payload: ScenarioTriggerRequest):
    valid_scenarios = ["normal", "cloudburst", "pore_surge", "seismic", "escalating_demo"]
    if payload.scenario_type not in valid_scenarios:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scenario type '{payload.scenario_type}'. Must be one of: {valid_scenarios}"
        )

    simulator.trigger_scenario(
        station_id=payload.station_id,
        scenario_type=payload.scenario_type,
        duration_seconds=payload.duration_seconds or 180,
        intensity=payload.intensity or 1.0
    )

    return {
        "status": "success",
        "message": f"Scenario '{payload.scenario_type}' injected successfully.",
        "station_id": payload.station_id or "ALL",
        "duration_seconds": payload.duration_seconds or 180
    }


@router.get("/status")
def get_simulation_status():
    return simulator.get_status()


@router.post("/start")
def start_simulation():
    simulator.start()
    return {"status": "started", "message": "Simulation is actively running."}


@router.post("/stop")
def stop_simulation():
    simulator.stop()
    return {"status": "stopped", "message": "Simulation is paused."}
