import json
import time
import urllib.request
import urllib.error
import asyncio
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/telemetry"

def api_get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def api_post(path, data):
    url = f"{BASE_URL}{path}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

def run_tests():
    print("=" * 60)
    print("LANDSLIDE SENTINEL - END-TO-END AUTOMATED VERIFICATION")
    print("=" * 60)

    # 1. Health check
    print("\n[TEST 1] System Health & Simulation Status...")
    health = api_get("/api/health")
    assert health["status"] == "operational", "API status not operational"
    assert health["simulation"]["is_running"] is True, "Simulator is not running"
    print("  -> Passed! Simulator is running.")

    # 2. Stations API
    print("\n[TEST 2] Stations Registry & Telemetry...")
    stations = api_get("/api/stations")
    assert len(stations) == 6, f"Expected 6 stations, got {len(stations)}"
    stn3 = [s for s in stations if s["id"] == 3][0]
    assert stn3["code"] == "STN-03", f"Station 3 code mismatch: {stn3['code']}"
    assert "current_reading" in stn3 and stn3["current_reading"] is not None
    assert "current_prediction" in stn3 and stn3["current_prediction"] is not None
    print(f"  -> Passed! 6 stations verified. STN-03 ({stn3['name']}) initial risk: {stn3['current_prediction']['risk_level']} ({stn3['current_prediction']['risk_score']}/100)")

    # 3. Readings time series
    print("\n[TEST 3] Historical Time-Series Telemetry...")
    readings = api_get("/api/stations/3/readings?limit=30")
    assert len(readings) > 0, "No readings found for station 3"
    assert "pore_water_pressure" in readings[0]
    assert "soil_moisture" in readings[0]
    print(f"  -> Passed! Retrieved {len(readings)} time-series data points.")

    # 4. ML Model Introspection & Inference
    print("\n[TEST 4] ML Model Metrics & Ad-hoc Risk Inference...")
    metrics = api_get("/api/model/metrics")
    assert metrics["accuracy"] >= 0.90, f"Accuracy too low: {metrics['accuracy']}"
    assert len(metrics["feature_importances"]) > 0
    print(f"  -> Passed! Classifier Accuracy: {metrics['accuracy']*100:.1f}%, R²: {metrics['r2_score']}")

    # Ad-hoc extreme disaster inference test
    catastrophic_reading = {
        "rainfall_rate": 115.0,
        "cumulative_rainfall_24h": 280.0,
        "soil_moisture": 96.0,
        "slope_angle": 48.0,
        "vibration_frequency": 3.2,
        "pore_water_pressure": 75.0,
        "temperature": 16.0
    }
    pred = api_post("/api/model/predict", catastrophic_reading)
    assert pred["risk_level"] == "Critical", f"Expected Critical, got {pred['risk_level']}"
    assert pred["risk_score"] >= 80.0, f"Expected risk score >= 80, got {pred['risk_score']}"
    assert len(pred["contributing_factors"]) >= 2
    print(f"  -> Passed! Model classified catastrophic telemetry as: {pred['risk_level']} (Score: {pred['risk_score']}/100)")
    print(f"     Factors: {pred['contributing_factors']}")

    # 5. Threshold Calibration API
    print("\n[TEST 5] Station Threshold Calibration...")
    thresh_update = api_post("/api/stations/3/thresholds", {"warning_threshold": 52.0, "critical_threshold": 78.0})
    assert thresh_update["status"] == "success"
    assert thresh_update["warning_threshold"] == 52.0
    assert thresh_update["critical_threshold"] == 78.0
    print("  -> Passed! Custom thresholds saved.")

    # 6. WebSocket Telemetry Streaming
    print("\n[TEST 6] Real-time WebSocket Telemetry Broadcast...")
    async def verify_ws():
        async with websockets.connect(WS_URL) as ws:
            handshake = json.loads(await ws.recv())
            assert handshake["type"] == "connection_established"
            
            # Receive real-time telemetry tick
            msg = json.loads(await ws.recv())
            assert msg["type"] == "telemetry_batch"
            assert "stations" in msg and len(msg["stations"]) == 6
            print(f"  -> Passed! Received live WebSocket batch (tick {msg.get('tick')}) with 6 stations.")

    asyncio.run(verify_ws())

    # 7. Scenario Injection & Real-Time Escalation
    print("\n[TEST 7] Scenario Injection & Alert Generation...")
    scenario_res = api_post("/api/simulation/scenario", {
        "scenario_type": "cloudburst",
        "station_id": 3,
        "duration_seconds": 60,
        "intensity": 1.5
    })
    assert scenario_res["status"] == "success"
    print(f"  -> Injected cloudburst scenario on Station 3. Waiting 6 seconds for physics escalation...")

    time.sleep(6)

    # Check alert feed
    alerts = api_get("/api/alerts?limit=10")
    assert len(alerts) > 0, "No alerts found"
    latest_alert = alerts[0]
    print(f"  -> Alert registered: [{latest_alert['risk_level']}] {latest_alert['message']}")
    print(f"     Action Protocol: {latest_alert['suggested_action'][:80]}...")

    # Test alert acknowledge
    ack_res = api_post(f"/api/alerts/{latest_alert['id']}/acknowledge", {})
    assert ack_res["status"] == "success"
    print(f"  -> Passed! Alert {latest_alert['id']} marked ACKNOWLEDGED.")

    # Test alert resolve
    res_res = api_post(f"/api/alerts/{latest_alert['id']}/resolve", {})
    assert res_res["status"] == "success"
    print(f"  -> Passed! Alert {latest_alert['id']} marked RESOLVED.")

    # Reset station 3 back to normal
    api_post("/api/simulation/scenario", {"scenario_type": "normal", "station_id": 3})
    print("  -> Station 3 reset to normal baseline.")

    # 8. Notification logs
    print("\n[TEST 8] Emergency Dispatch Notification Logs...")
    notifs = api_get("/api/alerts/notifications?limit=5")
    assert len(notifs) > 0, "No notifications logged"
    print(f"  -> Passed! Retrieved {len(notifs)} mock SMS/Email dispatch records.")
    print(f"     Example [{notifs[0]['channel']}] to {notifs[0]['recipient']}: {notifs[0]['status']}")

    print("\n" + "=" * 60)
    print("ALL 8 VERIFICATION TESTS PASSED CLEANLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
