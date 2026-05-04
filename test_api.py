"""
Tests for the CN_PROJ Flask API.
Run with: python -m pytest test_api.py -v
Requires the app to be running: python app.py
"""
import requests
import pytest

BASE_URL = "http://localhost:5001"


# ─── /health ──────────────────────────────────────────────────────────────────

def test_health_returns_ok():
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


# ─── /simulate — admin mode ───────────────────────────────────────────────────

def test_simulate_admin_basic():
    payload = {"mode": "admin", "nodes": 15, "malicious_ratio": 0.2, "test_episodes": 10}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "accuracy" in data
    assert "crash_rate" in data
    assert "demo_path" in data
    assert "nodes" in data
    assert "edges" in data
    assert "hop_count" in data
    assert 0.0 <= data["accuracy"] <= 100.0
    assert 0.0 <= data["crash_rate"] <= 100.0

def test_simulate_admin_demo_path_starts_at_source():
    payload = {"mode": "admin", "nodes": 15, "malicious_ratio": 0.2, "test_episodes": 5}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["demo_path"][0] == data["demo_src"], \
        "demo_path must start at demo_src"

def test_simulate_admin_hop_count_matches_path():
    payload = {"mode": "admin", "nodes": 20, "malicious_ratio": 0.15, "test_episodes": 5}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    expected_hops = len(data["demo_path"]) - 1
    assert data["hop_count"] == expected_hops, \
        f"hop_count {data['hop_count']} != path length - 1 ({expected_hops})"

def test_simulate_admin_high_node_count():
    """High node count should still return valid data (just fewer episodes internally)."""
    payload = {"mode": "admin", "nodes": 150, "malicious_ratio": 0.1, "test_episodes": 5}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["nodes"]) == 150

def test_simulate_admin_high_malicious_ratio():
    """Edge case: very high malicious ratio (0.9) should still return 200."""
    payload = {"mode": "admin", "nodes": 10, "malicious_ratio": 0.9, "test_episodes": 3}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200

def test_simulate_admin_nodes_have_trust():
    """Every node in response should have a trust field."""
    payload = {"mode": "admin", "nodes": 15, "malicious_ratio": 0.2, "test_episodes": 3}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    for node in data["nodes"]:
        assert "trust" in node, f"Node {node['id']} missing trust field"
        assert 0.0 <= node["trust"] <= 1.0


# ─── /simulate — input validation ────────────────────────────────────────────

def test_simulate_invalid_nodes_below_min():
    payload = {"mode": "admin", "nodes": 2, "malicious_ratio": 0.2}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 400

def test_simulate_invalid_nodes_above_max():
    payload = {"mode": "admin", "nodes": 999, "malicious_ratio": 0.2}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 400

def test_simulate_invalid_malicious_ratio_above_max():
    payload = {"mode": "admin", "nodes": 15, "malicious_ratio": 0.95}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 400

def test_simulate_invalid_string_nodes():
    payload = {"mode": "admin", "nodes": "abc", "malicious_ratio": 0.2}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 400


# ─── /simulate — user mode ───────────────────────────────────────────────────

def test_simulate_user_mode():
    payload = {"mode": "user"}
    res = requests.post(f"{BASE_URL}/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "demo_path" in data
    assert "reason" in data
    assert data["reason"] in ["SUCCESS", "MALICIOUS", "TIMEOUT", ""]


# ─── /retrain ─────────────────────────────────────────────────────────────────

def test_retrain_starts_async():
    """Retrain should return 202 immediately (background thread)."""
    res = requests.post(f"{BASE_URL}/retrain",
                        json={"episodes": 100},
                        headers={"Content-Type": "application/json"})
    assert res.status_code == 202
    body = res.json()
    assert body["status"] == "started"
    assert body["episodes"] == 100


def test_retrain_status_endpoint():
    """Status endpoint should return progress fields."""
    res = requests.get(f"{BASE_URL}/retrain/status")
    assert res.status_code == 200
    body = res.json()
    for field in ["running", "done", "progress", "total", "pct", "episodes"]:
        assert field in body, f"Missing field: {field}"


def test_retrain_concurrent_rejected():
    """A second retrain while one is running should return 409."""
    import threading, time
    # Fire a long retrain in a background thread
    requests.post(f"{BASE_URL}/retrain", json={"episodes": 500})
    time.sleep(0.2)
    # Check status — if running, a second call must 409
    status = requests.get(f"{BASE_URL}/retrain/status").json()
    if status.get("running"):
        res2 = requests.post(f"{BASE_URL}/retrain", json={"episodes": 100})
        assert res2.status_code == 409
