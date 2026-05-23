from __future__ import annotations

from fastapi.testclient import TestClient

from molecule_lab.api.app import app


client = TestClient(app)


def test_healthcheck() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_create_simulation() -> None:
    response = client.post(
        "/api/simulations",
        json={
            "preset": "debug",
            "graph": {"atoms": [{"id": "c1", "symbol": "C", "x": 0, "y": 0}]},
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["molecule"]["smiles"] == "C"
    assert body["events_url"].startswith("/api/simulations/")


def test_create_simulation_rejects_invalid_graph() -> None:
    response = client.post(
        "/api/simulations",
        json={
            "preset": "debug",
            "graph": {
                "atoms": [
                    {"id": "c1", "symbol": "C", "x": 0, "y": 0},
                    {"id": "c2", "symbol": "C", "x": 1, "y": 0},
                ],
                "bonds": [],
            },
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_molecule"


def test_stream_simulation_events() -> None:
    create_response = client.post(
        "/api/simulations",
        json={
            "preset": "debug",
            "graph": {"atoms": [{"id": "c1", "symbol": "C", "x": 0, "y": 0}]},
        },
    )
    events_url = create_response.json()["events_url"]

    with client.stream("GET", events_url) as response:
        text = response.read().decode()

    assert response.status_code == 200
    assert "event: metadata" in text
    assert "event: progress" in text
    assert "event: result" in text
