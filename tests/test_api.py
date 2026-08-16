from fastapi.testclient import TestClient

import app.main as api_module


MINIMAL_POLICY = {
    "policy": "contextual_thompson_sampling",
    "policy_version": "test",
    "bandit_state": {
        "arms": ["cellular", "telephone"],
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "min_segment_observations": 5,
        "seed": 42,
        "global_posteriors": {
            "cellular": {"alpha": 8.0, "beta": 2.0, "observations": 8},
            "telephone": {"alpha": 2.0, "beta": 8.0, "observations": 8},
        },
        "segment_posteriors": {},
    },
    "reward_probabilities": {},
    "global_reward_probabilities": {"cellular": 0.8, "telephone": 0.2},
}


def test_health() -> None:
    response = TestClient(api_module.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommend(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "POLICY", MINIMAL_POLICY)
    response = TestClient(api_module.app).post(
        "/recommend", json={"poutcome": "success", "previous": 1, "campaign": 1}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["recommended_action"] in {"cellular", "telephone"}
    assert body["policy"] == "contextual_thompson_sampling"


def test_invalid_payload_returns_validation_error(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "POLICY", MINIMAL_POLICY)
    response = TestClient(api_module.app).post(
        "/recommend", json={"poutcome": "success", "previous": -1, "campaign": 0}
    )
    assert response.status_code == 422


def test_missing_policy_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "POLICY", None)
    response = TestClient(api_module.app).post(
        "/recommend", json={"poutcome": "success", "previous": 1, "campaign": 1}
    )
    assert response.status_code == 503
    assert "scripts/train.py" in response.json()["detail"]
