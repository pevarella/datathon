"""Serving API for the persisted adaptive-offers policy snapshot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.bandits import ContextualThompsonSampling
from src.segmentation import make_segment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "artifacts" / "policy.json"


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Não foi possível carregar {path}: {exc}") from exc


POLICY = load_policy()
app = FastAPI(
    title="Adaptive Offers API",
    description="Recomendação de canal com política contextual persistida.",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    poutcome: str = Field(min_length=1)
    previous: int = Field(ge=0)
    campaign: int = Field(ge=1)


class RecommendationResponse(BaseModel):
    recommended_action: str
    segment: str
    estimated_conversion: float
    alternative_action: str
    alternative_estimated_conversion: float
    policy: str
    policy_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest) -> dict[str, Any]:
    if POLICY is None:
        raise HTTPException(
            status_code=503,
            detail="Política não encontrada. Execute `python scripts/train.py` primeiro.",
        )
    segment = make_segment(payload.poutcome, payload.previous, payload.campaign)
    bandit = ContextualThompsonSampling.from_dict(POLICY["bandit_state"])
    recommended = bandit.best_mean_arm(segment)
    alternative = next(arm for arm in bandit.arms if arm != recommended)
    rates = POLICY.get("reward_probabilities", {}).get(
        segment, POLICY["global_reward_probabilities"]
    )
    return {
        "recommended_action": recommended,
        "segment": segment,
        "estimated_conversion": float(
            rates.get(recommended, POLICY["global_reward_probabilities"][recommended])
        ),
        "alternative_action": alternative,
        "alternative_estimated_conversion": float(
            rates.get(alternative, POLICY["global_reward_probabilities"][alternative])
        ),
        "policy": POLICY["policy"],
        "policy_version": POLICY["policy_version"],
    }
