"""Contextual Thompson Sampling with Beta-Bernoulli posteriors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class BetaPosterior:
    alpha: float = 1.0
    beta: float = 1.0
    observations: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, reward: int) -> None:
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")
        self.alpha += reward
        self.beta += 1 - reward
        self.observations += 1

    def to_dict(self) -> dict[str, float | int]:
        return {
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "observations": int(self.observations),
        }


class ContextualThompsonSampling:
    """A small contextual bandit backed by discrete segment posteriors."""

    def __init__(
        self,
        arms: Iterable[str],
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        min_segment_observations: int = 5,
        seed: int = 42,
    ) -> None:
        self.arms = tuple(arms)
        if not self.arms:
            raise ValueError("At least one arm is required")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.min_segment_observations = int(min_segment_observations)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.global_posteriors = {
            arm: BetaPosterior(self.alpha_prior, self.beta_prior, 0)
            for arm in self.arms
        }
        self.segment_posteriors: dict[str, dict[str, BetaPosterior]] = {}

    @classmethod
    def warm_start(
        cls,
        frame: pd.DataFrame,
        arms: Iterable[str],
        seed: int = 42,
        min_segment_observations: int = 5,
    ) -> "ContextualThompsonSampling":
        """Create posteriors from observed segment/action/reward counts."""
        bandit = cls(
            arms=arms,
            seed=seed,
            min_segment_observations=min_segment_observations,
        )
        required = {"segment", "contact", "reward"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing warm-start columns: {sorted(missing)}")

        for arm in bandit.arms:
            arm_rewards = frame.loc[frame["contact"] == arm, "reward"]
            successes = int(arm_rewards.sum())
            observations = int(arm_rewards.count())
            bandit.global_posteriors[arm] = BetaPosterior(
                1.0 + successes, 1.0 + observations - successes, observations
            )

        grouped = frame.groupby(["segment", "contact"])["reward"].agg(["sum", "count"])
        for (segment, arm), row in grouped.iterrows():
            if arm not in bandit.arms:
                continue
            posterior = bandit._ensure_segment(str(segment))[str(arm)]
            successes = int(row["sum"])
            observations = int(row["count"])
            posterior.alpha = 1.0 + successes
            posterior.beta = 1.0 + observations - successes
            posterior.observations = observations
        return bandit

    def _ensure_segment(self, segment: str) -> dict[str, BetaPosterior]:
        if segment not in self.segment_posteriors:
            self.segment_posteriors[segment] = {
                arm: BetaPosterior(self.alpha_prior, self.beta_prior, 0)
                for arm in self.arms
            }
        return self.segment_posteriors[segment]

    def posterior(self, segment: str, arm: str) -> BetaPosterior:
        """Return segment posterior or the arm's global fallback."""
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")
        candidate = self.segment_posteriors.get(segment, {}).get(arm)
        if candidate is None or candidate.observations < self.min_segment_observations:
            return self.global_posteriors[arm]
        return candidate

    def posterior_mean(self, segment: str, arm: str) -> float:
        return self.posterior(segment, arm).mean

    def choose_arm(self, segment: str) -> str:
        samples = {
            arm: self.rng.beta(
                self.posterior(segment, arm).alpha,
                self.posterior(segment, arm).beta,
            )
            for arm in self.arms
        }
        return max(self.arms, key=lambda arm: samples[arm])

    def best_mean_arm(self, segment: str) -> str:
        """Return the deployment snapshot action with highest posterior mean."""
        return max(self.arms, key=lambda arm: self.posterior_mean(segment, arm))

    def update(self, segment: str, arm: str, reward: int) -> None:
        if arm not in self.arms:
            raise ValueError(f"Unknown arm: {arm}")
        self.global_posteriors[arm].update(reward)
        self._ensure_segment(segment)[arm].update(reward)

    def to_dict(self) -> dict[str, Any]:
        return {
            "arms": list(self.arms),
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "min_segment_observations": self.min_segment_observations,
            "seed": self.seed,
            "global_posteriors": {
                arm: posterior.to_dict()
                for arm, posterior in self.global_posteriors.items()
            },
            "segment_posteriors": {
                segment: {
                    arm: posterior.to_dict() for arm, posterior in posteriors.items()
                }
                for segment, posteriors in self.segment_posteriors.items()
            },
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "ContextualThompsonSampling":
        bandit = cls(
            arms=state["arms"],
            alpha_prior=state.get("alpha_prior", 1.0),
            beta_prior=state.get("beta_prior", 1.0),
            min_segment_observations=state.get("min_segment_observations", 5),
            seed=state.get("seed", 42),
        )
        bandit.global_posteriors = {
            arm: BetaPosterior(**posterior)
            for arm, posterior in state["global_posteriors"].items()
        }
        bandit.segment_posteriors = {
            segment: {
                arm: BetaPosterior(**posterior)
                for arm, posterior in posteriors.items()
            }
            for segment, posteriors in state.get("segment_posteriors", {}).items()
        }
        return bandit
