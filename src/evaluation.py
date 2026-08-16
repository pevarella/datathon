"""Reproducible observationally calibrated offline reward simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.bandits import ContextualThompsonSampling


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    trajectories: pd.DataFrame
    selection_counts: dict[str, int]


def estimate_reward_probabilities(
    train: pd.DataFrame,
    arms: Iterable[str],
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Estimate smoothed P(reward | segment, arm), with global fallbacks."""
    arm_list = tuple(arms)
    global_probabilities: dict[str, float] = {}
    for arm in arm_list:
        rewards = train.loc[train["contact"] == arm, "reward"]
        global_probabilities[arm] = float(
            (alpha_prior + rewards.sum())
            / (alpha_prior + beta_prior + rewards.count())
        )

    segment_probabilities: dict[str, dict[str, float]] = {}
    for segment in sorted(train["segment"].unique()):
        segment_probabilities[str(segment)] = {}
        segment_rows = train[train["segment"] == segment]
        for arm in arm_list:
            rewards = segment_rows.loc[segment_rows["contact"] == arm, "reward"]
            if rewards.empty:
                probability = global_probabilities[arm]
            else:
                probability = float(
                    (alpha_prior + rewards.sum())
                    / (alpha_prior + beta_prior + rewards.count())
                )
            segment_probabilities[str(segment)][arm] = probability
    return segment_probabilities, global_probabilities


def _probability(
    segment: str,
    arm: str,
    segment_probabilities: dict[str, dict[str, float]],
    global_probabilities: dict[str, float],
) -> float:
    return segment_probabilities.get(segment, {}).get(arm, global_probabilities[arm])


def evaluate_policies(
    train: pd.DataFrame,
    test: pd.DataFrame,
    arms: Iterable[str],
    segment_probabilities: dict[str, dict[str, float]],
    global_probabilities: dict[str, float],
    n_interactions: int = 50_000,
    seed: int = 42,
) -> EvaluationResult:
    """Compare policies on one shared, pre-generated potential-reward matrix."""
    arm_list = tuple(arms)
    if test.empty:
        raise ValueError("Test data must not be empty")
    repeats = int(np.ceil(n_interactions / len(test)))
    contexts = np.tile(test["segment"].to_numpy(), repeats)[:n_interactions]
    rng = np.random.default_rng(seed)
    reward_matrix: dict[str, np.ndarray] = {}
    probability_matrix: dict[str, np.ndarray] = {}
    for arm in arm_list:
        probabilities = np.array(
            [
                _probability(str(segment), arm, segment_probabilities, global_probabilities)
                for segment in contexts
            ]
        )
        probability_matrix[arm] = probabilities
        reward_matrix[arm] = (rng.random(n_interactions) < probabilities).astype(int)

    fixed_arm = "telephone"
    best_historical_arm = max(arm_list, key=lambda arm: global_probabilities[arm])
    adaptive = ContextualThompsonSampling.warm_start(train, arm_list, seed=seed)

    adaptive_rewards = np.zeros(n_interactions, dtype=int)
    adaptive_expected = np.zeros(n_interactions, dtype=float)
    chosen_arms: list[str] = []
    oracle_expected = np.maximum.reduce([probability_matrix[arm] for arm in arm_list])
    oracle_arms = [
        max(arm_list, key=lambda arm: probability_matrix[arm][index])
        for index in range(n_interactions)
    ]
    oracle_rewards = np.array(
        [reward_matrix[arm][index] for index, arm in enumerate(oracle_arms)], dtype=int
    )

    for index, raw_segment in enumerate(contexts):
        segment = str(raw_segment)
        arm = adaptive.choose_arm(segment)
        reward = int(reward_matrix[arm][index])
        adaptive_rewards[index] = reward
        adaptive_expected[index] = probability_matrix[arm][index]
        chosen_arms.append(arm)
        adaptive.update(segment, arm, reward)

    fixed_rewards = reward_matrix[fixed_arm]
    best_rewards = reward_matrix[best_historical_arm]
    baseline_fixed_conversion = float(fixed_rewards.mean())
    adaptive_conversion = float(adaptive_rewards.mean())
    lift_absolute = adaptive_conversion - baseline_fixed_conversion
    lift_percent = (
        100.0 * lift_absolute / baseline_fixed_conversion
        if baseline_fixed_conversion > 0
        else float("nan")
    )
    cumulative_regret = np.cumsum(oracle_expected - adaptive_expected)
    selection_counts = {
        arm: int(sum(chosen == arm for chosen in chosen_arms)) for arm in arm_list
    }
    selection_distribution = {
        arm: count / n_interactions for arm, count in selection_counts.items()
    }
    metrics: dict[str, Any] = {
        "baseline_fixed_policy": "always_telephone",
        "baseline_best_arm": best_historical_arm,
        "baseline_fixed_conversion": baseline_fixed_conversion,
        "baseline_best_arm_conversion": float(best_rewards.mean()),
        "adaptive_conversion": adaptive_conversion,
        "lift_absolute": lift_absolute,
        "lift_percent": lift_percent,
        "baseline_fixed_cumulative_reward": int(fixed_rewards.sum()),
        "baseline_best_arm_cumulative_reward": int(best_rewards.sum()),
        "adaptive_cumulative_reward": int(adaptive_rewards.sum()),
        "oracle_cumulative_reward": int(oracle_rewards.sum()),
        "cumulative_regret": float(cumulative_regret[-1]),
        "arm_selection_distribution": selection_distribution,
        "n_interactions": int(n_interactions),
        "seed": int(seed),
    }
    trajectories = pd.DataFrame(
        {
            "interaction": np.arange(1, n_interactions + 1),
            "baseline_fixed_cumulative_reward": np.cumsum(fixed_rewards),
            "baseline_best_arm_cumulative_reward": np.cumsum(best_rewards),
            "adaptive_cumulative_reward": np.cumsum(adaptive_rewards),
            "oracle_cumulative_reward": np.cumsum(oracle_rewards),
            "adaptive_cumulative_regret": cumulative_regret,
        }
    )
    return EvaluationResult(metrics, trajectories, selection_counts)
