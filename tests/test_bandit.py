import pandas as pd

from src.bandits import ContextualThompsonSampling


ARMS = ("cellular", "telephone")


def test_choose_arm_is_valid() -> None:
    bandit = ContextualThompsonSampling(ARMS, seed=42)
    assert bandit.choose_arm("new-segment") in ARMS


def test_update_increments_alpha_and_beta() -> None:
    bandit = ContextualThompsonSampling(ARMS, min_segment_observations=0)
    bandit.update("segment-a", "cellular", 1)
    assert bandit.segment_posteriors["segment-a"]["cellular"].alpha == 2
    bandit.update("segment-a", "cellular", 0)
    assert bandit.segment_posteriors["segment-a"]["cellular"].beta == 2


def test_unknown_segment_uses_global_fallback() -> None:
    frame = pd.DataFrame(
        {
            "segment": ["known"] * 4,
            "contact": ["cellular", "cellular", "telephone", "telephone"],
            "reward": [1, 1, 0, 0],
        }
    )
    bandit = ContextualThompsonSampling.warm_start(frame, ARMS)
    assert bandit.posterior_mean("unknown", "cellular") == bandit.global_posteriors["cellular"].mean
    assert bandit.posterior_mean("unknown", "telephone") == bandit.global_posteriors["telephone"].mean


def test_state_round_trip() -> None:
    bandit = ContextualThompsonSampling(ARMS, seed=42)
    bandit.update("segment-a", "cellular", 1)
    restored = ContextualThompsonSampling.from_dict(bandit.to_dict())
    assert restored.to_dict() == bandit.to_dict()
