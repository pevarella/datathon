"""Run the end-to-end adaptive-offers training and evaluation pipeline."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.artifacts import plot_evaluation, write_json
from src.bandits import ContextualThompsonSampling
from src.data import ARMS, DATASET_SLUG, load_bank_marketing, prepare_data, train_test_split
from src.evaluation import estimate_reward_probabilities, evaluate_policies
from src.segmentation import make_segment


SEED = 42
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXPERIMENT_NAME = "datathon_adaptive_offers"
POLICY_VERSION = "1.0.0"


def build_golden_set(
    bandit: ContextualThompsonSampling,
    probabilities: dict[str, dict[str, float]],
    global_probabilities: dict[str, float],
) -> pd.DataFrame:
    """Build exactly five representative, deterministic serving examples."""
    cases = [
        ("GS-001", "success", 1, 1),
        ("GS-002", "failure", 1, 2),
        ("GS-003", "nonexistent", 0, 1),
        ("GS-004", "nonexistent", 0, 4),
        ("GS-005", "failure", 0, 3),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, poutcome, previous, campaign in cases:
        segment = make_segment(poutcome, previous, campaign)
        recommendation = bandit.best_mean_arm(segment)
        alternative = next(arm for arm in ARMS if arm != recommendation)
        rates = probabilities.get(segment, global_probabilities)
        recommended_rate = float(rates.get(recommendation, global_probabilities[recommendation]))
        alternative_rate = float(rates.get(alternative, global_probabilities[alternative]))
        if segment in probabilities:
            rationale = (
                f"A recomendação está alinhada à maior taxa suavizada do segmento "
                f"({recommended_rate:.2%} vs. {alternative_rate:.2%})."
            )
        else:
            rationale = (
                "Segmento sem histórico suficiente; recomendação baseada no "
                f"fallback global ({recommended_rate:.2%} vs. {alternative_rate:.2%})."
            )
        rows.append(
            {
                "case_id": case_id,
                "poutcome": poutcome,
                "previous": previous,
                "campaign": campaign,
                "segment": segment,
                "recommended_action": recommendation,
                "estimated_conversion": recommended_rate,
                "alternative_action": alternative,
                "alternative_estimated_conversion": alternative_rate,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def log_mlflow_run(
    params: dict[str, Any], metrics: dict[str, Any], artifact_paths: list[Path]
) -> str:
    """Register the local experiment and return its run ID."""
    import mlflow

    tracking_database = (PROJECT_ROOT / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{tracking_database.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="contextual_thompson_sampling") as run:
        mlflow.log_params(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
                for key, value in params.items()
            }
        )
        scalar_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for arm, share in metrics["arm_selection_distribution"].items():
            scalar_metrics[f"selection_share_{arm}"] = float(share)
        mlflow.log_metrics(scalar_metrics)
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="outputs")
        return run.info.run_id


def run_training(n_interactions: int = 50_000) -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    raw, dataset_path = load_bank_marketing(download_if_missing=True)
    prepared, cleaning = prepare_data(raw)
    train, test = train_test_split(prepared, test_size=0.20, seed=SEED)
    probabilities, global_probabilities = estimate_reward_probabilities(train, ARMS)

    attempts = [n_interactions, n_interactions * 2, n_interactions * 4]
    result = None
    used_interactions = n_interactions
    for interactions in attempts:
        result = evaluate_policies(
            train,
            test,
            ARMS,
            probabilities,
            global_probabilities,
            n_interactions=interactions,
            seed=SEED,
        )
        used_interactions = interactions
        if result.metrics["adaptive_conversion"] > result.metrics["baseline_fixed_conversion"]:
            break
    assert result is not None
    if result.metrics["adaptive_conversion"] <= result.metrics["baseline_fixed_conversion"]:
        raise RuntimeError(
            "A política adaptativa não superou always_telephone após ampliar a "
            "simulação de forma determinística; revise os dados e o ambiente de recompensa."
        )

    serving_bandit = ContextualThompsonSampling.warm_start(train, ARMS, seed=SEED)
    generated_at = datetime.now(timezone.utc).isoformat()
    policy = {
        "policy": "contextual_thompson_sampling",
        "policy_version": POLICY_VERSION,
        "serving_rule": "posterior_mean_argmax",
        "dataset_slug": DATASET_SLUG,
        "generated_at": generated_at,
        "bandit_state": serving_bandit.to_dict(),
        "reward_probabilities": probabilities,
        "global_reward_probabilities": global_probabilities,
        "segment_definition": {
            "poutcome": ["success", "failure", "nonexistent/other"],
            "previous": ["0", "gt0"],
            "campaign": ["1-2", "3+"],
        },
        "metrics": result.metrics,
    }
    policy_path = ARTIFACTS_DIR / "policy.json"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    write_json(policy_path, policy)
    write_json(metrics_path, result.metrics)

    golden_set = build_golden_set(serving_bandit, probabilities, global_probabilities)
    golden_path = ARTIFACTS_DIR / "golden_set.csv"
    golden_set.to_csv(golden_path, index=False, encoding="utf-8")
    plot_paths = plot_evaluation(result, ARTIFACTS_DIR)

    eda_summary = {
        "dataset_file": str(dataset_path.relative_to(PROJECT_ROOT)),
        "cleaning": cleaning,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "overall_conversion": float(prepared["reward"].mean()),
        "conversion_by_contact": {
            str(key): float(value)
            for key, value in prepared.groupby("contact")["reward"].mean().items()
        },
        "rows_by_contact": {
            str(key): int(value)
            for key, value in prepared["contact"].value_counts().items()
        },
        "conversion_by_poutcome": {
            str(key): float(value)
            for key, value in prepared.groupby("poutcome")["reward"].mean().items()
        },
        "used_interactions": used_interactions,
        "generated_at": generated_at,
    }
    summary_path = ARTIFACTS_DIR / "run_summary.json"
    write_json(summary_path, eda_summary)

    params = {
        "dataset_slug": DATASET_SLUG,
        "algorithm": "contextual_thompson_sampling",
        "arms": list(ARMS),
        "alpha_prior": 1.0,
        "beta_prior": 1.0,
        "seed": SEED,
        "segment_definition": "poutcome x previous(0/>0) x campaign(1-2/3+)",
        "train_size": len(train),
        "test_size": len(test),
        "n_interactions": used_interactions,
    }
    mlflow_artifacts = [policy_path, metrics_path, golden_path, summary_path, *plot_paths]
    run_id = log_mlflow_run(params, result.metrics, mlflow_artifacts)
    eda_summary["mlflow_run_id"] = run_id
    write_json(summary_path, eda_summary)

    print(f"Dataset: {dataset_path}")
    print(f"Linhas utilizáveis: {len(prepared):,} (treino={len(train):,}, teste={len(test):,})")
    print(f"Interações simuladas: {used_interactions:,}")
    print(json.dumps(result.metrics, indent=2, ensure_ascii=False))
    print(f"Policy: {policy_path}")
    print(f"Metrics: {metrics_path}")
    print(f"MLflow run: {run_id}")
    return {"metrics": result.metrics, "summary": eda_summary, "run_id": run_id}


if __name__ == "__main__":
    run_training()
