"""Persistence and visualization helpers for training outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation import EvaluationResult


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_evaluation(result: EvaluationResult, output_dir: Path) -> list[Path]:
    """Create the four required comparison charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = result.metrics
    paths: list[Path] = []

    comparison_path = output_dir / "conversion_comparison.png"
    labels = ["Telefone fixo", "Melhor global", "Thompson Sampling"]
    values = [
        metrics["baseline_fixed_conversion"],
        metrics["baseline_best_arm_conversion"],
        metrics["adaptive_conversion"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, values, color=["#8b95a5", "#4c78a8", "#f58518"])
    ax.set_ylabel("Taxa de conversão simulada")
    ax.set_title("Conversão por política")
    ax.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=3)
    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)
    paths.append(comparison_path)

    reward_path = output_dir / "cumulative_reward.png"
    trajectories = result.trajectories
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trajectories["interaction"], trajectories["baseline_fixed_cumulative_reward"], label="Telefone fixo")
    ax.plot(trajectories["interaction"], trajectories["baseline_best_arm_cumulative_reward"], label="Melhor global")
    ax.plot(trajectories["interaction"], trajectories["adaptive_cumulative_reward"], label="Thompson Sampling")
    ax.set(xlabel="Interações", ylabel="Recompensa acumulada", title="Evolução da recompensa")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reward_path, dpi=150)
    plt.close(fig)
    paths.append(reward_path)

    regret_path = output_dir / "cumulative_regret.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(trajectories["interaction"], trajectories["adaptive_cumulative_regret"], color="#e45756")
    ax.set(xlabel="Interações", ylabel="Pseudo-regret acumulado", title="Regret contra o oráculo contextual")
    fig.tight_layout()
    fig.savefig(regret_path, dpi=150)
    plt.close(fig)
    paths.append(regret_path)

    distribution_path = output_dir / "arm_selection_distribution.png"
    arms = list(result.selection_counts)
    counts = [result.selection_counts[arm] for arm in arms]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(arms, counts, color=["#54a24b", "#b279a2"])
    ax.set(xlabel="Braço", ylabel="Seleções", title="Distribuição de ações do Thompson Sampling")
    ax.bar_label(bars, padding=3)
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=150)
    plt.close(fig)
    paths.append(distribution_path)
    return paths
