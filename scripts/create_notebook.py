"""Create the concise technical notebook used as the project presentation."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_eda_bandit.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        """
# Decisão adaptativa de canais — Datathon MLET

## 1. Contexto do negócio

Uma instituição financeira precisa escolher entre contato por **cellular** ou **telephone** para cada oportunidade elegível. Comparamos a regra determinística `always_telephone` a Thompson Sampling contextual por segmentos, buscando aumentar recompensa/conversão acumulada com uma decisão explicável.
"""
    ),
    markdown(
        """
## 2. Fonte Kaggle

Base pública [`henriqueyamahata/bank-marketing`](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing). A unidade é uma interação; `y` é normalizado para recompensa binária. Seed global: 42.
"""
    ),
    code(
        """
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display

cwd = Path.cwd()
PROJECT_ROOT = cwd if (cwd / "src").exists() else cwd.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.bandits import ContextualThompsonSampling
from src.data import ARMS, load_bank_marketing, prepare_data, train_test_split
from src.evaluation import estimate_reward_probabilities

raw, dataset_path = load_bank_marketing(download_if_missing=True)
print(f"Arquivo: {dataset_path}")
print(f"Dimensão bruta: {raw.shape}")
"""
    ),
    markdown("## 3–4. Carregamento, schema e qualidade"),
    code(
        """
quality = pd.Series({
    "linhas": len(raw),
    "colunas": raw.shape[1],
    "duplicadas": int(raw.duplicated().sum()),
    "células_nulas": int(raw.isna().sum().sum()),
})
display(quality.to_frame("valor"))
display(pd.DataFrame({"dtype": raw.dtypes.astype(str), "nulos": raw.isna().sum()}).head(25))
"""
    ),
    markdown("## 5–6. Limpeza e prevenção de leakage"),
    code(
        """
prepared, cleaning = prepare_data(raw)
display(pd.Series(cleaning, name="resultado da limpeza").to_frame())
assert "duration" not in prepared.columns
print("`duration` foi removida: só é conhecida após o contato e causaria leakage.")
"""
    ),
    markdown("## 7. Distribuição do target"),
    code(
        """
target_distribution = prepared["reward"].value_counts(normalize=True).sort_index()
display(target_distribution.rename(index={0: "não converteu", 1: "converteu"}).to_frame("proporção"))
target_distribution.plot.bar(title="Distribuição da recompensa", ylabel="Proporção", rot=0)
plt.show()
"""
    ),
    markdown("## 8. Conversão observada por canal"),
    code(
        """
contact_summary = prepared.groupby("contact")["reward"].agg(["count", "mean"])
display(contact_summary.style.format({"mean": "{:.2%}"}))
contact_summary["mean"].plot.bar(title="Conversão observada por contact", ylabel="Conversão", rot=0)
plt.show()
"""
    ),
    markdown("## 9. Contexto: `poutcome`, `previous` e `campaign`"),
    code(
        """
context_analysis = pd.DataFrame({
    "poutcome": prepared.groupby("poutcome")["reward"].mean(),
}).dropna()
display(context_analysis.style.format("{:.2%}"))
display(prepared.assign(previous_bucket=prepared["previous"].gt(0).map({False: "0", True: ">0"})).groupby("previous_bucket")["reward"].agg(["count", "mean"]))
display(prepared.assign(campaign_bucket=prepared["campaign"].le(2).map({True: "1-2", False: "3+"})).groupby("campaign_bucket")["reward"].agg(["count", "mean"]))
"""
    ),
    markdown("## 10. Segmentação explicável"),
    code(
        """
segment_summary = prepared.groupby("segment")["reward"].agg(["count", "mean"]).sort_values("mean", ascending=False)
display(segment_summary.style.format({"mean": "{:.2%}"}))
"""
    ),
    markdown("## 11. Baseline oficial — `always_telephone`"),
    code(
        """
metrics = json.loads((PROJECT_ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
print(f"Conversão simulada do baseline fixo: {metrics['baseline_fixed_conversion']:.3%}")
"""
    ),
    markdown("## 12. Baseline adicional — melhor braço histórico global"),
    code(
        """
print(f"Braço global: {metrics['baseline_best_arm']}")
print(f"Conversão simulada: {metrics['baseline_best_arm_conversion']:.3%}")
"""
    ),
    markdown("## 13. Thompson Sampling contextual"),
    code(
        """
train, test = train_test_split(prepared, seed=42)
bandit = ContextualThompsonSampling.warm_start(train, ARMS, seed=42)
probabilities, global_probabilities = estimate_reward_probabilities(train, ARMS)
print("Prior Beta(1,1); warm-start: alpha=1+sucessos e beta=1+falhas.")
display(pd.DataFrame({arm: {"posterior_mean_global": bandit.posterior_mean("segmento_novo", arm)} for arm in ARMS}).T)
"""
    ),
    markdown(
        """
## 14. Metodologia da simulação offline

As probabilidades `P(reward=1 | segmento, braço)` são estimadas **somente no treino** com suavização Beta/Laplace. Os contextos do teste são repetidos até 50 mil interações e uma matriz de recompensas potenciais é pré-gerada com seed 42. Todas as políticas usam essa mesma matriz; o oráculo escolhe a maior probabilidade estimada e existe apenas para o cálculo de regret.

Esta é uma simulação contrafactual calibrada em dados observacionais — **não é avaliação causal**. Só um experimento controlado/A-B pode estimar efeito em produção.
"""
    ),
    markdown("## 15. Métricas comparativas"),
    code(
        """
comparison = pd.DataFrame({
    "política": ["always_telephone", "global_best_historical", "contextual_thompson_sampling"],
    "conversão": [metrics["baseline_fixed_conversion"], metrics["baseline_best_arm_conversion"], metrics["adaptive_conversion"]],
    "recompensa acumulada": [metrics["baseline_fixed_cumulative_reward"], metrics["baseline_best_arm_cumulative_reward"], metrics["adaptive_cumulative_reward"]],
})
display(comparison.style.format({"conversão": "{:.3%}"}))
print(f"Lift absoluto: {metrics['lift_absolute']:.3%}")
print(f"Lift relativo: {metrics['lift_percent']:.2f}%")
print(f"Pseudo-regret acumulado: {metrics['cumulative_regret']:.3f}")
"""
    ),
    markdown("## 16. Gráficos gerados"),
    code(
        """
for filename in ["conversion_comparison.png", "cumulative_reward.png", "cumulative_regret.png", "arm_selection_distribution.png"]:
    print(filename)
    display(Image(filename=str(PROJECT_ROOT / "artifacts" / filename)))
"""
    ),
    markdown("## 17. Golden Set — exatamente 5 casos"),
    code(
        """
golden = pd.read_csv(PROJECT_ROOT / "artifacts" / "golden_set.csv")
assert len(golden) == 5
display(golden)
"""
    ),
    markdown(
        """
## 18. Limitações e próximos passos

- O histórico registra apenas o braço escolhido; não há recompensa contrafactual observada.
- As probabilidades simuladas podem reproduzir viés de seleção e mudanças temporais do dataset.
- O serviço atual entrega um snapshot por maior média posterior; aprendizado online exige endpoint/evento de feedback, guardrails e auditoria.
- Antes de produção: experimento controlado, monitoramento de drift/conversão/distribuição de ações e análise de disparidades, com humano no loop para decisões sensíveis.
"""
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11+"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
