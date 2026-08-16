"""Dataset discovery, loading, cleaning, and deterministic splitting."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.segmentation import add_segments


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_SLUG = "henriqueyamahata/bank-marketing"
ARMS = ("cellular", "telephone")
TARGET_CANDIDATES = ("y", "deposit", "subscribed", "subscription", "target")


def _read_csv(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited Bank Marketing variants."""
    try:
        frame = pd.read_csv(path, sep=None, engine="python")
    except (UnicodeDecodeError, pd.errors.ParserError):
        frame = pd.read_csv(path, sep=None, engine="python", encoding="latin-1")
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    return frame


def _csv_score(path: Path) -> tuple[int, int]:
    """Prefer candidates whose header resembles a Bank Marketing dataset."""
    try:
        columns = set(_read_csv(path).columns)
    except Exception:
        return (-1, -1)
    expected = {"contact", "poutcome", "previous", "campaign"}
    score = len(columns.intersection(expected)) + int(bool(columns.intersection(TARGET_CANDIDATES)))
    return (score, int(path.stat().st_size))


def discover_csv(search_dir: Path = RAW_DATA_DIR) -> Path | None:
    """Find the most plausible CSV recursively below ``search_dir``."""
    candidates = list(search_dir.rglob("*.csv")) if search_dir.exists() else []
    if not candidates:
        return None
    best = max(candidates, key=_csv_score)
    return best if _csv_score(best)[0] >= 4 else None


def download_dataset() -> Path:
    """Download the public Kaggle dataset and copy its selected CSV locally."""
    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            "Dataset ausente e kagglehub não instalado. Instale requirements.txt "
            "ou coloque o CSV em data/raw/."
        ) from exc

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        downloaded_dir = Path(kagglehub.dataset_download(DATASET_SLUG))
    except Exception as exc:
        raise FileNotFoundError(
            "Não foi possível baixar o dataset. Coloque um CSV de Bank Marketing "
            f"em {RAW_DATA_DIR}. Erro original: {exc}"
        ) from exc

    selected = discover_csv(downloaded_dir)
    if selected is None:
        raise FileNotFoundError(
            f"O download em {downloaded_dir} não contém um CSV compatível."
        )
    destination = RAW_DATA_DIR / selected.name
    if selected.resolve() != destination.resolve():
        shutil.copy2(selected, destination)
    return destination


def load_bank_marketing(download_if_missing: bool = True) -> tuple[pd.DataFrame, Path]:
    """Load a local compatible CSV or acquire it through KaggleHub."""
    path = discover_csv()
    if path is None and download_if_missing:
        path = download_dataset()
    if path is None:
        raise FileNotFoundError(
            f"Nenhum CSV compatível encontrado. Coloque a base em {RAW_DATA_DIR}."
        )
    return _read_csv(path), path


def _find_target(columns: pd.Index) -> str:
    for candidate in TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    raise ValueError(f"Target não encontrado. Nomes aceitos: {TARGET_CANDIDATES}")


def _normalize_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        unique = set(numeric.dropna().unique())
        if unique.issubset({0, 1}):
            return numeric
    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "0": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the observed data and retain only rows usable by the simulator."""
    data = frame.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    target_column = _find_target(data.columns)
    original_rows = len(data)
    duration_removed = "duration" in data.columns
    if duration_removed:
        data = data.drop(columns=["duration"])

    required = {"contact", "poutcome", "previous", "campaign", target_column}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

    data["contact"] = data["contact"].astype(str).str.strip().str.lower()
    unknown_contact_rows = int((~data["contact"].isin(ARMS)).sum())
    data = data[data["contact"].isin(ARMS)].copy()
    data["reward"] = _normalize_target(data[target_column])
    data["previous"] = pd.to_numeric(data["previous"], errors="coerce")
    data["campaign"] = pd.to_numeric(data["campaign"], errors="coerce")
    before_required_drop = len(data)
    data = data.dropna(subset=["reward", "previous", "campaign", "poutcome"])
    invalid_required_rows = before_required_drop - len(data)
    data["reward"] = data["reward"].astype(int)
    data = add_segments(data).reset_index(drop=True)

    metadata: dict[str, Any] = {
        "original_rows": int(original_rows),
        "usable_rows": int(len(data)),
        "unknown_or_unsupported_contact_rows_removed": unknown_contact_rows,
        "invalid_required_rows_removed": int(invalid_required_rows),
        "duration_removed": duration_removed,
        "target_source_column": target_column,
        "arms": list(ARMS),
        "segments": int(data["segment"].nunique()),
    }
    return data, metadata


def train_test_split(
    frame: pd.DataFrame, test_size: float = 0.20, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a reproducible shuffled holdout without an extra dependency."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    test_count = max(1, int(round(len(frame) * test_size)))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return (
        frame.iloc[train_indices].reset_index(drop=True),
        frame.iloc[test_indices].reset_index(drop=True),
    )
