"""Shared, explainable context segmentation used by training and serving."""

from __future__ import annotations

from typing import Any

import pandas as pd


POUTCOME_OTHER = "nonexistent/other"


def normalize_poutcome(value: Any) -> str:
    """Collapse historical outcomes into the three approved categories."""
    normalized = str(value).strip().lower() if not pd.isna(value) else ""
    if normalized == "success":
        return "success"
    if normalized == "failure":
        return "failure"
    return POUTCOME_OTHER


def make_segment(poutcome: Any, previous: Any, campaign: Any) -> str:
    """Build a stable segment key from the minimized operational context."""
    outcome_bucket = normalize_poutcome(poutcome)
    try:
        previous_bucket = "gt0" if float(previous) > 0 else "0"
    except (TypeError, ValueError):
        previous_bucket = "0"
    try:
        campaign_bucket = "1-2" if float(campaign) <= 2 else "3+"
    except (TypeError, ValueError):
        campaign_bucket = "1-2"
    return (
        f"poutcome={outcome_bucket}|previous={previous_bucket}"
        f"|campaign={campaign_bucket}"
    )


def add_segments(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized context and a derived segment column."""
    required = {"poutcome", "previous", "campaign"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing context columns: {sorted(missing)}")

    result = frame.copy()
    result["poutcome"] = result["poutcome"].map(normalize_poutcome)
    result["segment"] = [
        make_segment(outcome, previous, campaign)
        for outcome, previous, campaign in zip(
            result["poutcome"], result["previous"], result["campaign"]
        )
    ]
    return result
