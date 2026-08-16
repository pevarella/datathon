import pandas as pd

from src.segmentation import add_segments, make_segment, normalize_poutcome


def test_known_contexts_generate_expected_segments() -> None:
    assert make_segment("success", 1, 2) == "poutcome=success|previous=gt0|campaign=1-2"
    assert make_segment("failure", 0, 3) == "poutcome=failure|previous=0|campaign=3+"
    assert make_segment("nonexistent", 0, 1) == "poutcome=nonexistent/other|previous=0|campaign=1-2"


def test_poutcome_normalization_and_dataframe_helper() -> None:
    assert normalize_poutcome("UNKNOWN") == "nonexistent/other"
    result = add_segments(pd.DataFrame({"poutcome": ["other"], "previous": [2], "campaign": [4]}))
    assert result.loc[0, "segment"] == "poutcome=nonexistent/other|previous=gt0|campaign=3+"
