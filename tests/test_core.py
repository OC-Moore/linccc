"""
Tests for linccc.core.concordance_correlation_coefficient.

The `test_matches_r_epiR_reference` test uses reference_data.csv and
values cross-validated against R's epiR::epi.ccc() on identical data
(see the accompanying R script/notes for how it was generated). If you
change the core algorithm, re-validate against R before updating the
expected values below.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linccc import CCCResult, concordance_correlation_coefficient

DATA_DIR = Path(__file__).parent


def test_matches_r_epiR_reference():
    """Cross-validated against R's epiR::epi.ccc() on the same data."""
    df = pd.read_csv(DATA_DIR / "reference_data.csv")
    result = concordance_correlation_coefficient(df["x"], df["y"])

    assert isinstance(result, CCCResult)
    np.testing.assert_allclose(result.ccc, 0.8549844869, atol=1e-8)
    np.testing.assert_allclose(result.pearson_r, 0.8855754370, atol=1e-8)
    np.testing.assert_allclose(result.mean_diff, -0.4294705017, atol=1e-8)


def test_perfect_agreement():
    x = [1, 2, 3, 4, 5]
    result = concordance_correlation_coefficient(x, x)
    np.testing.assert_allclose(result.ccc, 1.0)
    np.testing.assert_allclose(result.pearson_r, 1.0)
    np.testing.assert_allclose(result.mean_diff, 0.0)


def test_perfect_negative_relationship_is_not_perfect_concordance():
    # Perfectly (negatively) correlated but not concordant: CCC should
    # be strongly negative, not equal to Pearson r's magnitude of 1.
    x = [1, 2, 3, 4, 5]
    y = [5, 4, 3, 2, 1]
    result = concordance_correlation_coefficient(x, y)
    np.testing.assert_allclose(result.pearson_r, -1.0)
    assert result.ccc < 0


def test_constant_input_returns_nan():
    x = [3, 3, 3, 3]
    y = [1, 2, 3, 4]
    result = concordance_correlation_coefficient(x, y)
    assert np.isnan(result.ccc)
    assert np.isnan(result.pearson_r)
    assert np.isnan(result.mean_diff)


def test_all_nan_returns_nan():
    x = [np.nan, np.nan, np.nan]
    y = [1, 2, 3]
    result = concordance_correlation_coefficient(x, y)
    assert np.isnan(result.ccc)


def test_pairwise_nan_masking():
    # A NaN in either array should drop that pair from both, not just
    # substitute/ignore it independently.
    x = [1, 2, np.nan, 4, 5]
    y = [1, np.nan, 3, 4, 5]
    result_masked = concordance_correlation_coefficient(x, y)
    result_clean = concordance_correlation_coefficient([1, 4, 5], [1, 4, 5])
    np.testing.assert_allclose(result_masked.ccc, result_clean.ccc)


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        concordance_correlation_coefficient([1, 2, 3], [1, 2])


def test_two_points():
    x = [1, 2]
    y = [1.1, 2.1]
    result = concordance_correlation_coefficient(x, y)
    assert not np.isnan(result.ccc)


def test_result_unpacks_like_tuple():
    x = [1, 2, 3]
    y = [1, 2, 3]
    ccc, r, diff = concordance_correlation_coefficient(x, y)
    assert ccc == pytest.approx(1.0)
    assert r == pytest.approx(1.0)
    assert diff == pytest.approx(0.0)
