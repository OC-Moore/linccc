"""
Lin's Concordance Correlation Coefficient (CCC).

Reference
---------
Lin, L. I. (1989). A concordance correlation coefficient to evaluate
reproducibility. Biometrics, 45(1), 255-268.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np


class CCCResult(NamedTuple):
    """Result of a concordance correlation coefficient calculation."""

    ccc: float
    pearson_r: float
    mean_diff: float


def concordance_correlation_coefficient(
    x: Sequence[float], y: Sequence[float]
) -> CCCResult:
    """
    Compute Lin's Concordance Correlation Coefficient (CCC) between two
    paired sets of measurements.

    The CCC evaluates agreement between two variables by measuring both
    precision (Pearson correlation) and accuracy (how far the best-fit
    line deviates from the 45-degree line of perfect concordance). It
    ranges from -1 (perfect discordance) to 1 (perfect concordance).

    Parameters
    ----------
    x, y : array-like
        Paired observations. Must be the same length. NaNs are dropped
        pairwise (a NaN in either array removes that pair from both).

    Returns
    -------
    CCCResult
        Named tuple with fields:

        - ``ccc`` : the concordance correlation coefficient
        - ``pearson_r`` : the Pearson correlation coefficient
        - ``mean_diff`` : ``mean(x) - mean(y)``

        Field access works both by name (``result.ccc``) and by index
        (``result[0]``), and the result also unpacks like a plain tuple.

    Raises
    ------
    ValueError
        If ``x`` and ``y`` have different lengths.

    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [1.1, 1.9, 3.2, 3.9, 5.05]
    >>> result = concordance_correlation_coefficient(x, y)
    >>> round(result.ccc, 3)
    0.996
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x.shape} and {y.shape}"
        )

    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0:
        return CCCResult(ccc=np.nan, pearson_r=np.nan, mean_diff=np.nan)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    var_x = np.var(x, ddof=0)
    var_y = np.var(y, ddof=0)

    rho = np.corrcoef(x, y)[0, 1]

    ccc = (2 * rho * np.sqrt(var_x * var_y)) / (
        var_x + var_y + (mean_x - mean_y) ** 2
    )

    return CCCResult(
        ccc=float(ccc),
        pearson_r=float(rho),
        mean_diff=float(mean_x - mean_y),
    )
