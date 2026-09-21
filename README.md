# linccc

Lin's Concordance Correlation Coefficient (CCC) for Python.

CCC (Lin, 1989) measures agreement between two sets of paired
measurements — for example, two instruments, raters, or methods
measuring the same thing. It combines precision (Pearson correlation)
and accuracy (deviation from the 45-degree line of perfect agreement)
into a single statistic ranging from -1 to 1.

> **Not to be confused with:** [`ccc-coef`](https://pypi.org/project/ccc-coef/)
> on PyPI, which implements the *Clustermatch Correlation Coefficient* —
> an unrelated statistic that happens to share the same acronym. This
> package implements Lin's original concordance correlation coefficient,
> the standard measure of agreement/reproducibility used in R's
> `epiR::epi.ccc()` and `DescTools::CCC()`.

## Installation

```bash
pip install linccc
```

## Usage

```python
from linccc import concordance_correlation_coefficient

x = [1, 2, 3, 4, 5]
y = [1.1, 1.9, 3.2, 3.9, 5.05]

result = concordance_correlation_coefficient(x, y)
print(result.ccc)         # 0.996...
print(result.pearson_r)   # Pearson correlation
print(result.mean_diff)   # mean(x) - mean(y)

# also unpacks like a plain tuple
ccc, r, diff = concordance_correlation_coefficient(x, y)
```

NaNs are dropped pairwise: if either `x[i]` or `y[i]` is `NaN`, that
pair is excluded from both arrays before computing the statistic.

## Validation

Results are cross-validated against R's `epiR::epi.ccc()` on identical
data (see `tests/test_core.py`, `test_matches_r_epiR_reference`).

## Citation

If you use this package in published work, please cite:

> Lin, L. I. (1989). A concordance correlation coefficient to evaluate
> reproducibility. *Biometrics*, 45(1), 255-268.

See `CITATION.cff` for citing this software package itself.

## License

MIT — see `LICENSE`.
