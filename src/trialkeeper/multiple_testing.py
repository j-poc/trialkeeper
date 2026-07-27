"""Multiple-testing corrections, and what they do to a Sharpe ratio.

Testing 300 factors at the 5% level produces roughly 15 "discoveries" from pure
noise. Harvey, Liu & Zhu (2016) argue that a large share of the published
cross-sectional asset-pricing literature is exactly this, and that the sensible
response is to raise the bar rather than to keep counting significant t-stats.

Three adjustments, in increasing order of power:

* **Bonferroni** — controls the chance of *any* false positive. Simple,
  correct, and brutally conservative when the tests are correlated.
* **Holm** — same guarantee, uniformly more powerful. There is no reason to
  prefer Bonferroni to it.
* **Benjamini–Hochberg–Yekutieli** — controls the *proportion* of discoveries
  that are false rather than the chance of any. Includes the ``c(M)`` term that
  keeps the guarantee valid under arbitrary dependence, which matters here
  because factor tests are heavily correlated with one another.

:func:`sharpe_haircut` converts an adjusted p-value back into the Sharpe ratio
you are entitled to claim. That is the number to put in a memo — the raw one
describes a search, not a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Method = Literal["bonferroni", "holm", "bhy"]


@dataclass(frozen=True, slots=True)
class Haircut:
    """What survives of a Sharpe ratio once the search is accounted for."""

    observed_sharpe: float
    adjusted_sharpe: float
    observed_pvalue: float
    adjusted_pvalue: float
    n_tests: int
    method: Method

    @property
    def haircut_fraction(self) -> float:
        """Proportion of the reported Sharpe attributable to the search itself."""
        if self.observed_sharpe == 0:
            return 0.0
        return 1.0 - self.adjusted_sharpe / self.observed_sharpe

    @property
    def survives(self) -> bool:
        return self.adjusted_pvalue < 0.05

    def explain(self) -> str:
        verdict = "survives" if self.survives else "does not survive"
        return (
            f"{self.method}: p {self.observed_pvalue:.4g} → {self.adjusted_pvalue:.4g} "
            f"across {self.n_tests} tests; Sharpe {self.observed_sharpe:.3f} → "
            f"{self.adjusted_sharpe:.3f} ({self.haircut_fraction:.0%} haircut), {verdict} at 5%"
        )


def adjust_pvalues(pvalues: NDArray[np.float64], *, method: Method = "bhy") -> NDArray[np.float64]:
    """Multiplicity-adjusted p-values, in the input order.

    All three methods are monotone: a test with a smaller raw p-value never
    receives a larger adjusted one.
    """
    raw = np.asarray(pvalues, dtype=np.float64).ravel()
    if raw.size == 0:
        raise ValueError("no p-values supplied")
    if np.any((raw < 0.0) | (raw > 1.0)) or not np.all(np.isfinite(raw)):
        raise ValueError("p-values must be finite and lie in [0, 1]")

    count = raw.size
    if method == "bonferroni":
        return np.minimum(raw * count, 1.0)

    order = np.argsort(raw)
    ordered = raw[order]
    adjusted = np.empty_like(ordered)

    if method == "holm":
        # Step-down: multiply by the number of hypotheses still under test, then
        # enforce monotonicity with a running maximum from the smallest p upward.
        scaled = ordered * (count - np.arange(count))
        adjusted = np.maximum.accumulate(scaled)
    elif method == "bhy":
        # Benjamini-Hochberg with Yekutieli's c(M) = sum_{i=1..M} 1/i, which keeps
        # the FDR guarantee under arbitrary dependence. Factor tests are strongly
        # dependent, so the plain BH form would understate the correction.
        c_m = float(np.sum(1.0 / np.arange(1, count + 1)))
        scaled = ordered * count * c_m / np.arange(1, count + 1)
        adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    else:
        raise ValueError(f"unknown method: {method!r}")

    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored


def sharpe_haircut(
    *,
    observed_sharpe: float,
    n_observations: int,
    n_tests: int,
    other_pvalues: NDArray[np.float64] | None = None,
    method: Method = "bhy",
) -> Haircut:
    """The Sharpe ratio that survives a multiple-testing correction.

    ``observed_sharpe`` is per-observation. ``n_tests`` is the total number of
    strategies tested, this one included.

    ``other_pvalues`` are the p-values of the other tests when you have them; the
    step-down and step-up methods use the whole set. Without them the p-values of
    the unreported tests are treated as uniform on (0, 1) — the least favourable
    honest assumption, and a reminder that keeping the record is cheaper than
    reconstructing it.
    """
    if n_tests < 1:
        raise ValueError("n_tests must be at least 1")
    if n_observations < 2:
        raise ValueError("n_observations must be at least 2")

    t_statistic = observed_sharpe * np.sqrt(n_observations)
    observed_p = float(2.0 * (1.0 - stats.norm.cdf(abs(t_statistic))))

    if other_pvalues is not None:
        population = np.concatenate([[observed_p], np.asarray(other_pvalues, dtype=np.float64)])
    else:
        # Evenly spaced quantiles of the uniform: the expected p-value pattern
        # when the other tests found nothing.
        filler = (np.arange(1, n_tests) + 0.5) / n_tests if n_tests > 1 else np.array([])
        population = np.concatenate([[observed_p], filler])

    adjusted_p = float(adjust_pvalues(population, method=method)[0])
    adjusted_t = float(stats.norm.ppf(1.0 - adjusted_p / 2.0))
    adjusted_sharpe = max(0.0, adjusted_t / np.sqrt(n_observations))

    return Haircut(
        observed_sharpe=observed_sharpe,
        adjusted_sharpe=adjusted_sharpe,
        observed_pvalue=observed_p,
        adjusted_pvalue=adjusted_p,
        n_tests=int(population.size),
        method=method,
    )
