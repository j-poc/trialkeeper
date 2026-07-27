"""Sharpe ratios that account for how many times you looked.

A Sharpe ratio reported without the number of trials behind it carries no
information. Selecting the best of enough random strategies produces an
arbitrarily good one: with 1,000 zero-edge strategies over five years of daily
data, the best in-sample Sharpe is around 1.4 — and the true Sharpe of every one
of them is zero.

Three corrections live here, from Bailey & López de Prado:

* :func:`probabilistic_sharpe_ratio` — the probability that the true Sharpe
  exceeds a benchmark, given the sample length and the non-normality of the
  returns. Fat tails and negative skew make a Sharpe less trustworthy than its
  point estimate suggests, and this is where that shows up.
* :func:`expected_max_sharpe` — what the best of ``n`` trials looks like when
  none of them has any edge. The bar a real finding has to clear.
* :func:`deflated_sharpe_ratio` — the first evaluated against the second.

**Frequency discipline.** Every function takes a *return series*, never a
pre-computed Sharpe, because the single commonest error in this arithmetic is
mixing an annualised Sharpe with per-observation moments. Annualisation is a
presentation concern and is applied only where it is labelled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy import stats

EULER_MASCHERONI: Final = 0.5772156649015329
MIN_OBSERVATIONS: Final = 3
"""Below this, skew and kurtosis are not estimable and the corrections are noise."""


@dataclass(frozen=True, slots=True)
class SharpeEstimate:
    """A Sharpe ratio with everything needed to judge it."""

    sharpe: float
    """Per-observation Sharpe ratio."""
    observations: int
    skewness: float
    kurtosis: float
    """Pearson (non-excess) kurtosis: 3.0 for a normal distribution."""
    periods_per_year: float | None = None

    @property
    def annualised(self) -> float:
        """Sharpe scaled to a year. Requires ``periods_per_year`` to be known."""
        if self.periods_per_year is None:
            raise ValueError(
                "periods_per_year was not supplied; annualising without it would invent a frequency"
            )
        return self.sharpe * math.sqrt(self.periods_per_year)

    @property
    def has_fat_tails(self) -> bool:
        return self.kurtosis > 3.0


@dataclass(frozen=True, slots=True)
class DeflatedSharpe:
    """The verdict, with its inputs kept alongside it.

    Every field is here so an evidence card can show the arithmetic rather than
    asserting a conclusion.
    """

    deflated_sharpe: float
    """P(true Sharpe > the best a lucky no-edge search would have produced)."""
    observed_sharpe: float
    expected_max_sharpe: float
    n_trials: int
    trial_sharpe_variance: float
    observations: int
    skewness: float
    kurtosis: float

    @property
    def survives(self) -> bool:
        """True at the conventional 95% threshold.

        A convention, not a law of nature — report the number, not just the flag.
        """
        return self.deflated_sharpe > 0.95

    def explain(self) -> str:
        verdict = "SURVIVES" if self.survives else "DOES NOT SURVIVE"
        return (
            f"{verdict}: observed Sharpe {self.observed_sharpe:.4f} per period against a "
            f"selection bar of {self.expected_max_sharpe:.4f} implied by {self.n_trials} "
            f"trial(s) (dispersion {math.sqrt(self.trial_sharpe_variance):.4f}), over "
            f"{self.observations} observations with skew {self.skewness:+.2f} and "
            f"kurtosis {self.kurtosis:.2f} → deflated Sharpe {self.deflated_sharpe:.4f}"
        )


def sharpe_ratio(
    returns: NDArray[np.float64], *, periods_per_year: float | None = None, ddof: int = 1
) -> SharpeEstimate:
    """Per-observation Sharpe plus the moments the corrections need.

    ``returns`` are excess returns. Passing raw returns while a risk-free rate
    was non-trivial overstates the Sharpe, and no statistical correction can
    recover from that.
    """
    series = _validated(returns)
    std = float(np.std(series, ddof=ddof))
    if std == 0.0:
        raise ValueError("return series has zero variance; a Sharpe ratio is undefined")
    return SharpeEstimate(
        sharpe=float(np.mean(series)) / std,
        observations=series.size,
        skewness=float(stats.skew(series, bias=False)),
        kurtosis=float(stats.kurtosis(series, fisher=False, bias=False)),
        periods_per_year=periods_per_year,
    )


def probabilistic_sharpe_ratio(
    returns: NDArray[np.float64], *, benchmark_sharpe: float = 0.0
) -> float:
    """P(true Sharpe > ``benchmark_sharpe``), adjusted for skew and fat tails.

    Bailey & López de Prado (2012). ``benchmark_sharpe`` is per-observation, on
    the same footing as the sample.

    The denominator is where non-normality bites: negative skew and excess
    kurtosis both inflate the standard error of a Sharpe estimate, so the same
    point estimate is less believable from a series with crash risk than from a
    well-behaved one.
    """
    estimate = sharpe_ratio(returns)
    return _psr(estimate, benchmark_sharpe)


def expected_max_sharpe(
    *,
    n_trials: int,
    trial_sharpe_variance: float | None = None,
    trial_sharpes: NDArray[np.float64] | None = None,
) -> float:
    """Expected best Sharpe from ``n_trials`` strategies that all have zero edge.

    The false-discovery bar. Given as either the variance of the trial Sharpes or
    the trial Sharpes themselves; supply whichever you actually have, and never
    substitute a guess for a measurement without saying so.

    Uses the standard extreme-value approximation for the maximum of ``n``
    Gaussian draws (Bailey & López de Prado 2014). Grows like ``sqrt(2 ln n)``:
    slow, but unbounded — enough trials clear any fixed bar.
    """
    if n_trials < 2:
        raise ValueError(
            "expected_max_sharpe needs at least 2 trials; with a single trial there "
            "is no selection to correct for (and the formula diverges)"
        )
    if trial_sharpes is not None:
        variance = float(np.var(_validated(trial_sharpes), ddof=1))
    elif trial_sharpe_variance is not None:
        variance = float(trial_sharpe_variance)
    else:
        raise ValueError("supply either trial_sharpes or trial_sharpe_variance")
    if variance < 0:
        raise ValueError("trial_sharpe_variance cannot be negative")

    quantile_high = stats.norm.ppf(1.0 - 1.0 / n_trials)
    quantile_low = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(
        math.sqrt(variance)
        * ((1.0 - EULER_MASCHERONI) * quantile_high + EULER_MASCHERONI * quantile_low)
    )


def deflated_sharpe_ratio(
    returns: NDArray[np.float64],
    *,
    n_trials: int,
    trial_sharpe_variance: float | None = None,
    trial_sharpes: NDArray[np.float64] | None = None,
) -> DeflatedSharpe:
    """The observed Sharpe judged against what luck alone would have produced.

    ``n_trials`` must be the number of configurations actually evaluated —
    including the ones abandoned early, the parameter grid, the universes tried,
    and the sample periods considered. Understating it is not a conservative
    error; it inflates the result in exactly the direction the correction exists
    to remove. :class:`~trialkeeper.ledger.TrialLedger` exists so the count is a
    record rather than a recollection.
    """
    estimate = sharpe_ratio(returns)
    bar = expected_max_sharpe(
        n_trials=n_trials,
        trial_sharpe_variance=trial_sharpe_variance,
        trial_sharpes=trial_sharpes,
    )
    variance = (
        float(np.var(_validated(trial_sharpes), ddof=1))
        if trial_sharpes is not None
        else float(trial_sharpe_variance or 0.0)
    )
    return DeflatedSharpe(
        deflated_sharpe=_psr(estimate, bar),
        observed_sharpe=estimate.sharpe,
        expected_max_sharpe=bar,
        n_trials=n_trials,
        trial_sharpe_variance=variance,
        observations=estimate.observations,
        skewness=estimate.skewness,
        kurtosis=estimate.kurtosis,
    )


def minimum_track_record_length(
    returns: NDArray[np.float64], *, benchmark_sharpe: float = 0.0, confidence: float = 0.95
) -> float:
    """Observations needed before this Sharpe would be significant.

    Answers "how long must this run before I believe it". A track record shorter
    than the number returned here is not evidence, however good it looks.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    estimate = sharpe_ratio(returns)
    excess = estimate.sharpe - benchmark_sharpe
    if excess <= 0:
        return math.inf
    z = float(stats.norm.ppf(confidence))
    return 1.0 + _variance_factor(estimate) * (z / excess) ** 2


def _psr(estimate: SharpeEstimate, benchmark_sharpe: float) -> float:
    if estimate.observations < MIN_OBSERVATIONS:
        raise ValueError(
            f"need at least {MIN_OBSERVATIONS} observations to estimate skew and kurtosis; "
            f"got {estimate.observations}"
        )
    factor = _variance_factor(estimate)
    if factor <= 0:
        # Extreme sample moments can drive the estimated variance non-positive.
        # Returning a fabricated probability would be worse than refusing.
        raise ValueError(
            "the estimated Sharpe variance is non-positive; the sample moments are "
            "too extreme for this approximation to mean anything"
        )
    numerator = (estimate.sharpe - benchmark_sharpe) * math.sqrt(estimate.observations - 1)
    return float(stats.norm.cdf(numerator / math.sqrt(factor)))


def _variance_factor(estimate: SharpeEstimate) -> float:
    """``1 - γ₃·SR + (γ₄-1)/4·SR²`` — the non-normality penalty."""
    return (
        1.0
        - estimate.skewness * estimate.sharpe
        + (estimate.kurtosis - 1.0) / 4.0 * estimate.sharpe**2
    )


def _validated(values: NDArray[np.float64]) -> NDArray[np.float64]:
    series = np.asarray(values, dtype=np.float64).ravel()
    if series.size == 0:
        raise ValueError("empty series")
    if not np.all(np.isfinite(series)):
        raise ValueError(
            "series contains NaN or infinity; decide what a missing return means "
            "before computing a statistic over it"
        )
    return series
