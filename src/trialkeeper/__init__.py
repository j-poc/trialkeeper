"""trialkeeper — honest evaluation of backtests.

The most important number in any backtest is the one almost nobody reports: how
many strategies were tried before this one was shown to you. Without it a Sharpe
ratio carries no information, because selecting the best of enough random
strategies produces an arbitrarily good one.

This library supplies the corrections that make a reported Sharpe interpretable:

* :func:`deflated_sharpe_ratio` — Bailey & López de Prado (2014), adjusting for
  the number of trials, the dispersion of their outcomes, and non-normal returns.
* :func:`probability_of_backtest_overfitting` — PBO via combinatorially purged
  cross-validation.
* :func:`purged_kfold` / :func:`combinatorial_purged_splits` — cross-validation
  that removes observations whose label windows overlap the test set, with an
  embargo, so overlapping labels cannot leak.
* :class:`TrialLedger` — an append-only, hash-chained record written *before*
  each run, so the trial count used by the corrections above is evidence rather
  than a self-report.

Deliberately dependency-light (numpy + scipy) and free of any coupling to the
research system it was extracted from.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
