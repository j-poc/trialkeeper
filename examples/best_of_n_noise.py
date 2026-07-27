"""How good a backtest can look with no edge at all.

Generates strategies whose true expected return is exactly zero, picks the best
one, and reports its Sharpe ratio. This is the number a backtest reports when
nothing is there.

Run: ``python examples/best_of_n_noise.py``
"""

from __future__ import annotations

import numpy as np

from trialkeeper import deflated_sharpe_ratio, sharpe_ratio

SEED = 20260727
OBSERVATIONS = 1260  # five years of daily data
DAILY_VOL = 0.01


def main() -> None:
    rng = np.random.default_rng(SEED)

    print(f"{'trials':>8}  {'best annualised Sharpe':>22}  {'true Sharpe':>11}  {'deflated':>9}")
    print("-" * 58)
    for n_trials in (1, 10, 100, 1_000, 10_000):
        panel = rng.normal(0.0, DAILY_VOL, size=(OBSERVATIONS, n_trials))
        sharpes = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
        winner = panel[:, int(np.argmax(sharpes))]
        annualised = sharpe_ratio(winner, periods_per_year=252).annualised

        if n_trials == 1:
            verdict = "     n/a"  # nothing was selected, so nothing to deflate
        else:
            deflated = deflated_sharpe_ratio(winner, n_trials=n_trials, trial_sharpes=sharpes)
            verdict = f"{deflated.deflated_sharpe:>9.3f}"

        print(f"{n_trials:>8,}  {annualised:>22.2f}  {0.0:>11.2f}  {verdict}")

    print("-" * 58)
    print("Every strategy above has zero expected return. The Sharpe is pure selection.")
    print("The deflated column is what you are entitled to claim once that is priced in.")


if __name__ == "__main__":
    main()
