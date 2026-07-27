"""Probability of backtest overfitting.

The question PBO answers is not "did this strategy work in-sample" but "does
picking the in-sample winner tell me anything about out-of-sample performance".

The procedure (Bailey, Borwein, López de Prado & Zhu, 2014): chop the return
history into ``S`` blocks, form every way of splitting them into equal in-sample
and out-of-sample halves, and for each split pick the strategy that looked best
in-sample. Then ask where that strategy ranks out-of-sample. If selection carries
information, the winner keeps ranking high. If it does not, it lands anywhere —
and half the time in the bottom half.

**PBO ≈ 0.5 means your selection process is worthless.** Not "the strategy is
mediocre" — the process. It would have done as well picking at random. That is a
different and more damaging finding than a low Sharpe, and it is invisible to any
single train/test split.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PBOResult:
    """PBO with the per-split detail that produced it."""

    pbo: float
    """Fraction of splits where the in-sample winner ranked in the bottom half OOS."""
    n_splits: int
    logits: NDArray[np.float64]
    """Logit of the OOS relative rank per split; negative means below median."""
    oos_ranks: NDArray[np.float64]
    """OOS relative rank of the in-sample winner, in (0, 1)."""
    median_oos_rank: float
    performance_degradation: float
    """Mean OOS performance of the chosen strategy minus its mean IS performance."""

    @property
    def is_overfit(self) -> bool:
        """True when selection is no better than chance. Convention: PBO > 0.5."""
        return self.pbo > 0.5

    def explain(self) -> str:
        verdict = (
            "selection carries no information"
            if self.is_overfit
            else "selection carries some information"
        )
        return (
            f"PBO {self.pbo:.1%} over {self.n_splits} splits — {verdict}. "
            f"The in-sample winner's median out-of-sample rank was "
            f"{self.median_oos_rank:.1%}; mean performance degraded by "
            f"{self.performance_degradation:+.4f} per period."
        )


def probability_of_backtest_overfitting(
    returns: NDArray[np.float64],
    *,
    n_blocks: int = 10,
    performance: str = "sharpe",
) -> PBOResult:
    """Run CSCV over a matrix of strategy returns.

    ``returns`` has shape ``(observations, strategies)`` — one column per
    configuration tried, all over the same period. At least two strategies are
    required: with one there is no selection, and PBO is a statement about
    selection.

    ``n_blocks`` must be even. It sets the number of splits to
    ``C(n_blocks, n_blocks/2)`` — 252 for the default 10, which is enough for a
    stable estimate without becoming slow.
    """
    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("returns must be a 2-D array of shape (observations, strategies)")
    n_observations, n_strategies = matrix.shape
    if n_strategies < 2:
        raise ValueError(
            "PBO measures whether picking a winner is informative; that needs at "
            "least 2 candidate strategies"
        )
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even so the halves are equal")
    if n_observations < n_blocks * 2:
        raise ValueError(
            f"need at least {n_blocks * 2} observations for {n_blocks} blocks; got {n_observations}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("returns contain NaN or infinity")

    score = _scorer(performance)
    blocks = np.array_split(np.arange(n_observations), n_blocks)
    half = n_blocks // 2

    logits: list[float] = []
    ranks: list[float] = []
    degradations: list[float] = []

    for chosen in combinations(range(n_blocks), half):
        in_sample_rows = np.concatenate([blocks[index] for index in chosen])
        out_rows = np.concatenate(
            [blocks[index] for index in range(n_blocks) if index not in chosen]
        )

        in_sample = score(matrix[in_sample_rows])
        out_sample = score(matrix[out_rows])
        if not np.all(np.isfinite(in_sample)) or not np.all(np.isfinite(out_sample)):
            continue  # a block with zero variance for some strategy; skip the split

        winner = int(np.argmax(in_sample))
        # Relative rank of the winner among OOS results, mapped into (0, 1) so the
        # logit is finite at both extremes.
        order = np.argsort(np.argsort(out_sample))
        relative_rank = (order[winner] + 1.0) / (n_strategies + 1.0)
        ranks.append(float(relative_rank))
        logits.append(float(math.log(relative_rank / (1.0 - relative_rank))))
        degradations.append(float(out_sample[winner] - in_sample[winner]))

    if not logits:
        raise ValueError("no usable splits — every split had a degenerate (zero-variance) strategy")

    logit_array = np.asarray(logits, dtype=np.float64)
    return PBOResult(
        pbo=float(np.mean(logit_array < 0.0)),
        n_splits=logit_array.size,
        logits=logit_array,
        oos_ranks=np.asarray(ranks, dtype=np.float64),
        median_oos_rank=float(np.median(ranks)),
        performance_degradation=float(np.mean(degradations)),
    )


def _scorer(name: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    if name == "sharpe":

        def sharpe(block: NDArray[np.float64]) -> NDArray[np.float64]:
            std = np.std(block, axis=0, ddof=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(std > 0, np.mean(block, axis=0) / std, np.nan)

        return sharpe

    if name == "mean":
        return lambda block: np.mean(block, axis=0)

    raise ValueError(f"unknown performance measure: {name!r} (want 'sharpe' or 'mean')")
