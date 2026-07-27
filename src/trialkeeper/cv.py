"""Cross-validation that does not leak through overlapping labels.

Standard k-fold assumes observations are independent. Financial labels are not:
a label formed over a five-day forward window shares four days with its
neighbour. Split naively and the training set contains the answer to the test
set — the model scores well and the strategy fails live.

Two corrections, from López de Prado's *Advances in Financial Machine Learning*:

* **Purging** — drop training observations whose label window overlaps any test
  observation's label window, on *either* side of the test block. Overlap is
  symmetric; a label reaching backwards into the test set leaks exactly as much
  as one reaching forwards out of it.
* **Embargo** — additionally drop training observations immediately *after* the
  test set, because serial correlation leaks in that direction even without
  overlap. This is a margin *beyond* the purge, not the mechanism that handles
  the forward direction — purging already covers overlap both ways, so a correct
  ``embargo=0`` run leaks nothing.

:func:`combinatorial_purged_splits` extends this to CPCV: instead of one path
through the data, test on every combination of ``k`` groups, producing many
backtest paths from one dataset. The dispersion across those paths is itself the
finding — a strategy whose result depends on which slice you looked at has not
been shown to work.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test partition, after purging and embargo."""

    train: NDArray[np.int64]
    test: NDArray[np.int64]
    purged: int
    """Training observations removed for overlapping the test labels, either side."""
    embargoed: int
    """Training observations removed by the embargo *alone*.

    Rows the purge had already taken are not counted again here, so an embargo
    shorter than the label horizon reports 0 — it genuinely added nothing.
    """

    @property
    def train_size(self) -> int:
        return int(self.train.size)

    @property
    def test_size(self) -> int:
        return int(self.test.size)


def purged_kfold(
    n_samples: int,
    *,
    n_splits: int = 5,
    label_horizon: int = 0,
    embargo: int = 0,
) -> Iterator[Split]:
    """K-fold over contiguous time blocks, purged and embargoed.

    ``label_horizon`` is how many observations forward each label reaches: 0 for
    a label known at the same observation, 5 for a five-period forward return.
    ``embargo`` is extra observations dropped after each test block.

    Folds are contiguous, never shuffled — shuffling time series data trains on
    the future by construction.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n_samples < n_splits:
        raise ValueError(f"cannot make {n_splits} folds from {n_samples} observations")
    if label_horizon < 0 or embargo < 0:
        raise ValueError("label_horizon and embargo must be non-negative")

    indices = np.arange(n_samples, dtype=np.int64)
    for test_index in np.array_split(indices, n_splits):
        yield _build_split(indices, test_index, label_horizon=label_horizon, embargo=embargo)


def combinatorial_purged_splits(
    n_samples: int,
    *,
    n_groups: int = 6,
    n_test_groups: int = 2,
    label_horizon: int = 0,
    embargo: int = 0,
) -> Iterator[Split]:
    """CPCV: every combination of ``n_test_groups`` out of ``n_groups`` as the test set.

    Yields ``C(n_groups, n_test_groups)`` splits — 15 for the default 6-choose-2
    — each a distinct path through history. One path is an anecdote; the spread
    across paths is the evidence.
    """
    if n_groups < 2:
        raise ValueError("n_groups must be at least 2")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be at least 1 and fewer than n_groups")
    if n_samples < n_groups:
        raise ValueError(f"cannot make {n_groups} groups from {n_samples} observations")

    indices = np.arange(n_samples, dtype=np.int64)
    groups = np.array_split(indices, n_groups)
    for chosen in combinations(range(n_groups), n_test_groups):
        test_index = np.concatenate([groups[position] for position in chosen])
        yield _build_split(indices, test_index, label_horizon=label_horizon, embargo=embargo)


def n_combinatorial_splits(*, n_groups: int, n_test_groups: int) -> int:
    """How many splits :func:`combinatorial_purged_splits` will yield."""
    return len(list(combinations(range(n_groups), n_test_groups)))


def _build_split(
    indices: NDArray[np.int64],
    test_index: NDArray[np.int64],
    *,
    label_horizon: int,
    embargo: int,
) -> Split:
    test_set = np.sort(test_index)
    blocked = np.zeros(indices.size, dtype=bool)
    blocked[test_set] = True

    purged = np.zeros(indices.size, dtype=bool)
    embargoed = np.zeros(indices.size, dtype=bool)
    for position in test_set:
        # Purge in BOTH directions. Overlap is a symmetric relation: with a label
        # window of `[i, i + label_horizon)`, observations i and p overlap
        # whenever `abs(i - p) < label_horizon`, and it makes no difference which
        # of the two came first. An earlier version purged only backwards and
        # left the forward side to `embargo` -- which defaults to 0, so on the
        # default configuration training rows immediately after a test block kept
        # their overlap and leaked. The first fold showed it most clearly: its
        # test block starts at index 0, so the backward pass had nothing to
        # remove and reported `purged=0` while four rows leaked forward.
        #
        # The window below is `[p - h, p + h]`, one wider on each side than the
        # strict `abs(i - p) < h` rule. That is deliberate and it is what the
        # backward pass already did: it treats a label as touching both of its
        # endpoints, which is the right reading when the label is a return
        # measured from the price at `i` to the price at `i + h`.
        low = max(0, int(position) - label_horizon)
        high = min(indices.size, int(position) + label_horizon + 1)
        purged[low:high] = True
        # Embargo forwards: serial correlation leaks into the immediate future
        # even where label windows do not overlap.
        edge = min(indices.size, int(position) + 1 + embargo)
        embargoed[int(position) + 1 : edge] = True

    removed = (purged | embargoed) & ~blocked
    train = indices[~blocked & ~removed]
    # Purge is primary and the embargo is the additional margin beyond it, so the
    # counts are reported that way: `embargoed` is rows dropped for the embargo
    # ALONE. An embargo smaller than the label horizon therefore reports 0, which
    # is the honest answer -- it removed nothing the purge had not already taken.
    return Split(
        train=train,
        test=test_set,
        purged=int((purged & ~blocked).sum()),
        embargoed=int((embargoed & ~blocked & ~purged).sum()),
    )
