"""Cross-validation that does not leak through overlapping labels.

Standard k-fold assumes observations are independent. Financial labels are not:
a label formed over a five-day forward window shares four days with its
neighbour. Split naively and the training set contains the answer to the test
set — the model scores well and the strategy fails live.

Two corrections, from López de Prado's *Advances in Financial Machine Learning*:

* **Purging** — drop training observations whose label window overlaps any test
  observation's label window.
* **Embargo** — additionally drop training observations immediately *after* the
  test set, because serial correlation leaks in that direction even without
  overlap.

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
    """Training observations removed for overlapping the test labels."""
    embargoed: int
    """Training observations removed for sitting just after the test set."""

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
        # Purge backwards: a training label starting up to `label_horizon` before
        # a test observation still overlaps it.
        low = max(0, int(position) - label_horizon)
        purged[low : int(position)] = True
        # Embargo forwards: serial correlation leaks into the immediate future
        # even where label windows do not overlap.
        high = min(indices.size, int(position) + 1 + embargo)
        embargoed[int(position) + 1 : high] = True

    removed = (purged | embargoed) & ~blocked
    train = indices[~blocked & ~removed]
    return Split(
        train=train,
        test=test_set,
        purged=int((purged & ~blocked & ~embargoed).sum()),
        embargoed=int((embargoed & ~blocked).sum()),
    )
