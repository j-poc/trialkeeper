"""PBO and purged cross-validation.

PBO is validated against three constructed regimes where the right answer is
known in advance:

* pure noise, where selection is worthless and PBO must be near 0.5;
* a genuine persistent edge, where selection works and PBO must be near 0;
* a strategy engineered to fit the first half and fail the second, where
  selection is actively misleading and PBO must be high.

A statistic that only ever returns "fine" is indistinguishable from a broken one,
so all three directions are asserted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trialkeeper.cv import (
    combinatorial_purged_splits,
    n_combinatorial_splits,
    purged_kfold,
)
from trialkeeper.pbo import probability_of_backtest_overfitting

SEED = 20260727
T = 1200


class TestPBODiagnostic:
    def test_pure_noise_gives_a_coin_flip(self) -> None:
        """No strategy has an edge, so picking the in-sample winner tells you nothing."""
        rng = np.random.default_rng(SEED)
        panel = rng.normal(0.0, 0.01, size=(T, 20))
        result = probability_of_backtest_overfitting(panel, n_blocks=10)
        assert result.pbo == pytest.approx(0.5, abs=0.15), result.explain()
        assert result.n_splits == 252  # C(10, 5)

    def test_a_genuine_edge_is_recognised(self) -> None:
        """Control: one column really is better, and selection finds it every time."""
        rng = np.random.default_rng(SEED)
        panel = rng.normal(0.0, 0.01, size=(T, 20))
        panel[:, 7] += 0.004  # a large, persistent, real edge
        result = probability_of_backtest_overfitting(panel, n_blocks=10)
        assert result.pbo < 0.05, result.explain()
        assert not result.is_overfit

    def test_a_strategy_fitted_to_the_first_half_is_exposed(self) -> None:
        """The failure mode PBO exists to catch.

        Each column gets a strong edge in the first half and the opposite in the
        second — the shape of a parameter set tuned on early data. A single
        train-then-test split would look fine or catastrophic depending on where
        it happened to cut; PBO sees the pattern either way.

        The bar is the *measured* noise baseline from the same generator rather
        than a guessed constant. CSCV mixes blocks from both halves into most
        splits, which dilutes any regime that is purely calendar-ordered, so the
        meaningful claim is "materially worse than chance", not a round number.
        """
        rng = np.random.default_rng(SEED)
        noise = rng.normal(0.0, 0.01, size=(T, 20))
        baseline = probability_of_backtest_overfitting(noise, n_blocks=10)

        fitted = noise.copy()
        half = T // 2
        for column in range(20):
            edge = 0.002 * (1 + column / 20)
            fitted[:half, column] += edge
            fitted[half:, column] -= edge
        result = probability_of_backtest_overfitting(fitted, n_blocks=10)

        assert result.is_overfit, result.explain()
        assert result.pbo > baseline.pbo + 0.10, (
            f"fitted {result.pbo:.1%} vs noise baseline {baseline.pbo:.1%}"
        )
        assert result.median_oos_rank < 0.5, result.explain()

    def test_performance_degradation_is_reported(self) -> None:
        rng = np.random.default_rng(SEED)
        panel = rng.normal(0.0, 0.01, size=(T, 20))
        result = probability_of_backtest_overfitting(panel, n_blocks=10)
        assert result.performance_degradation < 0, (
            "the in-sample winner should on average do worse out of sample"
        )

    def test_a_single_strategy_is_refused(self) -> None:
        """PBO is about selection; with nothing to select from there is no question."""
        rng = np.random.default_rng(SEED)
        with pytest.raises(ValueError, match="at least 2 candidate"):
            probability_of_backtest_overfitting(rng.normal(size=(T, 1)))

    def test_odd_block_counts_are_refused(self) -> None:
        rng = np.random.default_rng(SEED)
        with pytest.raises(ValueError, match="even"):
            probability_of_backtest_overfitting(rng.normal(size=(T, 5)), n_blocks=7)

    def test_too_little_data_is_refused(self) -> None:
        rng = np.random.default_rng(SEED)
        with pytest.raises(ValueError, match="need at least"):
            probability_of_backtest_overfitting(rng.normal(size=(10, 5)), n_blocks=10)


class TestPurgedKFold:
    def test_without_overlap_every_observation_is_used(self) -> None:
        """Control: with no label horizon and no embargo, nothing is dropped."""
        for split in purged_kfold(100, n_splits=5):
            assert split.train_size + split.test_size == 100
            assert split.purged == 0
            assert split.embargoed == 0

    def test_overlapping_labels_are_purged_from_training(self) -> None:
        """A five-period forward label shares four periods with its neighbour."""
        splits = list(purged_kfold(100, n_splits=5, label_horizon=5))
        middle = splits[2]
        first_test = int(middle.test.min())
        leaked = [index for index in middle.train if first_test - 5 <= index < first_test]
        assert leaked == []
        assert middle.purged > 0

    def test_the_embargo_removes_observations_after_the_test_block(self) -> None:
        splits = list(purged_kfold(100, n_splits=5, embargo=3))
        middle = splits[2]
        last_test = int(middle.test.max())
        leaked = [index for index in middle.train if last_test < index <= last_test + 3]
        assert leaked == []
        assert middle.embargoed > 0

    def test_labels_after_the_test_block_are_purged_too(self) -> None:
        """The direction the first version of this module missed.

        Overlap is symmetric: if a training label window reaches into the test
        block, it leaks regardless of which side it sits on. The original purge
        only looked backwards and left the forward side to ``embargo``, which
        defaults to 0 -- so the documented guarantee ("any test observation's
        label window") did not hold on the default configuration.

        The first fold is the sharpest case: its test block starts at index 0,
        so there is no history to purge and the backward pass removes literally
        nothing, while training rows immediately after the block still overlap.
        """
        first = next(iter(purged_kfold(100, n_splits=5, label_horizon=5, embargo=0)))
        last_test = int(first.test.max())
        leaked = [index for index in first.train if last_test < index < last_test + 5]
        assert leaked == [], f"training rows {leaked} overlap the test labels"

    def test_no_training_label_window_overlaps_any_test_label_window(self) -> None:
        """The guarantee stated as the docstring states it, checked directly.

        Rather than probing one boundary, this recomputes the overlap relation
        from the definition -- label window ``[i, i + horizon)`` -- over every
        fold and asserts the intersection is empty. A boundary-specific test
        passes when only one side is implemented; this one cannot.
        """
        horizon = 5
        for split in purged_kfold(100, n_splits=5, label_horizon=horizon, embargo=0):
            test_set = split.test.tolist()
            leaked = [
                index
                for index in split.train.tolist()
                if any(abs(int(index) - int(position)) < horizon for position in test_set)
            ]
            assert leaked == [], f"training rows {leaked} overlap test {test_set[:3]}..."

    def test_combinatorial_splits_purge_forward_as_well(self) -> None:
        """Same hole, checked on the CPCV path that the flagship study would use."""
        horizon = 10
        for split in combinatorial_purged_splits(
            600, n_groups=6, n_test_groups=2, label_horizon=horizon
        ):
            test_set = set(split.test.tolist())
            leaked = [
                index
                for index in split.train.tolist()
                if any(abs(int(index) - int(position)) < horizon for position in test_set)
            ]
            assert leaked == [], f"training rows {leaked[:5]} overlap the test labels"

    def test_folds_are_contiguous_in_time(self) -> None:
        """Shuffling a time series trains on the future by construction."""
        for split in purged_kfold(100, n_splits=5):
            assert np.all(np.diff(split.test) == 1)

    def test_train_and_test_never_intersect(self) -> None:
        for split in purged_kfold(100, n_splits=5, label_horizon=5, embargo=2):
            assert set(split.train.tolist()).isdisjoint(split.test.tolist())

    def test_degenerate_configurations_are_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            list(purged_kfold(100, n_splits=1))
        with pytest.raises(ValueError, match="cannot make"):
            list(purged_kfold(3, n_splits=5))


class TestCombinatorialPurgedSplits:
    def test_yields_every_combination_of_test_groups(self) -> None:
        splits = list(combinatorial_purged_splits(600, n_groups=6, n_test_groups=2))
        assert len(splits) == math.comb(6, 2) == 15
        assert n_combinatorial_splits(n_groups=6, n_test_groups=2) == 15

    def test_each_split_tests_the_expected_share_of_the_data(self) -> None:
        for split in combinatorial_purged_splits(600, n_groups=6, n_test_groups=2):
            assert split.test_size == pytest.approx(200, abs=6)

    def test_purging_applies_to_combinatorial_splits_too(self) -> None:
        for split in combinatorial_purged_splits(
            600, n_groups=6, n_test_groups=2, label_horizon=10
        ):
            for boundary in _block_starts(split.test):
                leaked = [index for index in split.train if boundary - 10 <= index < boundary]
                assert leaked == []

    def test_non_adjacent_test_groups_produce_disjoint_test_blocks(self) -> None:
        """The point of CPCV: many distinct paths, not one contiguous holdout."""
        splits = list(combinatorial_purged_splits(600, n_groups=6, n_test_groups=2))
        assert any(len(_block_starts(split.test)) == 2 for split in splits)

    def test_invalid_group_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="fewer than n_groups"):
            list(combinatorial_purged_splits(600, n_groups=4, n_test_groups=4))


def _block_starts(indices: np.ndarray) -> list[int]:
    """First index of each contiguous run."""
    if indices.size == 0:
        return []
    breaks = np.where(np.diff(indices) > 1)[0] + 1
    return [int(indices[0]), *[int(indices[position]) for position in breaks]]
