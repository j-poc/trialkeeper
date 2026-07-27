"""Sharpe corrections, validated against Monte Carlo ground truth.

The corrections here are not checked against numbers copied out of a paper —
they are checked against simulation. Generate strategies that provably have no
edge, measure what the best of them looks like, and require the closed form to
predict it. That is a stronger test than matching a printed table, because it
would catch an error in the paper as readily as an error in the code.

Every simulation uses a fixed seed, so a failure is reproducible rather than
occasional.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from trialkeeper.sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

SEED = 20260727
T = 1260  # five years of daily observations


class TestSharpeEstimate:
    def test_matches_the_definition(self) -> None:
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        estimate = sharpe_ratio(returns)
        expected = float(np.mean(returns) / np.std(returns, ddof=1))
        assert estimate.sharpe == pytest.approx(expected)

    def test_annualisation_requires_a_declared_frequency(self) -> None:
        """Refusing beats guessing: the wrong factor silently scales every claim."""
        estimate = sharpe_ratio(np.array([0.01, 0.02, -0.01, 0.03]))
        with pytest.raises(ValueError, match="invent a frequency"):
            _ = estimate.annualised

    def test_annualises_by_the_square_root_of_the_frequency(self) -> None:
        returns = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
        estimate = sharpe_ratio(returns, periods_per_year=252)
        assert estimate.annualised == pytest.approx(estimate.sharpe * math.sqrt(252))

    def test_a_flat_series_has_no_sharpe_ratio(self) -> None:
        with pytest.raises(ValueError, match="zero variance"):
            sharpe_ratio(np.zeros(10))

    def test_nan_is_refused_rather_than_ignored(self) -> None:
        """np.nanmean would quietly answer a different question."""
        with pytest.raises(ValueError, match="NaN or infinity"):
            sharpe_ratio(np.array([0.01, np.nan, 0.02]))


class TestExpectedMaxSharpe:
    def test_predicts_the_simulated_maximum_of_zero_edge_trials(self) -> None:
        """The core validation: closed form vs. brute-force simulation.

        200 replications, each searching 50 strategies with exactly zero true
        edge. The formula must predict the average best-of-50 Sharpe.
        """
        rng = np.random.default_rng(SEED)
        n_trials = 50
        maxima: list[float] = []
        dispersions: list[float] = []
        for _ in range(200):
            panel = rng.normal(0.0, 0.01, size=(T, n_trials))
            sharpes = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
            maxima.append(float(sharpes.max()))
            dispersions.append(float(np.var(sharpes, ddof=1)))

        predicted = expected_max_sharpe(
            n_trials=n_trials, trial_sharpe_variance=float(np.mean(dispersions))
        )
        simulated = float(np.mean(maxima))
        assert predicted == pytest.approx(simulated, rel=0.06), (
            f"closed form {predicted:.5f} vs simulated {simulated:.5f}"
        )

    def test_the_bar_rises_with_the_number_of_trials(self) -> None:
        bar_10 = expected_max_sharpe(n_trials=10, trial_sharpe_variance=0.25)
        bar_1000 = expected_max_sharpe(n_trials=1000, trial_sharpe_variance=0.25)
        assert bar_1000 > bar_10 > 0

    def test_growth_is_the_square_root_of_the_log_of_the_trial_count(self) -> None:
        """sqrt(2 ln n): slow, but unbounded. Enough trials clear any fixed bar."""
        ratio = expected_max_sharpe(
            n_trials=10_000, trial_sharpe_variance=1.0
        ) / expected_max_sharpe(n_trials=100, trial_sharpe_variance=1.0)
        assert ratio == pytest.approx(
            math.sqrt(2 * math.log(10_000)) / math.sqrt(2 * math.log(100)), rel=0.10
        )

    def test_identical_trials_leave_no_selection_bar(self) -> None:
        """Zero dispersion means the search had nothing to choose between."""
        assert expected_max_sharpe(n_trials=500, trial_sharpe_variance=0.0) == 0.0

    def test_a_single_trial_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2 trials"):
            expected_max_sharpe(n_trials=1, trial_sharpe_variance=0.25)

    def test_accepts_the_trial_sharpes_directly(self) -> None:
        sharpes = np.array([0.1, 0.4, -0.2, 0.3, 0.0])
        from_values = expected_max_sharpe(n_trials=5, trial_sharpes=sharpes)
        from_variance = expected_max_sharpe(
            n_trials=5, trial_sharpe_variance=float(np.var(sharpes, ddof=1))
        )
        assert from_values == pytest.approx(from_variance)


class TestProbabilisticSharpe:
    def test_a_sample_sharpe_of_exactly_zero_is_a_coin_flip(self) -> None:
        """PSR is a statement about the sample in hand, not the process that made it.

        A series whose *sample* Sharpe is zero gives 0.5 exactly. Note that a
        zero-mean *process* does not: over 100,000 observations even a tiny
        sampling fluctuation is statistically distinguishable from zero, which is
        the correct answer to the question PSR asks.
        """
        rng = np.random.default_rng(SEED)
        returns = _rescale(rng.normal(size=5_000), target_sharpe=0.0)
        assert probabilistic_sharpe_ratio(returns) == pytest.approx(0.5, abs=1e-9)

    def test_fat_tails_reduce_confidence_at_the_same_sharpe(self) -> None:
        """Two series, identical point estimate; the one with crash risk is weaker.

        Both are symmetrised by mirroring, so skewness is exactly zero in each and
        kurtosis is the only thing that differs. Any change in PSR is therefore
        attributable to tail weight alone.

        The sample is deliberately short. Over a long enough record any positive
        Sharpe becomes overwhelming and PSR saturates at 1.0 for both series —
        true, but it would make this test blind to the effect it is checking.
        """
        rng = np.random.default_rng(SEED)
        normal = _rescale(_symmetrise(rng.normal(size=40)), target_sharpe=0.25)
        heavy = _rescale(_symmetrise(rng.standard_t(df=4, size=40)), target_sharpe=0.25)

        assert sharpe_ratio(normal).sharpe == pytest.approx(sharpe_ratio(heavy).sharpe, abs=1e-9)
        assert sharpe_ratio(normal).skewness == pytest.approx(0.0, abs=1e-9)
        assert sharpe_ratio(heavy).skewness == pytest.approx(0.0, abs=1e-9)
        assert sharpe_ratio(heavy).kurtosis > sharpe_ratio(normal).kurtosis
        assert probabilistic_sharpe_ratio(heavy) < probabilistic_sharpe_ratio(normal)

    def test_negative_skew_reduces_confidence(self) -> None:
        """Picking up pennies in front of a steamroller looks fine until it does not.

        A series and its mirror image have identical kurtosis and opposite skew,
        so skewness is isolated exactly as tail weight was above.
        """
        rng = np.random.default_rng(SEED)
        raw = rng.gumbel(size=200)
        positive = _rescale(raw, target_sharpe=0.15)
        negative = _rescale(-raw, target_sharpe=0.15)

        assert sharpe_ratio(negative).skewness == pytest.approx(
            -sharpe_ratio(positive).skewness, abs=1e-9
        )
        assert sharpe_ratio(negative).kurtosis == pytest.approx(
            sharpe_ratio(positive).kurtosis, abs=1e-9
        )
        assert sharpe_ratio(negative).skewness < -0.5
        assert probabilistic_sharpe_ratio(negative) < probabilistic_sharpe_ratio(positive)

    def test_a_longer_track_record_raises_confidence(self) -> None:
        rng = np.random.default_rng(SEED)
        long_run = _rescale(rng.normal(size=5000), target_sharpe=0.05)
        short_run = _rescale(rng.normal(size=250), target_sharpe=0.05)
        assert probabilistic_sharpe_ratio(long_run) > probabilistic_sharpe_ratio(short_run)


class TestDeflatedSharpe:
    def test_a_real_edge_found_in_one_look_survives(self) -> None:
        rng = np.random.default_rng(SEED)
        returns = rng.normal(0.0008, 0.01, size=T)  # genuine drift
        result = deflated_sharpe_ratio(returns, n_trials=2, trial_sharpe_variance=0.001)
        assert result.survives
        assert "SURVIVES" in result.explain()

    def test_the_best_of_a_thousand_noise_strategies_does_not(self) -> None:
        """The headline case: a 1.4 Sharpe built entirely from selection.

        Every column has exactly zero expected return. The best of them looks
        tradeable; the deflated Sharpe says it is not.
        """
        rng = np.random.default_rng(SEED)
        panel = rng.normal(0.0, 0.01, size=(T, 1000))
        sharpes = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
        winner = panel[:, int(np.argmax(sharpes))]

        annualised = sharpe_ratio(winner, periods_per_year=252).annualised
        assert annualised > 1.0, "control: the winner must look good before deflation"

        undeflated = probabilistic_sharpe_ratio(winner)
        result = deflated_sharpe_ratio(winner, n_trials=1000, trial_sharpes=sharpes)

        assert undeflated > 0.99, "against a zero benchmark it looks overwhelming"
        assert not result.survives, "against the selection bar it does not"
        assert result.deflated_sharpe < undeflated - 0.3, (
            f"deflation must move the verdict materially: "
            f"{undeflated:.3f} → {result.deflated_sharpe:.3f}"
        )

    def test_hiding_the_trial_count_manufactures_significance(self) -> None:
        """Why the ledger exists: under-reporting trials inflates the verdict.

        The same return series, judged as a single idea versus as the winner of a
        thousand-strategy search. Only the second is honest.
        """
        rng = np.random.default_rng(SEED)
        panel = rng.normal(0.0, 0.01, size=(T, 1000))
        sharpes = panel.mean(axis=0) / panel.std(axis=0, ddof=1)
        winner = panel[:, int(np.argmax(sharpes))]

        understated = deflated_sharpe_ratio(winner, n_trials=2, trial_sharpes=sharpes[:2])
        honest = deflated_sharpe_ratio(winner, n_trials=1000, trial_sharpes=sharpes)
        assert understated.deflated_sharpe > honest.deflated_sharpe
        assert understated.survives and not honest.survives

    def test_the_verdict_carries_its_inputs(self) -> None:
        """An evidence card must be able to show the arithmetic, not assert it."""
        rng = np.random.default_rng(SEED)
        result = deflated_sharpe_ratio(
            rng.normal(0.0005, 0.01, size=T), n_trials=25, trial_sharpe_variance=0.002
        )
        assert result.n_trials == 25
        assert result.observations == T
        assert result.expected_max_sharpe > 0
        assert str(result.n_trials) in result.explain()


class TestMinimumTrackRecordLength:
    def test_a_weaker_edge_needs_a_longer_record(self) -> None:
        rng = np.random.default_rng(SEED)
        strong = _rescale(rng.normal(size=T), target_sharpe=0.10)
        weak = _rescale(rng.normal(size=T), target_sharpe=0.02)
        assert minimum_track_record_length(weak) > minimum_track_record_length(strong)

    def test_no_edge_can_never_be_proven(self) -> None:
        rng = np.random.default_rng(SEED)
        flat = _rescale(rng.normal(size=T), target_sharpe=-0.01)
        assert minimum_track_record_length(flat) == math.inf

    def test_a_higher_confidence_demands_more_data(self) -> None:
        rng = np.random.default_rng(SEED)
        returns = _rescale(rng.normal(size=T), target_sharpe=0.05)
        assert minimum_track_record_length(returns, confidence=0.99) > (
            minimum_track_record_length(returns, confidence=0.90)
        )

    def test_matches_the_closed_form_for_normal_returns(self) -> None:
        """Under normality the expression collapses to 1 + (z/SR)^2·(1 + SR²/2)."""
        rng = np.random.default_rng(SEED)
        returns = _rescale(rng.normal(size=200_000), target_sharpe=0.05)
        estimate = sharpe_ratio(returns)
        z = float(stats.norm.ppf(0.95))
        expected = 1.0 + (1.0 + estimate.sharpe**2 / 2.0) * (z / estimate.sharpe) ** 2
        assert minimum_track_record_length(returns) == pytest.approx(expected, rel=0.02)


def _rescale(values: np.ndarray, *, target_sharpe: float) -> np.ndarray:
    """Standardise, then shift to an exact per-observation Sharpe.

    Shifting does not change the standard deviation, so after standardising to
    unit variance the mean *is* the Sharpe ratio. This lets two series differ
    only in shape, so a test can attribute a change in the correction to skew or
    kurtosis rather than to the point estimate.
    """
    standardised = (values - values.mean()) / values.std(ddof=1)
    return standardised + target_sharpe


def _symmetrise(values: np.ndarray) -> np.ndarray:
    """Mirror a sample so its skewness is exactly zero, leaving kurtosis intact.

    Isolates tail weight from asymmetry, so a test can attribute a change in the
    correction to one moment rather than to their combination.
    """
    return np.concatenate([values, -values])
