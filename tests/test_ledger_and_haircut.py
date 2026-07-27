"""The trial ledger and multiple-testing haircuts.

The ledger's only real job is to make deletion visible. Removing a losing trial
is still possible — it is a text file — but it can no longer be done without
breaking the chain, and the break points at the exact entry. Both directions are
tested: an untouched ledger must verify, and every way of tampering with one must
be caught.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trialkeeper.ledger import TrialLedger
from trialkeeper.multiple_testing import adjust_pvalues, sharpe_haircut

REGISTERED_AT = "2026-07-27T12:00:00+00:00"


@pytest.fixture
def ledger(tmp_path: Path) -> TrialLedger:
    return TrialLedger(tmp_path / "trials.jsonl")


def _register(ledger: TrialLedger, n: int, *, family: str = "momentum") -> None:
    for index in range(n):
        ledger.register(
            hypothesis=f"{family} lookback {index}",
            family=family,
            config={"lookback": index},
            registered_at=REGISTERED_AT,
        )


class TestTrialLedger:
    def test_an_untouched_ledger_verifies(self, ledger: TrialLedger) -> None:
        """Control: the detector must not cry wolf."""
        _register(ledger, 5)
        result = ledger.verify()
        assert result.intact
        assert result.n_entries == 5
        assert "intact" in result.explain()

    def test_registration_is_required_before_a_run(self, ledger: TrialLedger) -> None:
        trial = ledger.register(
            hypothesis="restatement magnitude predicts returns",
            family="restatements",
            config={"window": 60},
            registered_at=REGISTERED_AT,
        )
        assert trial.sequence == 1
        assert trial.chain_head != "0" * 64
        assert ledger.count() == 1

    def test_an_empty_hypothesis_is_refused(self, ledger: TrialLedger) -> None:
        """ "test 47" tells a future reader nothing about what was tried."""
        with pytest.raises(ValueError, match="what is being tested"):
            ledger.register(hypothesis="   ", family="x", config={}, registered_at=REGISTERED_AT)

    def test_editing_a_past_entry_breaks_the_chain(self, ledger: TrialLedger) -> None:
        """The attack the design exists to stop: rewriting a losing trial."""
        _register(ledger, 5)
        assert ledger.verify().intact, "control: intact before tampering"

        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["config"]["lookback"] = 999
        lines[1] = json.dumps(entry, sort_keys=True)
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = ledger.verify()
        assert not result.intact
        assert result.first_corrupt_index == 1
        assert "broken at entry 1" in result.explain()

    def test_deleting_a_losing_trial_breaks_the_chain(self, ledger: TrialLedger) -> None:
        """Quietly dropping failures is what inflates every downstream correction."""
        _register(ledger, 5)
        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        del lines[2]
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = ledger.verify()
        assert not result.intact
        assert result.first_corrupt_index == 2

    def test_reordering_entries_breaks_the_chain(self, ledger: TrialLedger) -> None:
        _register(ledger, 5)
        lines = ledger.path.read_text(encoding="utf-8").splitlines()
        lines[1], lines[3] = lines[3], lines[1]
        ledger.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert not ledger.verify().intact

    def test_outcomes_are_appended_rather_than_edited(self, ledger: TrialLedger) -> None:
        """Recording a result must not rewrite the registration it refers to."""
        _register(ledger, 2)
        ledger.record_outcome(1, {"sharpe": 0.02, "verdict": "abandoned"})
        assert ledger.verify().intact
        assert ledger.count(include_amendments=True) == 3

    def test_amendments_do_not_inflate_the_trial_count(self, ledger: TrialLedger) -> None:
        """An outcome is not a new attempt; counting it would over-correct."""
        _register(ledger, 4)
        ledger.record_outcome(1, {"verdict": "abandoned"})
        ledger.record_outcome(2, {"verdict": "abandoned"})
        assert ledger.count() == 4
        assert ledger.count(include_amendments=True) == 6

    def test_counts_are_scoped_by_family(self, ledger: TrialLedger) -> None:
        """Corrections apply within a family of related attempts, not globally."""
        _register(ledger, 3, family="momentum")
        _register(ledger, 5, family="accruals")
        assert ledger.count(family="momentum") == 3
        assert ledger.count(family="accruals") == 5
        assert ledger.count() == 8

    def test_recording_against_an_unknown_trial_is_refused(self, ledger: TrialLedger) -> None:
        _register(ledger, 2)
        with pytest.raises(KeyError, match="no trial registered"):
            ledger.record_outcome(99, {"verdict": "?"})

    def test_an_absent_ledger_reads_as_empty(self, tmp_path: Path) -> None:
        empty = TrialLedger(tmp_path / "missing.jsonl")
        assert empty.count() == 0
        assert empty.verify().intact


class TestMultipleTesting:
    def test_bonferroni_scales_by_the_number_of_tests(self) -> None:
        adjusted = adjust_pvalues(np.array([0.01, 0.04]), method="bonferroni")
        assert adjusted == pytest.approx([0.02, 0.08])

    def test_adjusted_pvalues_are_capped_at_one(self) -> None:
        adjusted = adjust_pvalues(np.array([0.5, 0.9]), method="bonferroni")
        assert np.all(adjusted <= 1.0)

    def test_holm_is_never_weaker_than_the_raw_pvalue(self) -> None:
        raw = np.array([0.001, 0.01, 0.03, 0.2])
        assert np.all(adjust_pvalues(raw, method="holm") >= raw)

    def test_holm_dominates_bonferroni(self) -> None:
        """Same guarantee, uniformly more power. There is no reason to prefer Bonferroni."""
        raw = np.array([0.001, 0.01, 0.03, 0.2])
        assert np.all(
            adjust_pvalues(raw, method="holm") <= adjust_pvalues(raw, method="bonferroni") + 1e-12
        )

    def test_bhy_is_stricter_than_plain_benjamini_hochberg(self) -> None:
        """Yekutieli's c(M) is what keeps the guarantee under dependence.

        Factor tests are heavily correlated, so the plain BH form would understate
        the correction on exactly the data this library is for.
        """
        raw = np.array([0.001, 0.01, 0.03, 0.2])
        count = raw.size
        c_m = float(np.sum(1.0 / np.arange(1, count + 1)))
        plain_bh = np.minimum(raw * count / np.arange(1, count + 1), 1.0)
        assert np.all(adjust_pvalues(raw, method="bhy") >= plain_bh - 1e-12)
        assert c_m > 1.0

    def test_all_methods_preserve_the_ordering_of_evidence(self) -> None:
        raw = np.array([0.2, 0.001, 0.03, 0.01])
        for method in ("bonferroni", "holm", "bhy"):
            adjusted = adjust_pvalues(raw, method=method)  # type: ignore[arg-type]
            assert np.argmin(adjusted) == np.argmin(raw)

    def test_invalid_pvalues_are_refused(self) -> None:
        with pytest.raises(ValueError, match="lie in "):
            adjust_pvalues(np.array([0.5, 1.7]))


class TestSharpeHaircut:
    def test_a_single_test_leaves_the_sharpe_alone(self) -> None:
        """Control: with nothing to correct for, nothing is taken away."""
        result = sharpe_haircut(observed_sharpe=0.15, n_observations=1260, n_tests=1)
        assert result.adjusted_sharpe == pytest.approx(result.observed_sharpe, rel=0.01)
        assert result.haircut_fraction == pytest.approx(0.0, abs=0.01)

    def test_more_tests_take_more_away(self) -> None:
        few = sharpe_haircut(observed_sharpe=0.10, n_observations=1260, n_tests=10)
        many = sharpe_haircut(observed_sharpe=0.10, n_observations=1260, n_tests=500)
        assert many.adjusted_sharpe < few.adjusted_sharpe
        assert many.haircut_fraction > few.haircut_fraction

    def test_a_marginal_result_stops_surviving_once_the_search_is_counted(self) -> None:
        """The headline use: significance that belonged to the search, not the signal."""
        alone = sharpe_haircut(observed_sharpe=0.06, n_observations=1260, n_tests=1)
        searched = sharpe_haircut(observed_sharpe=0.06, n_observations=1260, n_tests=300)
        assert alone.survives
        assert not searched.survives
        assert "does not survive" in searched.explain()

    def test_a_strong_enough_result_survives_a_large_search(self) -> None:
        """Control: the correction is not simply a way of rejecting everything."""
        result = sharpe_haircut(observed_sharpe=0.20, n_observations=2520, n_tests=300)
        assert result.survives, result.explain()

    def test_supplied_pvalues_are_used_when_available(self) -> None:
        result = sharpe_haircut(
            observed_sharpe=0.10,
            n_observations=1260,
            n_tests=5,
            other_pvalues=np.array([0.5, 0.6, 0.7, 0.8]),
        )
        assert result.n_tests == 5

    def test_degenerate_inputs_are_refused(self) -> None:
        with pytest.raises(ValueError, match="n_tests"):
            sharpe_haircut(observed_sharpe=0.1, n_observations=100, n_tests=0)
        with pytest.raises(ValueError, match="n_observations"):
            sharpe_haircut(observed_sharpe=0.1, n_observations=1, n_tests=10)
