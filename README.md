# trialkeeper

**The most important number in a backtest is the one nobody reports: how many strategies were tried before this one.**

Without it, a Sharpe ratio carries no information. Here is the same script run with an increasing search budget, where every strategy has an expected return of *exactly zero* — verbatim output of [`examples/best_of_n_noise.py`](examples/best_of_n_noise.py):

```
  trials  best annualised Sharpe  true Sharpe   deflated
----------------------------------------------------------
       1                    0.09         0.00       n/a
      10                    0.86         0.00      0.488
     100                    0.76         0.00      0.274
   1,000                    1.41         0.00      0.464
  10,000                    1.69         0.00      0.456
----------------------------------------------------------
```

A 1.41 Sharpe from pure noise. Not a bug, not unlucky data — the arithmetic of taking a maximum. A modest parameter grid (5 lookbacks × 4 thresholds × 3 holding periods × 4 universes × 3 cost assumptions) is 720 trials before you have had a second thought, and the sample period, the universe screen, and the winsorisation rule are each further dimensions nobody counts.

`trialkeeper` supplies the corrections that make a reported Sharpe interpretable, and the record-keeping that makes the trial count evidence rather than a recollection.

## Install

```bash
pip install trialkeeper     # numpy + scipy, nothing else
```

## The four tools

### 1. Deflated Sharpe ratio

Judges an observed Sharpe against what the best of `n` no-edge trials would have produced, adjusting for sample length, skewness and fat tails (Bailey & López de Prado, 2014).

```python
from trialkeeper import deflated_sharpe_ratio

result = deflated_sharpe_ratio(returns, n_trials=1000, trial_sharpes=all_trial_sharpes)
print(result.explain())
# DOES NOT SURVIVE: observed Sharpe 0.0989 per period against a selection bar of
# 0.0914 implied by 1000 trial(s) (dispersion 0.0281), over 1260 observations with
# skew +0.03 and kurtosis 2.89 → deflated Sharpe 0.6048
```

`result.survives` applies the conventional 95% threshold, but the number is the point — report it, not the flag.

(The deflated figure here differs from the 1,000-trial row in the table above because the two scripts draw different panels from the same seed. Both are verbatim output; neither is illustrative.)

### 2. Probability of backtest overfitting

A different question: not "is this strategy good" but "does picking the in-sample winner tell me anything at all". Splits history into blocks, forms every equal in-sample/out-of-sample partition, and tracks where the in-sample winner lands out of sample.

```python
from trialkeeper import probability_of_backtest_overfitting

pbo = probability_of_backtest_overfitting(returns_matrix, n_blocks=10)  # 252 splits
print(pbo.explain())
# PBO 46.8% over 252 splits — the in-sample winner's median out-of-sample rank
# was 52.4%; mean performance degraded by -0.0813 per period.
#
# 46.8% on a pure-noise panel: near the 50% coin-flip, as it should be. Selection
# on this data is worth almost nothing, and the degradation figure shows the
# in-sample winner giving back its apparent edge out of sample.
```

**PBO near 0.5 does not mean the strategy is mediocre. It means the selection process is worthless** — it would have done as well picking at random. That is a more damaging finding than a low Sharpe, and no single train/test split can reveal it.

### 3. Purged, embargoed, combinatorial cross-validation

Standard k-fold assumes independent observations. Financial labels are not: a five-day forward return shares four days with its neighbour, so a naive split puts the answer in the training set.

```python
from trialkeeper import combinatorial_purged_splits

for split in combinatorial_purged_splits(
    n_samples=len(data), n_groups=6, n_test_groups=2, label_horizon=5, embargo=2
):
    fit(data[split.train])
    score(data[split.test])
    # 15 distinct paths through history, not one
```

Purging drops training observations whose label windows overlap the test set; the embargo drops those immediately after it, where serial correlation leaks even without overlap. One path is an anecdote — the spread across paths is the evidence.

### 4. The trial ledger

Every correction above needs the trial count, and by the time a result looks good the abandoned variants are genuinely forgotten. So trials are registered *before* they run, into an append-only hash chain.

```python
from trialkeeper import TrialLedger

ledger = TrialLedger(Path("trials.jsonl"))
ledger.register(
    hypothesis="restatement magnitude predicts 60-day returns",
    family="restatements",
    config={"window": 60, "min_change": 0.05},
    registered_at=now.isoformat(),
)
...
ledger.record_outcome(sequence=1, outcome={"sharpe": 0.02, "verdict": "abandoned"})

deflated_sharpe_ratio(returns, n_trials=ledger.count(family="restatements"), ...)
```

Deleting a losing trial is still possible — it is a text file. It is just no longer *invisible*: every entry hashes the previous head, so removing or editing one breaks the chain, and `ledger.verify()` names the entry where it broke. Outcomes are appended, never edited, so the record reads "this was registered, then this was found".

## How it is validated

Not by matching numbers copied out of a paper — by simulation against known ground truth, which would catch an error in the paper as readily as an error in the code:

| Claim | How it is checked |
|---|---|
| `expected_max_sharpe` is right | 200 replications × 50 zero-edge strategies; the closed form must predict the simulated best-of-50 within 6% |
| PBO ≈ 0.5 on noise | pure-noise panel, 252 splits |
| PBO ≈ 0 on a real edge | one column given a genuine persistent drift |
| PBO rises on a fitted strategy | edge in the first half, reversed in the second, compared against a *measured* noise baseline rather than a guessed constant |
| Fat tails lower confidence | two series with identical Sharpe and exactly zero skew, differing only in kurtosis |
| Negative skew lowers confidence | a series and its mirror image: identical kurtosis, opposite skew |
| The chain detects tampering | edit, delete and reorder attacks, each with an intact-ledger control |

Every simulation is seeded. A failure is reproducible, not occasional.

## What this is not

- **Not a backtester.** It evaluates return series; it does not produce them.
- **Not a substitute for out-of-sample data.** These corrections make an in-sample result honest about its own uncertainty. They cannot manufacture evidence that is not there.
- **Not a way to rescue a strategy.** If the deflated Sharpe says no, the answer is no. Searching for a correction that says yes is another trial, and it belongs in the ledger.

## References

- Bailey & López de Prado (2012), *The Sharpe Ratio Efficient Frontier* — probabilistic Sharpe ratio, minimum track record length
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2014), *The Probability of Backtest Overfitting*
- López de Prado (2018), *Advances in Financial Machine Learning* — purging, embargo, combinatorial purged CV
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns* — multiple-testing haircuts
- Benjamini & Yekutieli (2001) — FDR control under arbitrary dependence

## Where this lives

Developed inside [ALETHEIA](https://github.com/j-poc/aletheia), a point-in-time
evidence engine, and mirrored here as a standalone package. The monorepo is the
source of truth; this repository is a subtree split of `packages/trialkeeper`, so
its history is the real history rather than a squashed import.

The boundary is enforced rather than asserted: the test suite runs with **zero**
imports from `aletheia.*`, which is what makes the split honest instead of
aspirational.

MIT licensed.
