"""Append-only, hash-chained record of every trial.

Every correction in this library needs one input that nobody keeps: **how many
strategies were actually tried.** It is not that researchers lie about it. It is
that by the time a result looks good, the abandoned variants are genuinely
forgotten — the grid searched on a Tuesday, the universe swapped, the sample
period nudged.

So the count is not asked for at the end. Each trial is registered *before* it
runs, and the registration is chained: entry ``n`` hashes the head of entry
``n-1``. Editing or removing an earlier trial changes every head after it, and
:meth:`TrialLedger.verify` reports exactly where. Deleting the losers is still
possible — it is just no longer invisible, which is the whole point.

The ledger is a JSON-lines file. Human-readable, diffable, and committable, so
its history is timestamped by the version control system as well as by itself.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

GENESIS: Final = "0" * 64
CHAIN_KEY: Final = "chain_head"


@dataclass(frozen=True, slots=True)
class Trial:
    """One registered attempt."""

    sequence: int
    hypothesis: str
    """What was being tested, in words. Vague entries make the count useless later."""
    family: str
    """Trials sharing a family are corrected together. Same idea, different knobs."""
    config: dict[str, Any]
    registered_at: str
    """ISO-8601. Supplied by the caller so the ledger stays reproducible."""
    chain_head: str
    outcome: dict[str, Any] = field(default_factory=dict)
    """Filled in after the run. Recording a failure is the ledger's main job."""

    def body(self) -> dict[str, Any]:
        """The hashed content — everything except the head it produces."""
        return {
            "sequence": self.sequence,
            "hypothesis": self.hypothesis,
            "family": self.family,
            "config": self.config,
            "registered_at": self.registered_at,
            "outcome": self.outcome,
        }


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Whether the chain is intact, and where it first is not."""

    intact: bool
    first_corrupt_index: int | None
    n_entries: int
    head: str

    def explain(self) -> str:
        if self.intact:
            return f"chain intact over {self.n_entries} entries, head {self.head[:12]}…"
        return (
            f"chain broken at entry {self.first_corrupt_index} of {self.n_entries}: "
            f"that entry or one before it was altered after registration"
        )


class TrialLedger:
    """A hash-chained JSON-lines log of registered trials."""

    def __init__(self, path: Path) -> None:
        self.path = path

    # ------------------------------------------------------------- writing --

    def register(
        self,
        *,
        hypothesis: str,
        family: str,
        config: dict[str, Any],
        registered_at: str,
    ) -> Trial:
        """Record a trial **before** running it.

        Registering afterwards defeats the purpose: the value of the count comes
        from it including the attempts that turned out badly.
        """
        if not hypothesis.strip():
            raise ValueError("hypothesis must say what is being tested")
        if not family.strip():
            raise ValueError("family must name the group this trial is corrected within")

        entries = list(self.read())
        head = entries[-1].chain_head if entries else GENESIS
        trial = Trial(
            sequence=len(entries) + 1,
            hypothesis=hypothesis.strip(),
            family=family.strip(),
            config=config,
            registered_at=registered_at,
            chain_head=GENESIS,
        )
        chained = Trial(**{**_as_dict(trial), "chain_head": _next_head(head, trial.body())})
        self._append(chained)
        return chained

    def record_outcome(self, sequence: int, outcome: dict[str, Any]) -> None:
        """Attach a result to a registered trial by appending an amendment.

        The original entry is never edited — that would break the chain, which is
        exactly the property being protected. The amendment is a new link, so the
        history reads as "this was registered, then this was found".
        """
        entries = list(self.read())
        if not any(entry.sequence == sequence for entry in entries):
            raise KeyError(f"no trial registered with sequence {sequence}")
        original = next(entry for entry in entries if entry.sequence == sequence)
        head = entries[-1].chain_head
        amendment = Trial(
            sequence=len(entries) + 1,
            hypothesis=f"[outcome of trial {sequence}] {original.hypothesis}",
            family=original.family,
            config={"amends": sequence},
            registered_at=original.registered_at,
            chain_head=GENESIS,
            outcome=outcome,
        )
        self._append(
            Trial(**{**_as_dict(amendment), "chain_head": _next_head(head, amendment.body())})
        )

    # ------------------------------------------------------------- reading --

    def read(self) -> Iterator[Trial]:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield _from_json(json.loads(line))

    def count(self, *, family: str | None = None, include_amendments: bool = False) -> int:
        """Number of trials — the input every correction in this library needs.

        Amendments are excluded by default: an outcome record is not a new
        attempt, and counting it would inflate the correction.
        """
        return sum(
            1
            for entry in self.read()
            if (family is None or entry.family == family)
            and (include_amendments or "amends" not in entry.config)
        )

    def verify(self) -> VerificationResult:
        """Recompute the chain and report the first entry that does not match."""
        head = GENESIS
        count = 0
        for index, entry in enumerate(self.read()):
            count += 1
            head = _next_head(head, entry.body())
            if entry.chain_head != head:
                return VerificationResult(
                    intact=False, first_corrupt_index=index, n_entries=count, head=head
                )
        return VerificationResult(intact=True, first_corrupt_index=None, n_entries=count, head=head)

    def _append(self, trial: Trial) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_as_dict(trial), sort_keys=True) + "\n")


def _next_head(previous_head: str, body: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(previous_head.encode("ascii"))
    digest.update(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def _as_dict(trial: Trial) -> dict[str, Any]:
    return {field_name: getattr(trial, field_name) for field_name in Trial.__slots__}


def _from_json(payload: dict[str, Any]) -> Trial:
    return Trial(
        sequence=int(payload["sequence"]),
        hypothesis=str(payload["hypothesis"]),
        family=str(payload["family"]),
        config=dict(payload.get("config", {})),
        registered_at=str(payload["registered_at"]),
        chain_head=str(payload[CHAIN_KEY]),
        outcome=dict(payload.get("outcome", {})),
    )
