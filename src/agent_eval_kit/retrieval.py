"""Retrieval quality, scored separately from what the model did with what it retrieved.

Eight retrieval ports are bound across the launch set and the second wave, and not one of them
had a metric. Every "grounding" number in the fleet is computed downstream of retrieval, over
the passages retrieval happened to return, so a knowledge base that silently stopped returning
the right document still scores a clean citation set: the answer cites what it was given, and
what it was given is no longer the evidence. The regression is invisible precisely because the
metric is well behaved.

This module is the missing measurement, and it is deliberately upstream of generation:

* :func:`recall_at_k` : of the passages a reviewer marked relevant, how many were retrieved in
  the top ``k``. This is the one that catches a knowledge base going quiet.
* :func:`precision_at_k` : of the top ``k`` retrieved, how many are relevant. This is the one
  that catches a retriever padding its results, which raises recall and destroys a reader's
  ability to find the evidence.
* :func:`reciprocal_rank` and :func:`mean_reciprocal_rank` : how far down the list the first
  relevant passage sits. Rank matters wherever the prompt is truncated, which is everywhere.

The oracle is a hand-written ``(query, relevant passage ids)`` pair, never the retriever's own
score. A relevance label derived from what the retriever returned asks the component under test
whether it agrees with itself.

Fail-closed in the two ways that matter and are easy to get wrong:

* a query with **no relevant passages labelled** is refused, not scored 1.0. Recall over an
  empty relevant set is vacuously perfect, and a corpus of such queries reports a flawless
  retriever that has never been asked for anything.
* a query whose retrieval returned **nothing** scores 0.0 rather than raising, because an empty
  result set is a real and important retrieval outcome, and it must be able to lower the score.

Duplicate ids in a retrieved list are collapsed for the set-based metrics and kept for rank, so
a retriever cannot raise precision by returning the same passage twice.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

__all__ = [
    "RetrievalCase",
    "RetrievalError",
    "RetrievalScores",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_retrieval",
]


class RetrievalError(ValueError):
    """Raised when a retrieval case cannot be scored, or would be scored vacuously."""


def _checked_k(k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool):
        raise RetrievalError(f"k must be an int, got {k!r}")
    if k < 1:
        raise RetrievalError(f"k must be at least 1, got {k}; a top-0 cut retrieves nothing")
    return k


def _relevant(relevant: Iterable[str], *, query: str) -> frozenset[str]:
    ids = frozenset(str(item).strip() for item in relevant if str(item).strip())
    if not ids:
        raise RetrievalError(
            f"query {query!r} labels no relevant passage, so recall over it is vacuously 1.0. "
            "Either label what a reviewer expects this query to find, or drop the case: a "
            "corpus of unlabelled queries reports a flawless retriever that was never asked "
            "for anything."
        )
    return ids


def recall_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int, *, query: str = ""
) -> float:
    """Share of the labelled relevant passages that appear in the top ``k`` retrieved."""
    cut = _checked_k(k)
    wanted = _relevant(relevant, query=query)
    found = {str(item) for item in retrieved[:cut]} & wanted
    return len(found) / len(wanted)


def precision_at_k(
    retrieved: Sequence[str], relevant: Iterable[str], k: int, *, query: str = ""
) -> float:
    """Share of the top ``k`` retrieved that a reviewer marked relevant.

    The denominator is the number of DISTINCT passages actually returned, capped at ``k``, not
    ``k`` itself. Dividing by ``k`` would punish a retriever that honestly returned three good
    passages when asked for five, and reward one that padded the list to five with noise, which
    is the opposite of the behaviour this metric exists to encourage. An empty result set scores
    0.0: retrieving nothing is a real outcome and must be able to lower the score.
    """
    cut = _checked_k(k)
    wanted = _relevant(relevant, query=query)
    seen: list[str] = []
    for item in retrieved[:cut]:
        text = str(item)
        if text not in seen:
            seen.append(text)
    if not seen:
        return 0.0
    return sum(1 for item in seen if item in wanted) / len(seen)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str], *, query: str = "") -> float:
    """``1 / rank`` of the first relevant passage, or 0.0 when none was retrieved at all.

    Rank is counted over the list as returned, duplicates included, because that is the list a
    prompt is built from and a truncated context window cuts.
    """
    wanted = _relevant(relevant, query=query)
    for position, item in enumerate(retrieved, start=1):
        if str(item) in wanted:
            return 1.0 / position
    return 0.0


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    """One labelled query: what was asked, what came back, and what should have."""

    query: str
    retrieved: tuple[str, ...]
    relevant: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "retrieved", tuple(str(item) for item in self.retrieved))
        object.__setattr__(self, "relevant", tuple(str(item) for item in self.relevant))
        if not self.query.strip():
            raise RetrievalError("a retrieval case must name its query")
        _relevant(self.relevant, query=self.query)


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """The three numbers over a whole labelled set, plus the denominator behind them."""

    recall_at_k: float
    precision_at_k: float
    mean_reciprocal_rank: float
    k: int
    n_queries: int
    n_relevant: int

    def as_metrics(self, prefix: str = "retrieval") -> dict[str, float]:
        """The three scores under conventional metric names, ready for an ``EvalReport``."""
        return {
            f"{prefix}_recall_at_{self.k}": self.recall_at_k,
            f"{prefix}_precision_at_{self.k}": self.precision_at_k,
            f"{prefix}_mrr": self.mean_reciprocal_rank,
        }


def mean_reciprocal_rank(cases: Sequence[RetrievalCase]) -> float:
    """Mean of :func:`reciprocal_rank` over ``cases``. An empty set is 0.0, never 1.0."""
    if not cases:
        return 0.0
    return sum(reciprocal_rank(c.retrieved, c.relevant, query=c.query) for c in cases) / len(cases)


def score_retrieval(cases: Sequence[RetrievalCase], k: int = 5) -> RetrievalScores:
    """Score a labelled set, macro-averaged per query, with the denominator carried alongside.

    Macro-averaged rather than pooled: pooling every query's passages into one confusion count
    lets a single query with forty labelled passages decide the number for a corpus of twenty
    queries. Per-query averaging is what a reader means by "how often does retrieval find the
    right thing".

    ``n_relevant`` is the total labelled positives across the set, and it is carried so a caller
    can hand it to :func:`~agent_eval_kit.denominators.assert_denominator_supports`: a recall
    bar over four labelled passages is a 1.0 with a friendlier label.
    """
    cut = _checked_k(k)
    if not cases:
        raise RetrievalError(
            "no labelled retrieval cases, so every score would be a vacuous average over an "
            "empty set; an absent measurement is not a passing one"
        )
    recalls = [recall_at_k(c.retrieved, c.relevant, cut, query=c.query) for c in cases]
    precisions = [precision_at_k(c.retrieved, c.relevant, cut, query=c.query) for c in cases]
    return RetrievalScores(
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=sum(precisions) / len(precisions),
        mean_reciprocal_rank=mean_reciprocal_rank(cases),
        k=cut,
        n_queries=len(cases),
        n_relevant=sum(len(frozenset(c.relevant)) for c in cases),
    )
