"""Retrieval quality, scored upstream of what the model did with what it retrieved."""

from __future__ import annotations

import pytest

from agent_eval_kit.retrieval import (
    RetrievalCase,
    RetrievalError,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)


def test_recall_counts_the_labelled_passages_found_in_the_top_k():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["a", "z"], ["a", "z"], 3) == 1.0
    # Outside the cut is not retrieved, as far as a truncated prompt is concerned.
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_an_unlabelled_query_is_refused_rather_than_scored_a_vacuous_one():
    """Recall over an empty relevant set is 1.0 by construction, which measures nothing."""
    with pytest.raises(RetrievalError, match="vacuously"):
        recall_at_k(["a"], [], 3, query="what is our exposure")


def test_precision_divides_by_what_was_returned_not_by_k():
    """Three good passages when five were asked for is not a 0.6.

    Dividing by ``k`` would punish an honest short answer and reward padding the list with
    noise, which is the opposite of what this metric exists to encourage.
    """
    assert precision_at_k(["a", "b"], ["a", "b"], 5) == 1.0
    assert precision_at_k(["a", "noise"], ["a"], 5) == 0.5


def test_a_repeated_passage_cannot_raise_precision():
    assert precision_at_k(["a", "a", "a"], ["a"], 5) == 1.0
    assert precision_at_k(["a", "a", "noise"], ["a"], 5) == 0.5


def test_retrieving_nothing_scores_zero_because_it_is_a_real_outcome():
    assert precision_at_k([], ["a"], 5) == 0.0
    assert recall_at_k([], ["a"], 5) == 0.0
    assert reciprocal_rank([], ["a"]) == 0.0


def test_reciprocal_rank_counts_position_over_the_list_as_returned():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_scores_are_macro_averaged_per_query():
    """One query with forty labelled passages must not decide the number for twenty queries."""
    fat = RetrievalCase("q1", tuple(f"p{i}" for i in range(40)), tuple(f"p{i}" for i in range(40)))
    thin = RetrievalCase("q2", ("miss",), ("wanted",))
    scores = score_retrieval([fat, thin], k=40)
    assert scores.recall_at_k == 0.5  # 1.0 and 0.0, per query, not pooled
    assert scores.n_queries == 2
    assert scores.n_relevant == 41


def test_the_scores_render_under_conventional_metric_names():
    scores = score_retrieval([RetrievalCase("q", ("a",), ("a",))], k=5)
    assert scores.as_metrics() == {
        "retrieval_recall_at_5": 1.0,
        "retrieval_precision_at_5": 1.0,
        "retrieval_mrr": 1.0,
    }
    assert "kb_recall_at_5" in scores.as_metrics(prefix="kb")


def test_an_empty_corpus_is_refused_not_averaged():
    with pytest.raises(RetrievalError, match="vacuous"):
        score_retrieval([], k=5)
    assert mean_reciprocal_rank([]) == 0.0


def test_a_case_must_name_its_query_and_label_something():
    with pytest.raises(RetrievalError, match="name its query"):
        RetrievalCase("  ", ("a",), ("a",))
    with pytest.raises(RetrievalError, match="labels no relevant passage"):
        RetrievalCase("q", ("a",), ())


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_top_zero_cut_retrieves_nothing_and_is_refused(bad):
    with pytest.raises(RetrievalError):
        recall_at_k(["a"], ["a"], bad)


def test_the_denominator_rule_applies_to_a_recall_bar_over_labelled_passages():
    """The two modules meet here: `n_relevant` is what a recall bar is measured over."""
    from agent_eval_kit.denominators import DenominatorError, assert_denominator_supports

    scores = score_retrieval([RetrievalCase("q", ("a", "b"), ("a", "b"))], k=5)
    with pytest.raises(DenominatorError, match="identical to 1.0"):
        assert_denominator_supports(0.90, scores.n_relevant, metric="retrieval_recall_at_5")
