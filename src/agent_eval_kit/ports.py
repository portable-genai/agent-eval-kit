"""The evaluation-gate port, declared once so every repo binds the same shape.

A repository asks two questions of its quality authority: what did this dataset score, and may
this target be promoted. Sixteen repositories had each hand-copied a Protocol for that, and by the
time anyone compared them they disagreed. One had dropped the port entirely, two had dropped the
``gate`` method and kept only ``evaluate``, which is the half that cannot refuse a promotion. A
Protocol copied into N repositories is N Protocols, and only one of them gets fixed when a defect
is found.

It lives in this package rather than in ``hex-service-kit`` because
:class:`~agent_eval_kit.report.EvalReport` is already here, next to
:class:`~agent_eval_kit.gate_client.PromotionGateClient` and
:func:`~agent_eval_kit.modes.eval_main`. Putting the Protocol in the other kit would have dragged
the report type across a package boundary to satisfy an annotation, and left the two halves of the
same concern in different release lines.

Binding this port rather than constructing a client directly is what makes the offline profile
honest. ``--mode smoke`` runs against in-memory fakes with no credentials and gates every merge;
``--mode gate`` resolves through the container, so a profile with no quality service bound REFUSES
rather than quietly scoring itself and passing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .report import EvalReport


@runtime_checkable
class EvaluationGatePort(Protocol):
    """Scores a golden dataset, and answers whether a target may be promoted.

    The two methods are deliberately separate. :meth:`evaluate` is evidence and is safe to run
    anywhere; :meth:`gate` is an authority decision. An adapter that implements only the first is
    not a gate, which is exactly the drift this Protocol exists to stop: a port with no ``gate``
    still satisfies "we have an evaluation port" while being unable to refuse anything.

    Every implementation fails CLOSED. An unreachable quality service, an empty dataset or a
    metric that could not be computed is a refusal, never a pass, because the alternative is a
    promotion certified by the absence of evidence.
    """

    def evaluate(self, dataset_path: str) -> EvalReport:
        """Score the golden dataset and return the full report, passing or not."""
        ...

    def gate(self, target: str) -> bool:
        """Whether ``target`` clears the promotion thresholds.

        ``True`` only on positive evidence. See
        :func:`~agent_eval_kit.harness.assert_each_can_go_red` for the companion rule: a metric
        that cannot fail is a constant, and a gate built on constants approves everything.
        """
        ...


__all__ = ["EvaluationGatePort"]
