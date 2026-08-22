"""The ``run_eval.py --mode smoke|gate`` scaffold every repo's eval entry point reuses.

Two named layers, so an offline pre-merge check is never confused with a promotion verdict:

* **smoke** (default) - the offline pre-merge check CI runs on every change, using the repo's
  own deterministic heuristic evaluator. It is a smoke check, NOT the promotion authority.
* **gate** - the promotion verdict from a shared promotion-gate service (see
  :mod:`agent_eval_kit.gate_client`); it fails closed on the reconciled evaluate + gate result
  and typically requires a ``platform``/``gcp`` profile.

A project supplies two callables (its offline ``smoke`` scorer and its ``gate`` runner) and gets
a standard CLI: ``--dataset`` and ``--mode``, aligned report output, and the fail-closed exit
codes. Exit is ``0`` iff the report passes (and, in gate mode, the authority's verdict also
passes).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from .report import EvalReport, print_report

#: A repo's offline smoke evaluator: score a dataset and return the report.
SmokeRunner = Callable[[Path], EvalReport]
#: A repo's remote gate runner: return the scored report AND the authority's PASS/FAIL verdict.
GateRunner = Callable[[Path], tuple[EvalReport, bool]]


def build_parser(default_dataset: Path, description: str) -> argparse.ArgumentParser:
    """The shared ``--dataset`` / ``--mode`` argument parser."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help=f"Path to the JSONL golden case set (default: {default_dataset}).",
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "gate"),
        default="smoke",
        help=(
            "smoke (default): the offline pre-merge check (this repo's harness). "
            "gate: the promotion verdict from the gate authority (requires the platform/gcp "
            "profile)."
        ),
    )
    return parser


def eval_main(
    *,
    smoke: SmokeRunner,
    gate: GateRunner,
    default_dataset: Path,
    description: str = "Offline / remote evaluation gate.",
    smoke_label: str = "offline heuristic (no cloud creds)",
    gate_label: str = "promotion gate",
    argv: list[str] | None = None,
) -> int:
    """Parse args, dispatch to ``smoke`` or ``gate``, print the report, return the exit code.

    Fail closed: in gate mode the exit is ``0`` only when BOTH the scored report passes and the
    authority's gate verdict is True, so an offline smoke result can never be relabelled a
    promotion pass.
    """
    args = build_parser(default_dataset, description).parse_args(argv)
    dataset: Path = args.dataset
    if not dataset.exists():
        print(f"error: dataset not found: {dataset}", file=sys.stderr)
        return 2

    if args.mode == "gate":
        report, gate_passed = gate(dataset)
        print_report(report, gate_label)
        print(f"  PROMOTION GATE: {'PASS' if gate_passed else 'FAIL'}")
        return 0 if (report.passed and gate_passed) else 1

    report = smoke(dataset)
    print_report(report, smoke_label)
    return 0 if report.passed else 1
