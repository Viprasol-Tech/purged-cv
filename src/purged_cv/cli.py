"""``purgedcv`` command line: visualise a purged CV plan before you train on it.

Part of purged-cv by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ._core import find_leaks, horizon_spans
from .splitters import CombinatorialPurgedKFold, PurgedKFold, PurgedWalkForward

MAX_LANE = 200


def _lane(n: int, train: set[int], test: set[int], window: range) -> str:
    chars = []
    for i in range(n):
        if i in test:
            chars.append("T")
        elif i in train:
            chars.append("=")
        elif i in window:
            chars.append(".")
        else:
            chars.append(" ")
    return "".join(chars)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="purgedcv",
        description="Print a leak-free cross-validation plan (purging + embargo).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="draw train/test/purged lanes for each split")
    plan.add_argument("--method", choices=["kfold", "cpcv", "walkforward"], default="kfold")
    plan.add_argument("--samples", type=int, default=60, help="number of samples (bars)")
    plan.add_argument("--splits", type=int, default=5, help="folds (kfold / walkforward)")
    plan.add_argument("--groups", type=int, default=6, help="CPCV groups N")
    plan.add_argument("--test-groups", type=int, default=2, help="CPCV test groups k")
    plan.add_argument("--horizon", type=int, default=0, help="label look-ahead in bars")
    plan.add_argument("--embargo", type=float, default=0.0, help="embargo fraction, e.g. 0.01")
    plan.add_argument("--max-train", type=int, default=None, help="rolling window (walkforward)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n: int = args.samples
    if not 1 <= n <= MAX_LANE:
        print(f"error: --samples must be between 1 and {MAX_LANE}", file=sys.stderr)
        return 2
    spans = horizon_spans(n, args.horizon)
    try:
        splitter: PurgedKFold | CombinatorialPurgedKFold | PurgedWalkForward
        if args.method == "kfold":
            splitter = PurgedKFold(args.splits, spans=spans, embargo_pct=args.embargo)
        elif args.method == "cpcv":
            splitter = CombinatorialPurgedKFold(
                args.groups, args.test_groups, spans=spans, embargo_pct=args.embargo
            )
        else:
            splitter = PurgedWalkForward(args.splits, max_train_size=args.max_train, spans=spans)
        splits = list(splitter.split(range(n)))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{args.method}: {n} samples, horizon={args.horizon}, embargo={args.embargo}")
    print("legend: T test  = train  . purged/embargoed\n")
    total_leaks = 0
    for k, (train, test) in enumerate(splits):
        if args.method == "walkforward":
            lo = 0 if args.max_train is None else max(0, test[0] - args.max_train)
            window = range(lo, test[0])
        else:
            window = range(n)
        dropped = sum(1 for i in window if i not in train and i not in test)
        leaks = len(find_leaks(train, test, spans))
        total_leaks += leaks
        lane = _lane(n, set(train), set(test), window)
        print(f"{k:>3} |{lane}| train={len(train)} test={len(test)} dropped={dropped}")
    if isinstance(splitter, CombinatorialPurgedKFold):
        print(f"\n{len(splits)} splits -> {splitter.n_paths} backtest paths")
    print(f"\nleaking train/test pairs: {total_leaks}")
    return 0 if total_leaks == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
