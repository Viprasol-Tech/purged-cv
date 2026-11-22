"""Purging, embargo and leak detection primitives.

Part of purged-cv by Viprasol Tech Private Limited (https://viprasol.com).

Every sample ``i`` carries a label *span* ``(t0_i, t1_i)``: the first and last
moment of information its label depends on (e.g. bar ``i`` to bar ``i + horizon``
for a forward-return label). Two samples leak into each other when their spans
overlap, so a training sample must be dropped ("purged") if its span touches any
test span, and samples starting right after a test block are dropped too
("embargo") to absorb serial correlation.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence

Span = tuple[float, float]


def horizon_spans(n_samples: int, horizon: int = 0) -> list[Span]:
    """Spans for samples whose label looks ``horizon`` bars ahead: ``(i, i + horizon)``."""
    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    return [(float(i), float(i + horizon)) for i in range(n_samples)]


def validate_spans(spans: Sequence[Span]) -> list[Span]:
    """Return spans as floats; require ``t0 <= t1`` and starts sorted ascending."""
    out: list[Span] = []
    prev_start = -math.inf
    for i, (t0, t1) in enumerate(spans):
        a, b = float(t0), float(t1)
        if math.isnan(a) or math.isnan(b):
            raise ValueError(f"span {i} contains NaN")
        if b < a:
            raise ValueError(f"span {i} ends before it starts: ({t0}, {t1})")
        if a < prev_start:
            raise ValueError(f"spans must be sorted by start time (span {i} is out of order)")
        prev_start = a
        out.append((a, b))
    return out


def fold_bounds(n_samples: int, n_folds: int) -> list[tuple[int, int]]:
    """Split ``range(n_samples)`` into ``n_folds`` contiguous ``[start, stop)`` blocks.

    Sizes differ by at most one; the first ``n_samples % n_folds`` blocks are larger.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    if n_samples < n_folds:
        raise ValueError(f"cannot split {n_samples} samples into {n_folds} folds")
    base, extra = divmod(n_samples, n_folds)
    bounds: list[tuple[int, int]] = []
    start = 0
    for k in range(n_folds):
        stop = start + base + (1 if k < extra else 0)
        bounds.append((start, stop))
        start = stop
    return bounds


def embargo_size(n_samples: int, embargo_pct: float) -> int:
    """Number of samples embargoed after each test block (``ceil(n * pct)``)."""
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")
    return math.ceil(n_samples * embargo_pct)


def merge_blocks(blocks: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge touching or overlapping ``[start, stop)`` blocks."""
    merged: list[tuple[int, int]] = []
    for start, stop in sorted(blocks):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def purged_train(
    spans: Sequence[Span],
    test_blocks: Sequence[tuple[int, int]],
    embargo: int = 0,
    candidates: Sequence[int] | None = None,
) -> list[int]:
    """Training indices left after purging and embargoing around ``test_blocks``.

    A candidate ``j`` survives only if, for every test block, its span ends strictly
    before the block's first label starts, or it starts strictly after both the
    block's last label ends and the ``embargo`` samples that start after that
    (as in López de Prado, *Advances in Financial Machine Learning*, ch. 7).
    """
    n = len(spans)
    blocks = merge_blocks(test_blocks)
    starts = [t0 for t0, _ in spans]
    in_test = [False] * n
    windows: list[tuple[float, float]] = []
    for start, stop in blocks:
        if not 0 <= start < stop <= n:
            raise ValueError(f"invalid test block [{start}, {stop}) for {n} samples")
        for i in range(start, stop):
            in_test[i] = True
        test_start = min(spans[i][0] for i in range(start, stop))
        test_end = max(spans[i][1] for i in range(start, stop))
        # embargo: the first `embargo` samples starting after the test labels end
        last = bisect_right(starts, test_end) + embargo - 1
        limit = test_end if embargo == 0 else (spans[last][0] if last < n else math.inf)
        windows.append((test_start, limit))

    pool = range(n) if candidates is None else candidates
    keep: list[int] = []
    for j in pool:
        if in_test[j]:
            continue
        t0, t1 = spans[j]
        if all(t1 < lo or t0 > hi for lo, hi in windows):
            keep.append(j)
    return keep


def find_leaks(
    train: Sequence[int], test: Sequence[int], spans: Sequence[Span]
) -> list[tuple[int, int]]:
    """Every ``(train_idx, test_idx)`` pair whose label spans overlap.

    Use it to audit any split — including ones you did not make with this library.
    An empty list means no label information is shared between train and test.
    """
    clean = validate_spans(spans)
    test_spans = sorted((clean[i][0], clean[i][1], i) for i in test)
    leaks: list[tuple[int, int]] = []
    # ponytail: O(train x test) scan; fine for audit-sized data, use an interval tree if huge
    for j in train:
        a, b = clean[j]
        for t0, t1, i in test_spans:
            if t0 > b:
                break
            if t1 >= a:
                leaks.append((j, i))
    return leaks
