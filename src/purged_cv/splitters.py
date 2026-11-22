"""Leak-free cross-validation splitters for financial time series.

Part of purged-cv by Viprasol Tech Private Limited (https://viprasol.com).

All splitters follow the scikit-learn CV protocol (``split`` / ``get_n_splits``),
so they plug straight into ``cross_val_score``, ``GridSearchCV`` and friends, but
they have no dependencies themselves.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from itertools import combinations
from typing import Any

from ._core import Span, embargo_size, fold_bounds, horizon_spans, purged_train, validate_spans

Split = tuple[list[int], list[int]]


class _SpanSplitter:
    def __init__(self, spans: Sequence[Span] | None, horizon: int, embargo_pct: float) -> None:
        if spans is not None and horizon:
            raise ValueError("pass either spans or horizon, not both")
        self.spans = None if spans is None else validate_spans(spans)
        self.horizon = horizon
        self.embargo_pct = embargo_pct
        embargo_size(1, embargo_pct)  # validate early

    def _resolve(self, x: Any) -> list[Span]:
        n = len(x)
        if self.spans is None:
            return horizon_spans(n, self.horizon)
        if len(self.spans) != n:
            raise ValueError(f"got {n} samples but {len(self.spans)} spans")
        return self.spans


class PurgedKFold(_SpanSplitter):
    """K-fold over contiguous blocks with purging and an embargo (no shuffling).

    Args:
        n_splits: number of folds.
        spans: per-sample label span ``(t0, t1)``; sorted by ``t0``.
        horizon: shortcut for ``spans = (i, i + horizon)`` when labels look N bars ahead.
        embargo_pct: fraction of samples dropped from training right after each test fold.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        spans: Sequence[Span] | None = None,
        horizon: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        super().__init__(spans, horizon, embargo_pct)
        self.n_splits = n_splits

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Split]:
        spans = self._resolve(X)
        h = embargo_size(len(spans), self.embargo_pct)
        for start, stop in fold_bounds(len(spans), self.n_splits):
            yield purged_train(spans, [(start, stop)], h), list(range(start, stop))


class CombinatorialPurgedKFold(_SpanSplitter):
    """Combinatorial Purged CV (CPCV): every choice of ``n_test_groups`` of ``n_groups``.

    Produces ``C(N, k)`` train/test splits whose test sets recombine into
    ``k / N * C(N, k)`` full out-of-sample backtest paths (see :meth:`paths`),
    giving a distribution of backtest results instead of a single one.
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        *,
        spans: Sequence[Span] | None = None,
        horizon: int = 0,
        embargo_pct: float = 0.0,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must be in [1, n_groups)")
        super().__init__(spans, horizon, embargo_pct)
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups

    @property
    def n_paths(self) -> int:
        """Number of complete backtest paths: ``C(N - 1, k - 1)``."""
        return math.comb(self.n_groups - 1, self.n_test_groups - 1)

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return math.comb(self.n_groups, self.n_test_groups)

    def test_groups(self) -> list[tuple[int, ...]]:
        """The group indices used as the test set of each split, in split order."""
        return list(combinations(range(self.n_groups), self.n_test_groups))

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Split]:
        spans = self._resolve(X)
        h = embargo_size(len(spans), self.embargo_pct)
        bounds = fold_bounds(len(spans), self.n_groups)
        for combo in self.test_groups():
            blocks = [bounds[g] for g in combo]
            test = [i for start, stop in blocks for i in range(start, stop)]
            yield purged_train(spans, blocks, h), test

    def paths(self) -> list[list[tuple[int, int]]]:
        """Backtest paths as ``[(group, split_index), ...]`` covering every group once.

        Path ``p`` takes, for each group, the ``p``-th split in which that group was
        tested. Stitch each split's out-of-sample predictions for those groups
        together to get ``n_paths`` independent full-history backtests.
        """
        seen: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for s, combo in enumerate(self.test_groups()):
            for g in combo:
                seen[g].append(s)
        return [[(g, seen[g][p]) for g in range(self.n_groups)] for p in range(self.n_paths)]


class PurgedWalkForward(_SpanSplitter):
    """Walk-forward (expanding or rolling) splits with purging before each test block.

    Mirrors scikit-learn's ``TimeSeriesSplit`` layout: the series is cut into
    ``n_splits + 1`` blocks and each block after the first is tested on, trained on
    what came before. Training labels that reach into the test block are purged.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        max_train_size: int | None = None,
        spans: Sequence[Span] | None = None,
        horizon: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError("max_train_size must be >= 1")
        super().__init__(spans, horizon, 0.0)
        self.n_splits = n_splits
        self.max_train_size = max_train_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Split]:
        spans = self._resolve(X)
        for start, stop in fold_bounds(len(spans), self.n_splits + 1)[1:]:
            lo = 0 if self.max_train_size is None else max(0, start - self.max_train_size)
            train = purged_train(spans, [(start, stop)], 0, candidates=range(lo, start))
            yield train, list(range(start, stop))
