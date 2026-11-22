from __future__ import annotations

import pytest

from purged_cv import (
    CombinatorialPurgedKFold,
    PurgedKFold,
    PurgedWalkForward,
    find_leaks,
    horizon_spans,
    validate_spans,
)
from purged_cv._core import embargo_size, fold_bounds, merge_blocks
from purged_cv.cli import main


def test_fold_bounds_balanced_and_contiguous() -> None:
    assert fold_bounds(23, 5) == [(0, 5), (5, 10), (10, 15), (15, 19), (19, 23)]
    with pytest.raises(ValueError):
        fold_bounds(3, 5)


def test_merge_blocks_joins_adjacent() -> None:
    assert merge_blocks([(5, 10), (0, 5), (12, 15)]) == [(0, 10), (12, 15)]


def test_embargo_size() -> None:
    assert embargo_size(100, 0.0) == 0
    assert embargo_size(100, 0.01) == 1
    assert embargo_size(101, 0.01) == 2
    with pytest.raises(ValueError):
        embargo_size(10, 1.0)


def test_kfold_without_horizon_is_plain_kfold() -> None:
    n = 23
    for train, test in PurgedKFold(5).split(range(n)):
        assert sorted(train + test) == list(range(n))


def test_kfold_purges_overlapping_labels() -> None:
    splits = list(PurgedKFold(4, horizon=2).split(range(20)))
    train, test = splits[1]
    assert test == list(range(5, 10))
    # 3,4 reach into the test start; 10,11 start inside the test's label window
    assert train == [0, 1, 2, *range(12, 20)]


def test_kfold_embargo_drops_samples_after_test() -> None:
    train, test = list(PurgedKFold(4, embargo_pct=0.1).split(range(20)))[1]
    assert test == list(range(5, 10))
    assert train == [*range(5), *range(12, 20)]


def test_last_fold_embargo_does_not_overflow() -> None:
    train, test = list(PurgedKFold(4, horizon=1, embargo_pct=0.2).split(range(20)))[-1]
    assert test == list(range(15, 20))
    assert train == list(range(14))


@pytest.mark.parametrize("horizon", [0, 1, 5])
@pytest.mark.parametrize("embargo", [0.0, 0.03])
def test_no_splitter_ever_leaks(horizon: int, embargo: float) -> None:
    n = 60
    spans = horizon_spans(n, horizon)
    splitters = [
        PurgedKFold(5, spans=spans, embargo_pct=embargo),
        CombinatorialPurgedKFold(6, 2, spans=spans, embargo_pct=embargo),
        PurgedWalkForward(4, spans=spans),
        PurgedWalkForward(4, spans=spans, max_train_size=15),
    ]
    for cv in splitters:
        splits = list(cv.split(range(n)))
        assert len(splits) == cv.get_n_splits()
        for train, test in splits:
            assert train
            assert not set(train) & set(test)
            assert find_leaks(train, test, spans) == []


def test_find_leaks_catches_naive_kfold() -> None:
    spans = horizon_spans(20, 3)
    train = [*range(5), *range(10, 20)]
    test = list(range(5, 10))
    leaks = find_leaks(train, test, spans)
    assert (4, 5) in leaks
    assert (10, 9) in leaks
    assert all(j in train and i in test for j, i in leaks)


def test_cpcv_counts_and_paths() -> None:
    cv = CombinatorialPurgedKFold(6, 2)
    assert cv.get_n_splits() == 15
    assert cv.n_paths == 5
    combos = cv.test_groups()
    paths = cv.paths()
    assert len(paths) == 5
    used: set[tuple[int, int]] = set()
    for path in paths:
        assert [g for g, _ in path] == list(range(6))
        for g, s in path:
            assert g in combos[s]
            used.add((g, s))
    # every (group, split) test assignment is used by exactly one path
    assert len(used) == 15 * 2


def test_cpcv_test_sets_cover_groups() -> None:
    cv = CombinatorialPurgedKFold(4, 2, horizon=1)
    splits = list(cv.split(range(40)))
    train, test = splits[0]  # groups 0 and 1 -> one merged block [0, 20)
    assert test == list(range(20))
    assert train == list(range(21, 40))


def test_walk_forward_rolling_window() -> None:
    splits = list(PurgedWalkForward(3, max_train_size=5, horizon=1).split(range(40)))
    train, test = splits[-1]
    assert test == list(range(30, 40))
    assert train == list(range(25, 29))


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="sorted"):
        validate_spans([(2, 3), (1, 4)])
    with pytest.raises(ValueError, match="ends before"):
        validate_spans([(2, 1)])
    with pytest.raises(ValueError, match="either"):
        PurgedKFold(spans=[(0, 1)], horizon=2)
    with pytest.raises(ValueError, match="spans"):
        list(PurgedKFold(2, spans=horizon_spans(5)).split(range(6)))
    with pytest.raises(ValueError):
        CombinatorialPurgedKFold(4, 4)
    with pytest.raises(ValueError):
        PurgedKFold(1)


def test_matches_sklearn_when_no_overlap() -> None:
    model_selection = pytest.importorskip("sklearn.model_selection")
    n = 23
    ours = [(tr, te) for tr, te in PurgedKFold(5).split(range(n))]
    theirs = [(list(tr), list(te)) for tr, te in model_selection.KFold(5).split(range(n))]
    assert ours == theirs
    ours_wf = list(PurgedWalkForward(4).split(range(50)))
    theirs_wf = [
        (list(tr), list(te)) for tr, te in model_selection.TimeSeriesSplit(4).split(range(50))
    ]
    assert ours_wf == theirs_wf


def test_sklearn_cross_val_score_integration() -> None:
    model_selection = pytest.importorskip("sklearn.model_selection")
    linear_model = pytest.importorskip("sklearn.linear_model")
    x = [[float(i)] for i in range(100)]
    y = [2.0 * i + 1.0 for i in range(100)]
    scores = model_selection.cross_val_score(
        linear_model.LinearRegression(), x, y, cv=PurgedKFold(5, horizon=3, embargo_pct=0.01)
    )
    assert len(scores) == 5


def test_cli_plan(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--samples", "30", "--splits", "3", "--horizon", "2"]) == 0
    out = capsys.readouterr().out
    assert "leaking train/test pairs: 0" in out
    assert out.count("|") == 6
    assert main(["plan", "--method", "cpcv", "--samples", "36", "--embargo", "0.05"]) == 0
    assert "15 splits -> 5 backtest paths" in capsys.readouterr().out
    assert main(["plan", "--method", "walkforward", "--max-train", "10"]) == 0


def test_cli_rejects_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "--samples", "0"]) == 2
    assert main(["plan", "--samples", "3", "--splits", "5"]) == 2
    assert "error" in capsys.readouterr().err


def test_embargo_starts_after_label_end() -> None:
    train, test = list(PurgedKFold(4, horizon=2, embargo_pct=0.1).split(range(20)))[1]
    assert test == list(range(5, 10))
    # labels of the test fold end at 11; the 2-sample embargo then covers 12 and 13
    assert train == [0, 1, 2, *range(14, 20)]
