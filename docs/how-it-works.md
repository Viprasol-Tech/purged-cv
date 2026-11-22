# How purged-cv works

## Label spans

Financial labels look into the future. A label built from the return over the
next 10 bars uses prices from bar `i` to bar `i + 10`. purged-cv gives every
sample a **span** `(t0, t1)`: the first and last moment its label depends on.

- `horizon=10` is a shortcut for `spans = [(i, i + 10) for i in range(n)]`.
- For event-based labels (triple barrier, variable holding periods), pass your
  own `spans` list. Times can be bar numbers or timestamps (`datetime.timestamp()`),
  and the list must be sorted by `t0`.

## Purging

A training sample leaks into a test fold when the two spans overlap: the model
has seen part of the future it is being tested on. For each test block the
splitter computes the block's window `[min t0, max t1]` and drops every training
sample whose span touches it. Touching ends count as overlap.

## Embargo

Purging removes direct overlap, but features such as moving averages carry
information a little further. The embargo also drops the first
`ceil(n * embargo_pct)` samples that start after the test labels end. This
follows López de Prado, *Advances in Financial Machine Learning*, chapter 7.
Training data before a test block needs no embargo, since only purging applies there.

## Combinatorial purged CV (CPCV)

The series is cut into `N` groups. Every choice of `k` test groups is one split,
which gives `C(N, k)` splits. Each group is tested in `C(N-1, k-1)` of them, so
the out-of-sample predictions can be stitched into that many full-history
**backtest paths**. `CombinatorialPurgedKFold.paths()` returns, for each path,
the `(group, split_index)` pairs to stitch. You get a distribution of Sharpe
ratios instead of one number. Pair it with
[deflated-sharpe](https://github.com/Viprasol-Tech/deflated-sharpe) to judge it.

## Auditing

`find_leaks(train, test, spans)` lists every overlapping `(train, test)` pair.
It works on any split, including ones made by other libraries. An empty list
means the split is clean.
