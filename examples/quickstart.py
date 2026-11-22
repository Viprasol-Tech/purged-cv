"""Leakage demo: naive K-fold vs purged K-fold on forward-return labels.

Part of purged-cv by Viprasol Tech Private Limited (https://viprasol.com).
Run: python examples/quickstart.py
"""

from purged_cv import CombinatorialPurgedKFold, PurgedKFold, find_leaks, horizon_spans

N, HORIZON = 500, 10  # 500 bars, each label = return over the next 10 bars
spans = horizon_spans(N, HORIZON)

# Naive K-fold: train is simply "everything that is not test"
naive_leaks = 0
fold = N // 5
for k in range(5):
    test = list(range(k * fold, (k + 1) * fold))
    train = [i for i in range(N) if i not in set(test)]
    naive_leaks += len(find_leaks(train, test, spans))
print(f"naive 5-fold leaking pairs:  {naive_leaks}")

cv = PurgedKFold(5, spans=spans, embargo_pct=0.01)
purged_leaks = sum(len(find_leaks(tr, te, spans)) for tr, te in cv.split(range(N)))
print(f"purged 5-fold leaking pairs: {purged_leaks}")

cpcv = CombinatorialPurgedKFold(6, 2, spans=spans, embargo_pct=0.01)
print(f"CPCV(6, 2): {cpcv.get_n_splits()} splits -> {cpcv.n_paths} backtest paths")
print("path 0 uses (group, split):", cpcv.paths()[0])
