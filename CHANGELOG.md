# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[SemVer](https://semver.org/).

## [0.1.0] - 2026-11-22

### Added
- `PurgedKFold` with label-span purging and an embargo after each test fold.
- `CombinatorialPurgedKFold` (CPCV) with `n_paths` and `paths()` for stitching
  full out-of-sample backtest paths.
- `PurgedWalkForward` (expanding or rolling) with purging before each test block.
- `find_leaks` to audit any train/test split for overlapping label spans.
- `purgedcv plan` CLI that draws each split and counts leaking pairs.
- scikit-learn compatible `split` / `get_n_splits` on every splitter.
