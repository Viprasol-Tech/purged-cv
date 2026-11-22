"""purged-cv — leak-free cross-validation for financial machine learning.

Part of purged-cv by Viprasol Tech Private Limited (https://viprasol.com).
"""

from ._core import Span, find_leaks, horizon_spans, purged_train, validate_spans
from .splitters import CombinatorialPurgedKFold, PurgedKFold, PurgedWalkForward

__all__ = [
    "CombinatorialPurgedKFold",
    "PurgedKFold",
    "PurgedWalkForward",
    "Span",
    "find_leaks",
    "horizon_spans",
    "purged_train",
    "validate_spans",
]
__version__ = "0.1.0"
