"""
Infrastructure module for Task 4: Reusable CV harnesses and Autoencoder Representation Learning.
"""

from .cv.splitters import StratifiedKFoldHarness, LOSOHarness, NestedCVHarness
from .cv.evaluator import MetricsEvaluator
from .embeddings.autoencoder import OmicsAutoencoder, VariationalOmicsAutoencoder

__all__ = [
    "StratifiedKFoldHarness",
    "LOSOHarness",
    "NestedCVHarness",
    "MetricsEvaluator",
    "OmicsAutoencoder",
    "VariationalOmicsAutoencoder"
]
