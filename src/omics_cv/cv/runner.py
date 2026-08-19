"""
Unified Cross-Validation Execution Engine (Task 4 Infrastructure).
Orchestrates model execution, out-of-fold prediction tracking, and aggregate metrics.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .splitters import BaseSplitterHarness
from .evaluator import MetricsEvaluator


class CrossValidationRunner:
    """
    Executes cross-validation workflows using any splitter harness and model factory/trainer.
    """

    def __init__(
        self,
        splitter: BaseSplitterHarness,
        task_type: str = "classification"
    ):
        self.splitter = splitter
        self.task_type = task_type

    def run(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        model_factory: Callable[[], Any],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Dict[str, Any]:
        """
        Runs CV splits, trains model, evaluates metrics per fold, and computes out-of-fold (OOF) aggregate performance.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        oof_preds = np.zeros(len(y_arr), dtype=float)
        fold_metrics: List[Dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.splitter.split(X_arr, y_arr, groups=groups)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]

            model = model_factory()
            model.fit(X_tr, y_tr)

            if self.task_type == "classification":
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_val)
                    val_preds = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
                else:
                    val_preds = model.predict(X_val)
                oof_preds[val_idx] = val_preds
                fold_m = MetricsEvaluator.evaluate_classification(y_val, val_preds)
            else:
                val_preds = model.predict(X_val)
                oof_preds[val_idx] = val_preds
                fold_m = MetricsEvaluator.evaluate_regression(y_val, val_preds)

            fold_m["fold"] = fold_idx
            fold_metrics.append(fold_m)

        # Aggregate OOF metrics
        if self.task_type == "classification":
            oof_overall = MetricsEvaluator.evaluate_classification(y_arr, oof_preds)
        else:
            oof_overall = MetricsEvaluator.evaluate_regression(y_arr, oof_preds)

        return {
            "oof_overall": oof_overall,
            "fold_metrics": fold_metrics,
            "oof_predictions": oof_preds
        }
