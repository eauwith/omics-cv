"""
Metrics Evaluator Harness for standardized evaluation across all models (GNN, ProteinChat, KG).
"""

from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


class MetricsEvaluator:
    """
    Standardized metrics evaluator supporting classification, regression, and tissue-stratified evaluations.
    """

    @staticmethod
    def evaluate_classification(
        y_true: Union[np.ndarray, list],
        y_prob: Union[np.ndarray, list],
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculates AUROC, AUPRC, F1, Accuracy, Precision, and Recall."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0))
        }

        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["auroc"] = float("nan")

        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["auprc"] = float("nan")

        return metrics

    @staticmethod
    def evaluate_regression(
        y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list]
    ) -> Dict[str, float]:
        """Calculates RMSE, MAE, R2, Spearman correlation, and Pearson correlation."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        s_corr, _ = spearmanr(y_true, y_pred)
        p_corr, _ = pearsonr(y_true, y_pred)

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "spearman_rho": float(s_corr) if not np.isnan(s_corr) else 0.0,
            "pearson_r": float(p_corr) if not np.isnan(p_corr) else 0.0
        }

    @classmethod
    def evaluate_stratified(
        cls,
        df: pd.DataFrame,
        group_col: str,
        target_col: str,
        pred_col: str,
        task_type: str = "classification"
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates metrics grouped by a column (e.g. tissue or study ID).
        """
        results = {}
        for group_val, sub_df in df.groupby(group_col):
            if len(sub_df) < 2:
                continue
            if task_type == "classification":
                results[str(group_val)] = cls.evaluate_classification(sub_df[target_col], sub_df[pred_col])
            else:
                results[str(group_val)] = cls.evaluate_regression(sub_df[target_col], sub_df[pred_col])
        return results
