"""
Splitters and Cross-Validation Harnesses (Task 4 Infrastructure).
Includes Stratified K-Fold, Leave-One-Study-Out (LOSO) / Group K-Fold, and Nested CV.
"""

from typing import Generator, Tuple, Union, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


class BaseSplitterHarness:
    """Base interface for all cross-validation splitters."""
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        raise NotImplementedError


class StratifiedKFoldHarness(BaseSplitterHarness):
    """
    Stratified K-Fold cross-validation harness.
    Ensures class proportion balance across folds.
    """
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        for train_idx, val_idx in self.skf.split(X_arr, y_arr):
            yield train_idx, val_idx


class LOSOHarness(BaseSplitterHarness):
    """
    Leave-One-Study-Out (LOSO) / Group Cross-Validation Harness.
    Ensures that samples from the same study or tissue group are never split across train and test folds.
    """
    def __init__(self, n_splits: Optional[int] = None):
        self.n_splits = n_splits

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Union[np.ndarray, pd.Series] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if groups is None:
            raise ValueError("LOSOHarness requires 'groups' (e.g., study_id or tissue) to split upon.")

        groups_arr = np.asarray(groups)
        unique_groups = np.unique(groups_arr)
        
        if self.n_splits is None or self.n_splits >= len(unique_groups):
            # True Leave-One-Group-Out
            for group_val in unique_groups:
                val_idx = np.where(groups_arr == group_val)[0]
                train_idx = np.where(groups_arr != group_val)[0]
                yield train_idx, val_idx
        else:
            # GroupKFold with specified n_splits
            gkf = GroupKFold(n_splits=self.n_splits)
            X_arr = np.asarray(X)
            for train_idx, val_idx in gkf.split(X_arr, y, groups=groups_arr):
                yield train_idx, val_idx


class NestedCVHarness(BaseSplitterHarness):
    """
    Nested Cross-Validation Harness.
    Outer loop evaluates model generalization; inner loop yields validation folds for tuning.
    """
    def __init__(
        self,
        outer_splits: int = 5,
        inner_splits: int = 3,
        random_state: int = 42
    ):
        self.outer_splits = outer_splits
        self.inner_splits = inner_splits
        self.random_state = random_state

    def split_nested(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Generator[Tuple[np.ndarray, np.ndarray, Generator[Tuple[np.ndarray, np.ndarray], None, None]], None, None]:
        """
        Yields (outer_train_idx, outer_test_idx, inner_splits_generator)
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        
        outer_skf = StratifiedKFold(n_splits=self.outer_splits, shuffle=True, random_state=self.random_state)
        
        for outer_train_idx, outer_test_idx in outer_skf.split(X_arr, y_arr):
            X_inner = X_arr[outer_train_idx]
            y_inner = y_arr[outer_train_idx]
            
            inner_skf = StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.random_state)
            inner_gen = (
                (outer_train_idx[in_tr], outer_train_idx[in_val])
                for in_tr, in_val in inner_skf.split(X_inner, y_inner)
            )
            
            yield outer_train_idx, outer_test_idx, inner_gen
