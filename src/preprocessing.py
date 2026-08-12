import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from . import config


class Preprocessor:
    def __init__(self, max_missing_ratio=config.MAX_MISSING_RATIO,
                 corr_threshold=config.CORR_THRESHOLD, top_k=None, seed=config.RANDOM_SEED):
        self.max_missing_ratio = max_missing_ratio
        self.corr_threshold = corr_threshold
        self.top_k = top_k
        self.seed = seed
        self.columns = None
        self.fill_values = None
        self.dropped = {}

    def fit(self, X, y=None):
        X = self._clean(X)
        missing = X.isnull().mean()
        self.dropped["high_missing"] = list(X.columns[missing > self.max_missing_ratio])
        X = X.loc[:, missing <= self.max_missing_ratio]

        const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
        self.dropped["constant"] = const
        X = X.drop(columns=const)

        self.fill_values = X.median()
        X = X.fillna(self.fill_values)

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high = [c for c in upper.columns if any(upper[c] > self.corr_threshold)]
        self.dropped["correlated"] = high
        X = X.drop(columns=high)

        self.columns = list(X.columns)

        if self.top_k and y is not None and self.top_k < len(self.columns):
            mi = mutual_info_classif(X, y, random_state=self.seed, n_neighbors=3)
            keep = np.argsort(mi)[::-1][: self.top_k]
            self.columns = [X.columns[i] for i in keep]
            self.dropped["top_k_cut"] = len(X.columns) - self.top_k
            X = X[self.columns]
        return self

    def transform(self, X):
        if self.columns is None:
            raise RuntimeError("fit() must be called before transform()")
        X = self._clean(X)
        X = X.reindex(columns=self.columns)
        X = X.fillna(self.fill_values.reindex(self.columns))
        X = X.astype(float)
        return X

    def _clean(self, X):
        X = X.copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.select_dtypes(include=[np.number])
        return X
