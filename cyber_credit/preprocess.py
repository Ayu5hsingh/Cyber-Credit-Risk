"""Preprocessing shared by 03_features and 04_modelling (and saved inside the model)."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st and 99th percentile, learned on the training fold."""
    def fit(self, X, y=None):
        X = pd.DataFrame(X); self.lo_, self.hi_ = X.quantile(0.01), X.quantile(0.99); return self
    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1).values
