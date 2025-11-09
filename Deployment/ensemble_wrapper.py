import numpy as np
import pandas as pd


class EnsembleProbClassifier:
    """A simple predict_proba/predict wrapper that averages two binary classifiers.

    - Assumes both underlying models implement predict_proba(X) -> (n_samples, 2)
    - Uses provided weights for the positive-class probability (column 1)
    - Works with pandas DataFrame (preferred) or numpy arrays
    - If DataFrame is passed, will align features using model.feature_names_in_ (if available)
    """

    def __init__(self, model_a, model_b, weight_a=0.5, weight_b=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = float(weight_a)
        self.weight_b = float(weight_b)
        self.feature_names_in_ = None
        a_feats = getattr(model_a, "feature_names_in_", None)
        b_feats = getattr(model_b, "feature_names_in_", None)
        if a_feats is not None and b_feats is not None:
            inter = [c for c in a_feats if c in set(b_feats)]
            self.feature_names_in_ = np.array(inter)

    def _prepare_X(self, X):
        if self.feature_names_in_ is not None and isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names_in_ if c in X.columns]
            return X[cols]
        return X

    def predict_proba(self, X):
        Xp = self._prepare_X(X)
        proba_a = self.model_a.predict_proba(Xp)
        proba_b = self.model_b.predict_proba(Xp)
        pos = self.weight_a * proba_a[:, 1] + self.weight_b * proba_b[:, 1]
        pos = np.clip(pos, 0.0, 1.0)
        neg = 1.0 - pos
        return np.vstack([neg, pos]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


