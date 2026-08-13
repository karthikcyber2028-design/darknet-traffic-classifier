from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config

MODEL_HELP = {
    "rf": "Random Forest",
    "gb": "Histogram Gradient Boosting",
    "xgb": "XGBoost",
    "lr": "L2 Logistic Regression",
}


class LabelEncodedXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost 2.x requires class labels in [0, n_classes); this wrapper
    LabelEncodes arbitrary string labels on fit and decodes on predict so the
    rest of the pipeline (reports, webapp) can keep using display strings."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.estimator = XGBClassifier(**kwargs)
        self.encoder = LabelEncoder()

    def fit(self, X, y):
        self.encoder.fit(y)
        self.classes_ = self.encoder.classes_
        self.estimator.fit(X, self.encoder.transform(y))
        return self

    def predict(self, X):
        return self.encoder.inverse_transform(self.estimator.predict(X))

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def get_params(self, deep=True):
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        self.estimator.set_params(**params)
        self.kwargs.update(params)
        return self


def build_pipeline(model, seed=config.RANDOM_SEED):
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=250, class_weight="balanced", n_jobs=-1,
            random_state=seed, min_samples_leaf=1,
        )
        return Pipeline([("clf", clf)])
    if model == "gb":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, random_state=seed,
        )
        return Pipeline([("clf", clf)])
    if model == "xgb":
        clf = LabelEncodedXGBClassifier(
            n_estimators=300, learning_rate=0.08, max_depth=6,
            eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
        return Pipeline([("clf", clf)])
    if model == "lr":
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs",
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model: {model}")
