from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config

MODEL_HELP = {
    "rf": "Random Forest",
    "gb": "Histogram Gradient Boosting",
    "lr": "L2 Logistic Regression",
}


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
    if model == "lr":
        clf = LogisticRegression(
            max_iter=3000, class_weight="balanced", solver="lbfgs",
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model: {model}")
