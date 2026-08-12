from pathlib import Path

import joblib
import pandas as pd

from . import data


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    return joblib.load(path)


def predict_csv(bundle, csv_path, out_path=None):
    df = data.load_dataset(csv_path)
    X = bundle["preprocessor"].transform(df)
    y_pred = bundle["pipeline"].predict(X)
    classes = bundle["classes"]

    result = df.copy()
    result["prediction"] = y_pred

    if len(classes) == 2 and hasattr(bundle["pipeline"], "predict_proba"):
        prob = bundle["pipeline"].predict_proba(X)[:, 1]
        result[f"prob_{classes[0]}"] = 1.0 - prob
        result[f"prob_{classes[1]}"] = prob

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out_path, index=False)

    summary = result["prediction"].value_counts().to_dict()
    return result, summary
