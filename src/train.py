import datetime
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config, data, evaluate, models
from .preprocessing import Preprocessor


def run_training(df, target, model_name, test_size=config.TEST_SIZE, seed=config.RANDOM_SEED,
                 top_k=None, tag="", source="real"):
    y, classes = data.make_target(df, target)
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class present after target mapping; nothing to learn")

    X = df
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )

    pre = Preprocessor(top_k=top_k, seed=seed).fit(X_train, y_train)
    Xtr = pre.transform(X_train)
    Xte = pre.transform(X_test)

    if Xtr.shape[1] == 0:
        raise ValueError("No usable numeric features after preprocessing")

    pipe = models.build_pipeline(model_name, seed=seed)
    pipe.fit(Xtr, y_train)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stamp = f"{tag}_{ts}" if tag else ts
    model_dir = config.MODELS_DIR / target
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir = config.REPORTS_DIR / f"{target}_{model_name}_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate.evaluate(pipe, Xte, y_test, classes, target, model_name,
                                report_dir, tag=tag)
    metrics.update({
        "seed": seed,
        "test_size": test_size,
        "top_k": top_k,
        "source": source,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": len(classes),
        "classes": classes,
        "timestamp": ts,
        "model_file": f"{model_name}_{stamp}.joblib",
    })

    bundle = {
        "pipeline": pipe,
        "preprocessor": pre,
        "feature_columns": pre.columns,
        "classes": classes,
        "target": target,
        "model": model_name,
        "metrics": metrics,
    }
    bundle_path = model_dir / f"{model_name}_{stamp}.joblib"
    joblib.dump(bundle, bundle_path)
    with open(bundle_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    best = model_dir / f"{model_name}_best.joblib"
    joblib.dump(bundle, best)

    print(f"  target       : {target} ({config.TARGET_DESCRIPTIONS.get(target, '')})")
    print(f"  dataset      : {source}")
    print(f"  train/test   : {len(y_train)} / {len(y_test)}")
    print(f"  features     : {Xtr.shape[1]}")
    print(f"  classes      : {classes}")
    print(f"  result       : {evaluate.format_metrics_line(metrics)}")
    print(f"  report       : {report_dir}")
    print(f"  model saved  : {bundle_path}")
    return bundle, metrics


def find_latest_bundle(target=None):
    pattern = config.MODELS_DIR / "**" / "*_best.joblib"
    bundles = sorted(Path(config.MODELS_DIR).glob("**/*.joblib"))
    if not bundles:
        return None
    if target:
        bundles = [b for b in bundles if b.parent.name == target]
        if not bundles:
            return None
    return max(bundles, key=lambda p: p.stat().st_mtime)
