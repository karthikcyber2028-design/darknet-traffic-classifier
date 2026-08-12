import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src import config, data  # noqa: E402

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

MODELS_DIR = config.MODELS_DIR
REPORTS_DIR = config.REPORTS_DIR
BUNDLE_CACHE = {}
REGISTRY = {}


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def build_registry():
    reg = {}
    for run_dir in REPORTS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        metrics = _read_json(run_dir / "metrics.json")
        if not metrics or "target" not in metrics or "model" not in metrics:
            continue
        target = metrics["target"]
        model = metrics["model"]
        entry = {
            "run_dir": run_dir.name,
            "metrics": metrics,
            "images": [p.name for p in sorted(run_dir.glob("*.png"))],
            "report_text": None,
        }
        txt = run_dir / "classification_report.txt"
        if txt.exists():
            entry["report_text"] = txt.read_text(encoding="utf-8")
        reg.setdefault(target, {})[model] = entry
    # keep only the most recent run per target/model by timestamp
    for target, models in reg.items():
        for model, entry in models.items():
            pass
    return reg


def get_registry():
    global REGISTRY
    if not REGISTRY:
        REGISTRY = build_registry()
    return REGISTRY


def load_bundle(target, model):
    key = (target, model)
    if key in BUNDLE_CACHE:
        return BUNDLE_CACHE[key]
    path = MODELS_DIR / target / f"{model}_best.joblib"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    BUNDLE_CACHE[key] = bundle
    return bundle


def available_targets():
    reg = get_registry()
    targets = []
    for t in config.TARGETS:
        models = reg.get(t)
        targets.append({
            "name": t,
            "description": config.TARGET_DESCRIPTIONS.get(t, ""),
            "models": sorted(models.keys()) if models else [],
        })
    return targets


@app.get("/")
def index():
    return render_template("index.html", active="predict")


@app.get("/api/status")
def api_status():
    reg = get_registry()
    real = config.DATASET_FILE.exists()
    synth_rows = 0
    if (config.DATA_DIR / "synth_darknet.csv").exists():
        try:
            synth_rows = len(pd.read_csv(config.DATA_DIR / "synth_darknet.csv", nrows=1000000))
        except Exception:
            synth_rows = 0
    return jsonify({
        "targets": available_targets(),
        "models": config.MODELS,
        "model_help": {
            "rf": "Random Forest",
            "gb": "Histogram Gradient Boosting",
            "lr": "L2 Logistic Regression",
        },
        "dataset_source": "real" if real else "synthetic",
        "dataset_path": str(config.DATASET_FILE),
        "synth_rows": synth_rows,
        "trained_runs": {t: list(m.keys()) for t, m in reg.items()},
    })


@app.get("/api/features")
def api_features():
    target = request.args.get("target")
    model = request.args.get("model")
    bundle = load_bundle(target, model)
    if bundle is None:
        return jsonify({"error": f"No trained model for {target}/{model}"}), 404
    cols = bundle["feature_columns"]
    medians = bundle["preprocessor"].fill_values
    defaults = {c: (float(medians[c]) if c in medians else 0.0) for c in cols}
    return jsonify({
        "target": target,
        "model": model,
        "classes": bundle["classes"],
        "features": [{"name": c, "default": round(defaults[c], 6)} for c in cols],
    })


@app.get("/api/sample")
def api_sample():
    target = request.args.get("target")
    model = request.args.get("model")
    bundle = load_bundle(target, model)
    if bundle is None:
        return jsonify({"error": f"No trained model for {target}/{model}"}), 404

    row = data.generate_synthetic(n=1).iloc[0]
    truth_label = _normalize_label_for_target(row.get("label"), target)
    truth_type = row.get("type", "")
    features = {}
    medians = bundle["preprocessor"].fill_values
    for c in bundle["feature_columns"]:
        if c in row:
            v = row[c]
            if isinstance(v, (int, float)) and pd.notna(v):
                features[c] = round(float(v), 6)
                continue
        features[c] = round(float(medians[c]), 6) if c in medians else 0.0
    return jsonify({
        "features": features,
        "truth_label": truth_label,
        "truth_type": str(truth_type),
    })


def _normalize_label_for_target(label, target):
    s = str(label).lower().replace("-", "").replace("_", "").replace(" ", "")
    if target == "label4":
        return {"tor": "tor", "vpn": "vpn", "nontor": "nontor", "nonvpn": "nonvpn"}.get(s, s)
    if target == "tor_binary":
        return "tor" if s == "tor" else "non-tor"
    if target == "darknet_binary":
        return "darknet" if s in ("tor", "vpn") else "benign"
    return str(label)


def class_label_map(bundle):
    """Map raw pipeline labels (e.g. 0/1 or 'tor') to display class names."""
    classes = bundle["classes"]
    raw = list(bundle["pipeline"].classes_)
    numeric = raw and all(isinstance(c, (int, np.integer, float)) for c in raw)
    if numeric:
        return {int(c): classes[i] for i, c in enumerate(raw)}, True
    return {str(c): str(c) for c in raw}, False


def _label_key(label, numeric):
    try:
        return int(label) if numeric else str(label)
    except (TypeError, ValueError):
        return str(label)


def predict_bundle(bundle, df):
    X = bundle["preprocessor"].transform(df)
    y_pred = bundle["pipeline"].predict(X)
    classes = bundle["classes"]
    label_map, numeric = class_label_map(bundle)
    probs = None
    if len(classes) == 2:
        prob_pos = bundle["pipeline"].predict_proba(X)[:, 1]
        probs = {classes[0]: round(float(1 - prob_pos[0]), 6),
                 classes[1]: round(float(prob_pos[0]), 6)}
    return {
        "prediction": label_map.get(_label_key(y_pred[0], numeric), str(y_pred[0])),
        "classes": classes,
        "probabilities": probs,
    }


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    target = payload.get("target")
    model = payload.get("model")
    values = payload.get("values") or {}
    bundle = load_bundle(target, model)
    if bundle is None:
        return jsonify({"error": f"No trained model for {target}/{model}"}), 404
    try:
        floats = {str(k): float(v) for k, v in values.items()}
    except (TypeError, ValueError):
        return jsonify({"error": "All feature values must be numeric"}), 400
    df = pd.DataFrame([floats])
    result = predict_bundle(bundle, df)
    return jsonify({"target": target, "model": model, **result})


@app.post("/api/predict-csv")
def api_predict_csv():
    target = request.form.get("target")
    model = request.form.get("model")
    file = request.files.get("file")
    bundle = load_bundle(target, model)
    if bundle is None:
        return jsonify({"error": f"No trained model for {target}/{model}"}), 404
    if file is None or not file.filename:
        return jsonify({"error": "No CSV file uploaded"}), 400

    df = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8", errors="replace")),
                     low_memory=False)
    df = data.normalize_columns(df)
    X = bundle["preprocessor"].transform(df)
    y_pred = bundle["pipeline"].predict(X)
    classes = bundle["classes"]
    label_map, numeric = class_label_map(bundle)

    result = df.copy()
    result["prediction"] = [label_map.get(_label_key(v, numeric), str(v)) for v in y_pred]
    if len(classes) == 2:
        prob_pos = bundle["pipeline"].predict_proba(X)[:, 1]
        result[f"prob_{classes[0]}"] = 1.0 - prob_pos
        result[f"prob_{classes[1]}"] = prob_pos

    preview = result.head(50).where(pd.notna(result), None)
    cols = [c for c in result.columns]
    rows = [list(r) for r in preview.itertuples(index=False, name=None)]
    rows = [[float(v) if isinstance(v, (int, float)) else v for v in r] for r in rows]

    counts = result["prediction"].value_counts()
    return jsonify({
        "target": target,
        "model": model,
        "classes": classes,
        "n_rows": int(len(result)),
        "columns": cols,
        "rows": rows,
        "summary": {str(k): int(v) for k, v in counts.items()},
    })


@app.get("/api/analytics")
def api_analytics():
    target = request.args.get("target")
    model = request.args.get("model")
    reg = get_registry()
    entry = (reg.get(target) or {}).get(model)
    if entry is None:
        return jsonify({"error": f"No analytics for {target}/{model}"}), 404
    return jsonify({
        "target": target,
        "model": model,
        "description": config.TARGET_DESCRIPTIONS.get(target, ""),
        "run_dir": entry["run_dir"],
        "metrics": entry["metrics"],
        "images": [{"name": n, "url": f"/report/{entry['run_dir']}/{n}"}
                   for n in entry["images"]],
        "report_text": entry["report_text"],
    })


@app.get("/api/compare")
def api_compare():
    csv_path = REPORTS_DIR / "compare_results.csv"
    if not csv_path.exists():
        return jsonify({"rows": [], "error": "No compare_results.csv yet"})
    df = pd.read_csv(csv_path)
    return jsonify({
        "columns": list(df.columns),
        "rows": df.fillna("").values.tolist(),
    })


@app.get("/report/<run_dir>/<filename>")
def report_file(run_dir, filename):
    reg = get_registry()
    safe = any(entry["run_dir"] == run_dir
               for target in reg.values() for entry in target.values())
    if not safe or not Path(filename).name == filename:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(REPORTS_DIR / run_dir, filename)


if __name__ == "__main__":
    import os
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=8000, debug=debug, use_reloader=debug)
