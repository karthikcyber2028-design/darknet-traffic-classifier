import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from . import config


def evaluate(model, X_test, y_test, classes, target, model_name, out_dir, tag=""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = float(model.score(X_test, y_test))
    metrics = {
        "target": target,
        "model": model_name,
        "tag": tag,
        "accuracy": acc,
        "n_features": int(X_test.shape[1]),
        "n_test_samples": int(len(y_test)),
    }

    pos = len(classes) == 2
    if pos:
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
        metrics["average_precision"] = float(average_precision_score(y_test, y_prob[:, 1]))
        _plot_binary_curves(y_test, y_prob[:, 1], out_dir)
    else:
        metrics["precision_macro"] = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["recall_macro"] = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    report = classification_report(y_test, y_pred, target_names=classes, zero_division=0)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    _plot_confusion(y_test, y_pred, classes, out_dir)
    _plot_importances(model, X_test, out_dir)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def _plot_confusion(y_test, y_pred, classes, out_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(classes) * 0.9), max(4, len(classes) * 0.8)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes,
                yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_binary_curves(y_test, y_prob_pos, out_dir):
    fpr, tpr, _ = roc_curve(y_test, y_prob_pos)
    prec, rec, _ = precision_recall_curve(y_test, y_prob_pos)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_pos):.3f}")
    axes[0].plot([0, 1], [0, 1], ls="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y_test, y_prob_pos):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "roc_pr_curves.png", dpi=120)
    plt.close(fig)


def _plot_importances(model, X_test, out_dir):
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return
    imp = np.asarray(clf.feature_importances_)
    if imp.size != X_test.shape[1]:
        return
    idx = np.argsort(imp)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([X_test.columns[i] for i in idx[::-1]], imp[idx][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importances.png", dpi=120)
    plt.close(fig)


def format_metrics_line(metrics):
    m = metrics
    if "f1" in m:
        return (f"{m['model']:<3} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  auc={m['roc_auc']:.4f}")
    return (f"{m['model']:<3} acc={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
            f"prec_macro={m['precision_macro']:.4f}  rec_macro={m['recall_macro']:.4f}")
