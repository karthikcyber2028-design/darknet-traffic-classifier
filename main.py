import argparse
import json
import sys

from src import config, data, predict as predict_mod, train
from src import evaluate as evaluate_mod


def _ensure_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_train(args):
    df, source = data.load_or_synthesize(args.data)
    print(f"  Loaded dataset ({source}): {df.shape[0]} rows x {df.shape[1]} cols")
    if source == "synthetic":
        print("  NOTE: real Darknet.csv not found - using synthetic demo data")
    bundle, metrics = train.run_training(
        df, args.target, args.model, test_size=args.test_size,
        seed=args.seed, top_k=args.top_k, tag=args.tag, source=source,
    )


def cmd_predict(args):
    if not args.model:
        args.model = train.find_latest_bundle()
        if not args.model:
            print("  No trained model found. Run `train` first.")
            return
    bundle = predict_mod.load_bundle(args.model)
    print(f"  Model   : {args.model}")
    print(f"  Target  : {bundle['target']}  classes={bundle['classes']}")
    result, summary = predict_mod.predict_csv(bundle, args.csv, args.out)
    print(f"  Predictions: {json.dumps(summary)}")
    if args.out:
        print(f"  Saved   : {args.out}")


def cmd_synth(args):
    out = data.save_synthetic(args.out, n=args.n)
    print(f"  Synthetic dataset written to {out} ({args.n} rows)")


def cmd_compare(args):
    df, source = data.load_or_synthesize(args.data)
    print(f"  Loaded dataset ({source}): {df.shape[0]} rows x {df.shape[1]} cols")
    if source == "synthetic":
        print("  NOTE: real Darknet.csv not found - using synthetic demo data")

    if args.quick and df.shape[0] > 5000:
        df = df.sample(5000, random_state=args.seed).reset_index(drop=True)
        print("  quick mode: subsampled to 5000 rows")

    targets = args.targets.split(",") if args.targets else config.TARGETS
    models = args.models.split(",") if args.models else config.MODELS

    all_rows = []
    for t in targets:
        for m in models:
            print(f"\n=== {t} / {m} ===")
            try:
                bundle, metrics = train.run_training(
                    df, t, m, test_size=args.test_size, seed=args.seed,
                    top_k=args.top_k, tag="compare", source=source,
                )
                all_rows.append(metrics)
            except ValueError as exc:
                print(f"  skipped ({exc})")

    if all_rows:
        table = [{
            "target": r["target"], "model": r["model"], "accuracy": r["accuracy"],
            "f1": r.get("f1"), "f1_macro": r.get("f1_macro"),
            "roc_auc": r.get("roc_auc"), "precision": r.get("precision"),
            "recall": r.get("recall"), "n_features": r["n_features"],
        } for r in all_rows]
        import pandas as pd
        out_csv = config.REPORTS_DIR / "compare_results.csv"
        pd.DataFrame(table).to_csv(out_csv, index=False)
        print(f"\nComparison table saved: {out_csv}")


def cmd_info(args):
    print(f"  Project dir : {config.BASE_DIR}")
    print(f"  Data dir    : {config.DATA_DIR}")
    print(f"  Models dir  : {config.MODELS_DIR}")
    print(f"  Reports dir : {config.REPORTS_DIR}")
    real = config.DATASET_FILE.exists()
    print(f"  Darknet.csv : {'present' if real else 'NOT found (synthetic fallback active)'}")
    print(f"  Targets     : {config.TARGETS}")
    print(f"  Models      : {config.MODELS}")
    latest = train.find_latest_bundle()
    print(f"  Latest model: {latest if latest else 'none trained yet'}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="darknet-classifier",
        description="Darknet (Tor / VPN) traffic classification from the CIC-Darknet2020 dataset",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train and save a model")
    p_train.add_argument("--data", default=None, help="Path to Darknet CSV (default: data/Darknet.csv)")
    p_train.add_argument("--target", default=config.DEFAULT_TARGET, choices=config.TARGETS)
    p_train.add_argument("--model", default="rf", choices=config.MODELS)
    p_train.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p_train.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_train.add_argument("--top-k", type=int, default=None, help="Select top-K features by mutual info")
    p_train.add_argument("--tag", default="", help="Optional tag for output filenames")
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Run inference on a CSV of flows")
    p_pred.add_argument("--csv", required=True)
    p_pred.add_argument("--model", default=None, help="Path to a .joblib bundle (default: latest)")
    p_pred.add_argument("--out", default=None, help="Output CSV path")
    p_pred.set_defaults(func=cmd_predict)

    p_syn = sub.add_parser("synth", help="Generate a synthetic demo dataset")
    p_syn.add_argument("--out", default=str(config.DATA_DIR / "synth_darknet.csv"))
    p_syn.add_argument("--n", type=int, default=8000)
    p_syn.set_defaults(func=cmd_synth)

    p_cmp = sub.add_parser("compare", help="Train all models/targets and compare")
    p_cmp.add_argument("--data", default=None)
    p_cmp.add_argument("--targets", default=None, help="Comma-separated targets")
    p_cmp.add_argument("--models", default=None, help="Comma-separated models")
    p_cmp.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    p_cmp.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    p_cmp.add_argument("--top-k", type=int, default=None)
    p_cmp.add_argument("--quick", action="store_true", help="Subsample to 5000 rows")
    p_cmp.add_argument("--tag", default="compare")
    p_cmp.set_defaults(func=cmd_compare)

    p_info = sub.add_parser("info", help="Show project paths and status")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    _ensure_utf8()
    try:
        args.func(args)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
