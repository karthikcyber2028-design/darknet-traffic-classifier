# Darknet Traffic Classification from Tor Network

Classifies network flow records into Tor / darknet traffic vs benign traffic using
supervised machine learning. Trained on the [CIC-Darknet2020](https://www.unb.ca/cic/datasets/darknet2020.html)
dataset format (CICFlowMeter features) with Random Forest, HistGradientBoosting and
Logistic Regression baselines.

## Tasks (targets)

| Target         | Task                                                          |
|----------------|---------------------------------------------------------------|
| `tor_binary`   | Tor vs everything else                                        |
| `darknet_binary` | Darknet (Tor + VPN) vs benign (NonTor + NonVPN)             |
| `label4`       | 4-class: Tor / VPN / NonTor / NonVPN                          |
| `apptype`      | Application type (Browsing, Chat, Email, File Transfer, P2P, Audio, Video, VoIP) |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset

Download `Darknet.csv` from the CIC-Darknet2020 page and place it at `data/Darknet.csv`.
The header is auto-normalized (spaces, casing), so the Kaggle mirror works too.

If the file is missing, `train` / `compare` automatically fall back to a **synthetic
demo dataset** so the whole pipeline can be exercised without the real data. Use
`synth` to generate it explicitly.

## Usage

```powershell
.\.venv\Scripts\python.exe main.py info
.\.venv\Scripts\python.exe main.py train --target tor_binary --model rf
.\.venv\Scripts\python.exe main.py train --data data\Darknet.csv --target label4 --model gb
.\.venv\Scripts\python.exe main.py compare --quick            # all targets x all models
.\.venv\Scripts\python.exe main.py predict --csv flows.csv --out predictions.csv
.\.venv\Scripts\python.exe main.py synth --n 8000             # demo data
```

`train` saves a model bundle (`<model>_best.joblib`) plus a metrics JSON, confusion
matrix, ROC/PR curves, and feature-importance plot under `reports/`. `predict` loads
the most recent bundle (or one you pass with `--model`) and appends `prediction` and
probability columns.

## Web app

A Flask web app (`webapp/`) wraps the trained models with an interactive UI:

```powershell
.\.venv\Scripts\python.exe webapp\app.py
# open http://127.0.0.1:8000
```

Features:

- **Predict** – choose a target (`tor_binary`, `darknet_binary`, `label4`, `apptype`)
  and model (RF / Gradient Boosting / LR), edit the flow features (defaults = training
  medians), or load a random sample row, and get a prediction with probabilities.
- **Batch CSV** – upload a CSV of flow records and classify them all at once.
- **Analytics** – per target/model: accuracy, F1, ROC-AUC, confusion matrix,
  ROC/PR curves, feature importance, and the full classification report.
- **Model Comparison** – the cross-model result table from `compare`.

Set `FLASK_DEBUG=0` to run without the auto-reloader/debugger.

## Pipeline

1. **Loading** – robust header normalization, label/type column detection.
2. **Preprocessing** – drop non-numeric / inf, drop high-missing and constant columns,
   drop correlated columns, median imputation, optional top-K mutual-information selection.
3. **Modeling** – RF, HistGradientBoosting, L2 Logistic Regression (stratified split,
   balanced classes).
4. **Evaluation** – accuracy, F1 / precision / recall, ROC-AUC, AP, confusion matrix,
   classification report.
