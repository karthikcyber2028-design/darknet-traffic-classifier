import numpy as np
import pandas as pd
from pathlib import Path

from . import config

APP_TYPE_ORDER = [
    "Audio", "Browsing", "Chat", "Email", "File_Transfer", "P2P", "Video", "VoIP",
]

SYNTH_FEATURES = [
    "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "totlen_fwd_pkts", "totlen_bwd_pkts",
    "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
    "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
    "flow_byts_s", "flow_pkts_s", "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_header_len", "bwd_header_len", "fwd_pkts_s", "bwd_pkts_s",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt",
    "urg_flag_cnt", "cwe_flag_cnt", "ece_flag_cnt", "down_up_ratio",
    "avg_pkt_size", "init_win_bytes_fwd", "init_win_bytes_bwd",
    "act_data_pkt_fwd", "min_seg_size_fwd",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
]

SYNTH_BASE = {
    "flow_duration": 200000.0, "tot_fwd_pkts": 12.0, "tot_bwd_pkts": 15.0,
    "totlen_fwd_pkts": 2200.0, "totlen_bwd_pkts": 2600.0,
    "fwd_pkt_len_max": 800.0, "fwd_pkt_len_min": 20.0, "fwd_pkt_len_mean": 150.0, "fwd_pkt_len_std": 90.0,
    "bwd_pkt_len_max": 900.0, "bwd_pkt_len_min": 24.0, "bwd_pkt_len_mean": 180.0, "bwd_pkt_len_std": 110.0,
    "flow_byts_s": 11000.0, "flow_pkts_s": 95.0,
    "flow_iat_mean": 2200.0, "flow_iat_std": 3200.0, "flow_iat_max": 42000.0, "flow_iat_min": 0.0,
    "fwd_iat_mean": 1800.0, "fwd_iat_std": 2600.0, "fwd_iat_max": 30000.0, "fwd_iat_min": 0.0,
    "bwd_iat_mean": 2000.0, "bwd_iat_std": 2800.0, "bwd_iat_max": 34000.0, "bwd_iat_min": 0.0,
    "fwd_header_len": 80.0, "bwd_header_len": 96.0, "fwd_pkts_s": 60.0, "bwd_pkts_s": 45.0,
    "pkt_len_min": 20.0, "pkt_len_max": 900.0, "pkt_len_mean": 165.0, "pkt_len_std": 100.0, "pkt_len_var": 10000.0,
    "fin_flag_cnt": 1.0, "syn_flag_cnt": 1.0, "rst_flag_cnt": 0.0, "psh_flag_cnt": 2.0, "ack_flag_cnt": 8.0,
    "urg_flag_cnt": 0.0, "cwe_flag_cnt": 0.0, "ece_flag_cnt": 0.0, "down_up_ratio": 1.0,
    "avg_pkt_size": 165.0, "init_win_bytes_fwd": 60000.0, "init_win_bytes_bwd": 60000.0,
    "act_data_pkt_fwd": 8.0, "min_seg_size_fwd": 20.0,
    "active_mean": 5200.0, "active_std": 9000.0, "active_max": 120000.0, "active_min": 0.0,
    "idle_mean": 11000.0, "idle_std": 16000.0, "idle_max": 240000.0, "idle_min": 0.0,
}

SYNTH_OFFSETS = {
    "Tor": {
        "flow_duration": 150000.0, "flow_iat_mean": 3000.0, "flow_iat_std": 3000.0,
        "idle_mean": 9000.0, "idle_std": 12000.0, "tot_fwd_pkts": -4.0, "tot_bwd_pkts": -5.0,
        "down_up_ratio": 0.35, "totlen_fwd_pkts": -600.0, "totlen_bwd_pkts": -700.0,
        "fwd_pkt_len_mean": -40.0, "bwd_pkt_len_mean": -45.0, "avg_pkt_size": -40.0,
        "syn_flag_cnt": -0.2, "ack_flag_cnt": -3.0, "flow_pkts_s": -25.0, "flow_byts_s": -4000.0,
    },
    "VPN": {
        "flow_duration": 320000.0, "flow_byts_s": 9000.0, "flow_pkts_s": 130.0,
        "totlen_fwd_pkts": 1800.0, "totlen_bwd_pkts": 1800.0,
        "fwd_pkt_len_mean": 70.0, "bwd_pkt_len_mean": 70.0, "avg_pkt_size": 70.0,
        "tot_fwd_pkts": 10.0, "tot_bwd_pkts": 12.0, "psh_flag_cnt": 3.0, "ack_flag_cnt": 10.0,
    },
    "NonTor": {
        "flow_duration": -50000.0, "flow_iat_mean": -1500.0, "idle_mean": -5000.0,
        "flow_byts_s": 2500.0, "flow_pkts_s": 20.0, "down_up_ratio": 0.5,
    },
    "NonVPN": {
        "flow_duration": 40000.0, "tot_fwd_pkts": 4.0, "tot_bwd_pkts": 3.0,
        "ack_flag_cnt": 6.0, "flow_iat_mean": -800.0, "flow_byts_s": -1500.0,
    },
}

SYNTH_TYPES = {
    "Tor": ["Tor-Browsing", "Tor-Chat", "Tor-Email", "Tor-Audio-Streaming",
            "Tor-Video-Streaming", "Tor-File-Transfer", "Tor-VoIP"],
    "VPN": ["VPN-Audio-Streaming", "VPN-Browsing", "VPN-Chat", "VPN-Email",
            "VPN-File-Transfer", "VPN-Video-Streaming", "VPN-VoIP"],
    "NonTor": ["Browsing", "Chat", "Email", "Audio-Streaming",
               "Video-Streaming", "File-Transfer", "VoIP", "P2P"],
    "NonVPN": ["Browsing", "Chat", "Email", "Audio-Streaming",
               "Video-Streaming", "File-Transfer", "VoIP", "P2P"],
}


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def detect_column(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    return None


def load_dataset(path=None):
    path = Path(path) if path else config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    return df


def load_or_synthesize(path=None):
    try:
        df = load_dataset(path)
        return df, "real"
    except FileNotFoundError:
        df = generate_synthetic(n=8000)
        return df, "synthetic"


def _normalize_label(label):
    s = str(label).lower().replace("-", "").replace("_", "").replace(" ", "")
    if s in ("tor", "torvpn", "vpntor"):
        return "tor"
    if s in ("vpn",):
        return "vpn"
    if s in ("nontor", "tor2"):
        return "nontor"
    if s in ("nonvpn",):
        return "nonvpn"
    return "other"


def _normalize_type(t):
    s = str(t).lower()
    if "audio" in s:
        return "Audio"
    if "brows" in s:
        return "Browsing"
    if "chat" in s:
        return "Chat"
    if "email" in s or "mail" in s:
        return "Email"
    if "file" in s or "ft" == s:
        return "File_Transfer"
    if "p2p" in s:
        return "P2P"
    if "video" in s:
        return "Video"
    if "voip" in s or "voice" in s:
        return "VoIP"
    return "Other"


def make_target(df, target):
    label_col = detect_column(df, config.LABEL_ALIASES)
    if label_col is None:
        raise ValueError("No label column found in dataset")

    if target == "label4":
        y = df[label_col].map(_normalize_label)
        y = y.replace("other", np.nan)
        classes = ["tor", "vpn", "nontor", "nonvpn"]
        y = y.astype("category").cat.set_categories(classes)
        return y.astype(str).values, classes
    if target == "tor_binary":
        y = df[label_col].map(_normalize_label)
        y = (y == "tor").astype(int)
        return y.values, ["non-tor", "tor"]
    if target == "darknet_binary":
        y = df[label_col].map(_normalize_label)
        y = y.isin(["tor", "vpn"]).astype(int)
        return y.values, ["benign", "darknet"]
    if target == "apptype":
        type_col = detect_column(df, config.TYPE_ALIASES)
        if type_col is None:
            raise ValueError("apptype target requires a Type column in the dataset")
        y = df[type_col].map(_normalize_type)
        y = y.replace("Other", np.nan)
        y = y.astype("category").cat.set_categories(APP_TYPE_ORDER)
        return y.astype(str).values, list(APP_TYPE_ORDER)
    raise ValueError(f"Unknown target: {target}")


def generate_synthetic(n=8000, seed=config.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    labels = ["NonVPN", "Tor", "NonTor", "VPN"]
    per_class = max(1, n // len(labels))
    rows = []

    for label in labels:
        offset = SYNTH_OFFSETS[label]
        types = SYNTH_TYPES[label]
        for _ in range(per_class):
            row = {}
            for feat, base in SYNTH_BASE.items():
                val = base + offset.get(feat, 0.0)
                if feat.endswith("_cnt") or feat in ("min_seg_size_fwd",):
                    noise = rng.poisson(max(0.1, abs(val) * 0.4))
                    row[feat] = max(0, int(val + noise))
                else:
                    noise = rng.normal(0, max(0.05, abs(val) * 0.18))
                    row[feat] = max(0.0, round(val + noise, 6))
            row["label"] = label
            row["type"] = rng.choice(types)
            rows.append(row)

    df = pd.DataFrame(rows)
    df["const_feature"] = 0
    df["empty_feature"] = np.nan
    df["inf_feature"] = np.inf
    return df


def save_synthetic(out_path, n=8000):
    out_path = Path(out_path)
    df = generate_synthetic(n=n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
