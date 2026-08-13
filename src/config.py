from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATASET_FILE = DATA_DIR / "Darknet.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_MISSING_RATIO = 0.8
CORR_THRESHOLD = 0.98
TOP_K_FEATURES = 60
MIN_TRAIN_ROWS = 200

TARGETS = ["tor_binary", "darknet_binary", "label4", "apptype"]
MODELS = ["rf", "gb", "xgb", "lr"]
DEFAULT_TARGET = "tor_binary"

NON_FEATURE_COLUMNS = [
    "flow id", "source ip", "source port", "destination ip", "destination port",
    "protocol", "timestamp", "srcip", "dstip", "srcport", "dstport", "flow id.",
]
LABEL_ALIASES = ["label", "class", "category", "label2"]
TYPE_ALIASES = ["type", "label2", "subclass"]

TARGET_DESCRIPTIONS = {
    "tor_binary": "Binary: Tor traffic vs everything else",
    "darknet_binary": "Binary: darknet (Tor + VPN) vs benign (NonTor + NonVPN)",
    "label4": "Multiclass: Tor / VPN / NonTor / NonVPN",
    "apptype": "Multiclass: application type (Browsing, Chat, Email, File Transfer, P2P, Streaming, VoIP)",
}

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
