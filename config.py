"""Central paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RAW_URL = ("https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en/"
           "resolve/main/train.jsonl")
RAW_PATH = DATA_DIR / "raw_train.jsonl"
CLEAN_PATH = DATA_DIR / "reviews_clean.csv"

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_PATH = MODEL_DIR / "logreg_model.joblib"

LABELS = ["negative", "neutral", "positive"]
SEED = 42
REVIEWS_PER_STAR = 10_000  # 5 stars x 10K = 50K reviews

for _d in (DATA_DIR, MODEL_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)
