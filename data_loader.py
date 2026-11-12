"""Download the Amazon reviews, clean, map stars -> sentiment, sample 50K."""
import urllib.request

import pandas as pd

from .config import (CLEAN_PATH, RAW_PATH, RAW_URL, REVIEWS_PER_STAR, SEED)
from .preprocessing import clean_text


def star_to_sentiment(stars: int) -> str:
    """1-2 stars negative, 3 neutral, 4-5 positive."""
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


def download_raw():
    if not RAW_PATH.exists():
        print(f"Downloading dataset to {RAW_PATH} ...")
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    return RAW_PATH


def build_dataset(force: bool = False) -> pd.DataFrame:
    """Create (or load cached) cleaned dataset with columns:
    text, stars, sentiment, clean_text."""
    if CLEAN_PATH.exists() and not force:
        return pd.read_csv(CLEAN_PATH, keep_default_na=False)

    df = pd.read_json(download_raw(), lines=True)
    df = df.rename(columns={"label": "stars"})[["text", "stars"]]
    df["stars"] = df["stars"] + 1  # source labels are 0-4
    print(f"raw rows: {len(df):,}")

    df = df.dropna(subset=["text"])
    df["text"] = df["text"].str.strip()
    df = df[df["text"].str.len() > 0].drop_duplicates(subset="text")
    print(f"after dropping nulls/duplicates: {len(df):,}")

    # equal number of reviews per star rating -> 50K total
    df = pd.concat([g.sample(min(len(g), REVIEWS_PER_STAR), random_state=SEED)
                    for _, g in df.groupby("stars")]).reset_index(drop=True)
    df["sentiment"] = df["stars"].map(star_to_sentiment)
    df["clean_text"] = df["text"].map(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"saved {len(df):,} reviews -> {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    build_dataset(force=True)
