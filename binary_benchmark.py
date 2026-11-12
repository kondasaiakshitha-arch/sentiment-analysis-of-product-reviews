"""Context benchmark: same pipeline on positive-vs-negative only (3-star dropped).

Run:  python -m src.binary_benchmark
"""
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .config import REPORT_DIR
from .train import load_split, make_vectorizer


def main():
    train_df, test_df = load_split()
    tr = train_df[train_df["sentiment"] != "neutral"]
    te = test_df[test_df["sentiment"] != "neutral"]
    vec = make_vectorizer()
    X_tr = vec.fit_transform(tr["clean_text"])
    clf = LogisticRegression(C=2, max_iter=1000).fit(X_tr, tr["sentiment"])
    pred = clf.predict(vec.transform(te["clean_text"]))
    res = {"n_test": len(te),
           "accuracy": round(accuracy_score(te["sentiment"], pred), 4),
           "f1_macro": round(f1_score(te["sentiment"], pred, average="macro"), 4)}
    print(res)
    (REPORT_DIR / "binary_benchmark.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
