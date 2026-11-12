"""Train, tune, compare and evaluate sentiment models.

Run:  python -m src.train
"""
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report,  # noqa: E402
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, train_test_split  # noqa: E402
from sklearn.naive_bayes import MultinomialNB  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from .config import (FIG_DIR, LABELS, MODEL_PATH, REPORT_DIR, SEED,  # noqa: E402
                     VECTORIZER_PATH)
from .data_loader import build_dataset  # noqa: E402


def make_vectorizer() -> TfidfVectorizer:
    # unigrams + bigrams so "not working" / "not good" are captured
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                           sublinear_tf=True)


def load_split(test_size: float = 0.2):
    df = build_dataset()
    return train_test_split(df, test_size=test_size, random_state=SEED,
                            stratify=df["sentiment"])


def tune_logreg(train_df: pd.DataFrame, cv: int = 3) -> GridSearchCV:
    """GridSearchCV over C and class_weight. The vectorizer sits inside the
    pipeline so every CV fold fits TF-IDF on its own training part (no leakage)."""
    pipe = Pipeline([("tfidf", make_vectorizer()),
                     ("clf", LogisticRegression(max_iter=1000))])
    grid = {"clf__C": [0.5, 1, 2, 5, 10],
            "clf__class_weight": [None, "balanced"]}
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1,
                      verbose=1)
    gs.fit(train_df["clean_text"], train_df["sentiment"])
    return gs


def compare_models(X_train, y_train, X_test, y_test, best_C, best_cw):
    models = {
        "Logistic Regression": LogisticRegression(C=best_C, class_weight=best_cw,
                                                  max_iter=1000),
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM": LinearSVC(C=0.5, class_weight=best_cw),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                                random_state=SEED,
                                                class_weight=best_cw),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        secs = time.time() - t0
        pred = m.predict(X_test)
        p, r, f, _ = precision_recall_fscore_support(y_test, pred,
                                                     average="macro",
                                                     zero_division=0)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision (macro)": p, "Recall (macro)": r,
                     "F1 (macro)": f, "Train time (s)": secs})
        fitted[name] = m
        print(f"{name:20s} acc={rows[-1]['Accuracy']:.4f} f1={f:.4f} ({secs:.0f}s)")
    table = (pd.DataFrame(rows).sort_values("F1 (macro)", ascending=False)
             .reset_index(drop=True))
    return table, fitted


def plot_confusion(y_true, y_pred, path=FIG_DIR / "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", "Counts"),
        (axes[1], cm / cm.sum(axis=1, keepdims=True), ".1%", "Row-normalised"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                    xticklabels=LABELS, yticklabels=LABELS, cbar=False)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix - {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def top_words(vectorizer, model, n: int = 20) -> pd.DataFrame:
    """Highest-weight features pushing towards each class (from LR coefficients)."""
    feats = np.array(vectorizer.get_feature_names_out())
    out = []
    for i, cls in enumerate(model.classes_):
        idx = np.argsort(model.coef_[i])[::-1][:n]
        out += [{"class": cls, "rank": r + 1, "word": feats[j],
                 "coef": model.coef_[i][j]} for r, j in enumerate(idx)]
    return pd.DataFrame(out)


def plot_top_words(tw: pd.DataFrame, path=FIG_DIR / "top_words.png"):
    colors = {"negative": "#d9534f", "neutral": "#f0ad4e", "positive": "#5cb85c"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for ax, cls in zip(axes, LABELS):
        d = tw[tw["class"] == cls].head(15).iloc[::-1]
        ax.barh(d["word"], d["coef"], color=colors[cls])
        ax.set_title(f"Top words -> {cls}")
        ax.set_xlabel("LR coefficient")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_analysis(test_df, y_pred, proba, classes) -> pd.DataFrame:
    d = test_df[["text", "stars", "sentiment"]].copy()
    d["predicted"] = y_pred
    d["confidence"] = proba.max(axis=1)
    return (d[d["sentiment"] != d["predicted"]]
            .sort_values("confidence", ascending=False))


def main():
    train_df, test_df = load_split()
    print(f"train={len(train_df):,} test={len(test_df):,}")

    print("\n== GridSearchCV (Logistic Regression) ==")
    gs = tune_logreg(train_df)
    print("best params:", gs.best_params_, f"cv macro-F1={gs.best_score_:.4f}")
    pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score",
                                  "rank_test_score"]].sort_values(
        "rank_test_score").to_csv(REPORT_DIR / "gridsearch_results.csv",
                                  index=False)

    best_C = gs.best_params_["clf__C"]
    best_cw = gs.best_params_["clf__class_weight"]
    vec = gs.best_estimator_.named_steps["tfidf"]
    X_train = vec.transform(train_df["clean_text"])
    X_test = vec.transform(test_df["clean_text"])
    y_train, y_test = train_df["sentiment"], test_df["sentiment"]

    print("\n== Model comparison ==")
    table, fitted = compare_models(X_train, y_train, X_test, y_test,
                                   best_C, best_cw)
    table.round(4).to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(table.round(4).to_string(index=False))

    lr = fitted["Logistic Regression"]
    pred = lr.predict(X_test)
    proba = lr.predict_proba(X_test)
    report = classification_report(y_test, pred, digits=3)
    print("\n== Logistic Regression test report ==\n" + report)
    (REPORT_DIR / "classification_report.txt").write_text(report)

    plot_confusion(y_test, pred)
    tw = top_words(vec, lr)
    tw.to_csv(REPORT_DIR / "top_words.csv", index=False)
    plot_top_words(tw)
    err = error_analysis(test_df, pred, proba, lr.classes_)
    err.to_csv(REPORT_DIR / "misclassified_reviews.csv", index=False)
    print(f"misclassified: {len(err):,} / {len(test_df):,}")

    joblib.dump(vec, VECTORIZER_PATH)
    joblib.dump(lr, MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}\nsaved vectorizer -> {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
