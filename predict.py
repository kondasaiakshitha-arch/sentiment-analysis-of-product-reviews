"""Load the saved model + vectorizer and predict sentiment for raw text.

CLI:  python -m src.predict "Battery life is terrible"
"""
import sys

import joblib
import numpy as np

from .config import MODEL_PATH, VECTORIZER_PATH
from .preprocessing import clean_text


class SentimentPredictor:
    def __init__(self):
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        self.model = joblib.load(MODEL_PATH)
        self.classes = list(self.model.classes_)
        self._features = np.array(self.vectorizer.get_feature_names_out())

    def predict_proba(self, texts) -> np.ndarray:
        """Raw texts -> (n, 3) class probabilities (also used by LIME)."""
        X = self.vectorizer.transform([clean_text(t) for t in texts])
        return self.model.predict_proba(X)

    def predict(self, text: str) -> dict:
        proba = self.predict_proba([text])[0]
        i = int(proba.argmax())
        return {"label": self.classes[i], "confidence": float(proba[i]),
                "probabilities": dict(zip(self.classes, map(float, proba)))}

    def contributions(self, text: str, n: int = 8):
        """Per-word contribution (tfidf * coef) towards the predicted class."""
        X = self.vectorizer.transform([clean_text(text)])
        i = int(self.model.predict_proba(X)[0].argmax())
        row = X.tocoo()
        contrib = [(self._features[j], float(v * self.model.coef_[i][j]))
                   for j, v in zip(row.col, row.data)]
        contrib.sort(key=lambda t: t[1])
        return self.classes[i], contrib[::-1][:n], contrib[:n]


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "Battery life is terrible and it stopped working after a week"
    r = SentimentPredictor().predict(text)
    print(f"{r['label']} ({r['confidence']:.1%})")
    for k, v in r["probabilities"].items():
        print(f"  {k:9s} {v:.1%}")
