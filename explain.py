"""LIME explanation of individual predictions.

CLI:  python -m src.explain "not good at all, stopped working"
"""
import sys

from lime.lime_text import LimeTextExplainer

from .predict import SentimentPredictor


def explain_review(predictor: SentimentPredictor, text: str,
                   num_features: int = 8, num_samples: int = 1000):
    explainer = LimeTextExplainer(class_names=predictor.classes,
                                  random_state=42)
    pred_idx = predictor.classes.index(predictor.predict(text)["label"])
    exp = explainer.explain_instance(text, predictor.predict_proba,
                                     labels=[pred_idx],
                                     num_features=num_features,
                                     num_samples=num_samples)
    return predictor.classes[pred_idx], exp.as_list(label=pred_idx)


if __name__ == "__main__":
    txt = " ".join(sys.argv[1:]) or "Battery life is terrible and it stopped working after a week"
    label, weights = explain_review(SentimentPredictor(), txt)
    print("predicted:", label)
    for w, s in weights:
        print(f"  {w:15s} {s:+.3f}")
