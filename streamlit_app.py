"""Streamlit demo.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.explain import explain_review  # noqa: E402
from src.predict import SentimentPredictor  # noqa: E402

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬")


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


EXAMPLES = {
    "Negative": "Battery life is terrible and it stopped working after a week",
    "Neutral": "It's okay. Does the job but nothing special for the price.",
    "Positive": "Love it! Great quality, fast shipping and works perfectly.",
    "Negation": "The sound is not good at all.",
}
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}

st.title("💬 Product Review Sentiment Analyzer")
st.caption("TF-IDF (unigrams + bigrams) + Logistic Regression, trained on 50K Amazon reviews.")

try:
    predictor = get_predictor()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

if "review" not in st.session_state:
    st.session_state.review = EXAMPLES["Negative"]

cols = st.columns(len(EXAMPLES))
for col, (name, txt) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.review = txt

review = st.text_area("Paste a product review", key="review", height=140)
use_lime = st.checkbox("Explain with LIME (slower)")

if st.button("Analyze", type="primary"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        res = predictor.predict(review)
        st.subheader(f"{EMOJI[res['label']]} {res['label'].title()}  "
                     f"({res['confidence']:.1%} confidence)")
        st.bar_chart(pd.Series(res["probabilities"], name="probability"))

        label, pos, neg = predictor.contributions(review)
        c1, c2 = st.columns(2)
        c1.markdown(f"**Words supporting _{label}_**")
        c1.write(", ".join(f"`{w}`" for w, s in pos if s > 0) or "-")
        c2.markdown(f"**Words working against it**")
        c2.write(", ".join(f"`{w}`" for w, s in neg if s < 0) or "-")

        if use_lime:
            with st.spinner("Running LIME..."):
                lbl, weights = explain_review(predictor, review)
            st.markdown(f"**LIME word weights for _{lbl}_**")
            st.bar_chart(pd.DataFrame(weights, columns=["word", "weight"])
                         .set_index("word"))

st.divider()
st.caption("Limitations: trained on Amazon product reviews; neutral (3-star) "
           "reviews are the hardest class; sarcasm is not handled.")
