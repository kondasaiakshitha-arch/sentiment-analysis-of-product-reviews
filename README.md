# Sentiment Analysis of Product Reviews (NLP + Logistic Regression)

Classifies Amazon product reviews as **positive / neutral / negative** using TF-IDF (unigrams + bigrams) and a tuned Logistic Regression, with coefficient-based and LIME explainability and a Streamlit demo app.

## Quick start

```bash
pip install -r requirements.txt
python -m src.data_loader          # download 200K reviews, clean, sample 50K (cached in data/)
python -m src.train                # tune, compare, evaluate, save model + vectorizer
python -m src.binary_benchmark     # optional: positive-vs-negative only
python -m src.predict "Battery life is terrible and it stopped working after a week"
python -m src.explain "great quality but the battery is not good"   # LIME
streamlit run app/streamlit_app.py
```

The notebook [notebooks/sentiment_analysis.ipynb](notebooks/sentiment_analysis.ipynb) walks through the full workflow (already executed, with outputs).

## Project layout

```
src/config.py            paths, constants
src/data_loader.py       download, dedupe, star->label mapping, 50K sample
src/preprocessing.py     cleaning, negation-safe stop words, lemmatization
src/train.py             GridSearchCV, model comparison, evaluation, plots, saving
src/predict.py           SentimentPredictor (raw text -> label + confidence)
src/explain.py           LIME explanations
src/binary_benchmark.py  pos-vs-neg benchmark
app/streamlit_app.py     web app
models/                  tfidf_vectorizer.joblib, logreg_model.joblib
reports/                 metrics CSVs, misclassified reviews, figures/
notebooks/               executed analysis notebook
```

## Data

[SetFit/amazon_reviews_multi_en](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en) (English Amazon reviews, 1-5 stars). After dropping nulls and 574 duplicates, 10,000 reviews per star rating were sampled (50,000 total). Mapping: 1-2 stars = negative, 3 = neutral, 4-5 = positive, giving 40% / 20% / 40%. Split: 80% train / 20% test, stratified, seed 42.

## Method

1. **Cleaning/EDA** - nulls, duplicates, class balance, review length, word clouds.
2. **Preprocessing** - lowercase, strip URLs/HTML/punctuation, expand contractions (`didn't` -> `did not`), stop-word removal **excluding negations** (`not`, `no`, `never`, `without`...) and modifiers (`but`, `very`, `too`), WordNet lemmatization.
3. **Features** - TF-IDF, 1-2 grams, `min_df=3`, 100K features, sublinear TF.
4. **Model** - Logistic Regression, `GridSearchCV` (3-fold, macro-F1) over `C` in {0.5, 1, 2, 5, 10} and `class_weight` in {None, balanced}. The vectorizer is inside the CV pipeline so there is no leakage. Best: `C=0.5`, `class_weight=balanced` (CV macro-F1 0.653).
5. **Comparison** - Naive Bayes, Linear SVM, Random Forest on the same features.

## Results (10,000-review held-out test set, 3 classes)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| **Logistic Regression** | 0.693 | **0.659** | **0.658** | **0.656** |
| Linear SVM | **0.705** | 0.645 | 0.643 | 0.643 |
| Random Forest | 0.702 | 0.629 | 0.602 | 0.572 |
| Naive Bayes | 0.702 | 0.642 | 0.597 | 0.560 |

Logistic Regression has the best macro-F1 and is by far the best on the minority neutral class, because class weighting trades a little raw accuracy for balanced per-class performance. Naive Bayes and Random Forest reach similar accuracy mostly by under-predicting neutral.

Per class (Logistic Regression): negative F1 0.75, neutral F1 0.43, positive F1 0.79.

**Positive vs negative only** (neutral dropped, same pipeline, 8,000 test reviews): accuracy **0.887**, F1 0.887.

### What drives predictions (top LR coefficients)
- **Negative:** not, return, disappoint, waste, break, poor, very disappoint, terrible, not work, horrible, never
- **Neutral:** but, ok, however, not very, okay, though, three star, decent, unfortunately
- **Positive:** love, great, perfect, easy, good, excellent, nice, best, amaze, awesome

Full list: `reports/top_words.csv`; plot: `reports/figures/top_words.png`.

## Limitations

- **3-class accuracy is ~69%, not 90%.** Neutral reviews (3 stars) are inherently ambiguous; many read like mild positives or negatives. A ~90% accuracy figure is only realistic for the binary task (0.887 here). Quote the number that matches the task.
- Labels come from star ratings, which are noisy (mis-clicked stars, praise with a low rating).
- Bag-of-n-grams cannot model sarcasm, long-range negation, or mixed reviews ("great design but broke in a week").
- Trained on Amazon product reviews; performance will drop on other domains (e.g. movies, restaurants).
- English only. The Streamlit app does not detect language.
- Classes use equal-per-star sampling, so real-world class priors differ from training.

## Possible improvements
Fine-tune a transformer (e.g. DistilBERT) for the neutral class, add ordinal regression on stars, use `NOT_` negation scoping, or calibrate probabilities.
