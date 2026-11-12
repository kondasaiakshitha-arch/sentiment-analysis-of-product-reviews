"""Text cleaning: lowercase, strip URLs/HTML/punctuation, stop words, lemmatize.

Negations ("not", "no", "never", "n't" ...) are deliberately KEPT, because
"not good" must not collapse into "good". Contractions are expanded first so
"didn't work" becomes "did not work".
"""
import re
from functools import lru_cache

import nltk


def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"),
                      ("wordnet", "corpora/wordnet"),
                      ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

NEGATIONS = {
    "not", "no", "nor", "never", "none", "nothing", "neither", "cannot",
    "against", "without",
}
# Words that flip/modify sentiment and should survive stop-word removal.
KEEP = NEGATIONS | {"but", "very", "too", "only", "few", "more", "most", "less"}
STOP_WORDS = set(stopwords.words("english")) - KEEP

CONTRACTIONS = [
    (r"\bwon't\b", "will not"), (r"\bcan't\b", "can not"),
    (r"\bain't\b", "is not"), (r"n't\b", " not"), (r"'re\b", " are"),
    (r"'ve\b", " have"), (r"'ll\b", " will"), (r"'d\b", " would"),
    (r"'m\b", " am"), (r"'s\b", ""),
]
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
HTML_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z]+")

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=200_000)
def _lemma(token: str) -> str:
    # try verb first ("stopped" -> "stop"), then noun ("batteries" -> "battery")
    v = _lemmatizer.lemmatize(token, pos="v")
    return v if v != token else _lemmatizer.lemmatize(token, pos="n")


def clean_text(text) -> str:
    """Raw review -> space-joined cleaned lemmas."""
    if not isinstance(text, str):
        return ""
    t = text.lower().replace("’", "'")
    t = URL_RE.sub(" ", t)
    t = HTML_RE.sub(" ", t)
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    tokens = TOKEN_RE.findall(t)
    tokens = [_lemma(w) for w in tokens if len(w) > 1 and w not in STOP_WORDS]
    return " ".join(tokens)
