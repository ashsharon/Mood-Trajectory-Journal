"""
features.py
------------
Extracts linguistic features from a single journal entry that prior NLP research
has associated with mood and wellbeing (NOT diagnostic markers - just measurable
signals that shift over time):

- First-person singular pronoun ratio (I, me, my) - elevated self-focus is a
  well-documented correlate in linguistic psychology research (Pennebaker et al.)
- Negation word ratio (not, never, can't, don't, ...)
- Absolutist word ratio (always, never, everything, nothing, completely) - linked
  to all-or-nothing thinking patterns in some studies
- Social word ratio (friend, we, us, talk, family) - social engagement/withdrawal proxy
- Lexical diversity (type-token ratio) - reduced complexity can correlate with
  reduced cognitive engagement
- Sentiment polarity - via simple lexicon-based scoring (VADER)
- Entry length - very short entries over time can indicate withdrawal

IMPORTANT: none of these features are diagnostic on their own or in combination.
They are aggregate linguistic patterns studied in academic literature, used here
ONLY to visualize trends over time, never to label a person.
"""

import re
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER lexicon is available (downloads once, cached locally after)
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
NEGATION_WORDS = {"not", "no", "never", "cant", "can't", "dont", "don't", "wont", "won't",
                   "nothing", "nobody", "none", "neither", "nor"}
ABSOLUTIST_WORDS = {"always", "never", "everything", "nothing", "everyone", "no one",
                     "completely", "totally", "entirely", "constantly", "all", "none"}
SOCIAL_WORDS = {"friend", "friends", "we", "us", "our", "talk", "talked", "family",
                 "together", "party", "hangout", "call", "called", "visit"}

_sia = SentimentIntensityAnalyzer()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def extract_features(text: str) -> dict:
    tokens = _tokenize(text)
    n_tokens = len(tokens) if tokens else 1  # avoid div by zero

    first_person_count = sum(1 for t in tokens if t in FIRST_PERSON_SINGULAR)
    negation_count = sum(1 for t in tokens if t in NEGATION_WORDS)
    absolutist_count = sum(1 for t in tokens if t in ABSOLUTIST_WORDS)
    social_count = sum(1 for t in tokens if t in SOCIAL_WORDS)

    unique_tokens = len(set(tokens))
    lexical_diversity = unique_tokens / n_tokens

    sentiment = _sia.polarity_scores(text)

    return {
        "word_count": len(tokens),
        "first_person_singular_ratio": round(first_person_count / n_tokens, 4),
        "negation_ratio": round(negation_count / n_tokens, 4),
        "absolutist_ratio": round(absolutist_count / n_tokens, 4),
        "social_word_ratio": round(social_count / n_tokens, 4),
        "lexical_diversity": round(lexical_diversity, 4),
        "sentiment_compound": round(sentiment["compound"], 4),  # -1 (negative) to +1 (positive)
        "sentiment_negative": round(sentiment["neg"], 4),
    }


if __name__ == "__main__":
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "data", "sample_journal_entries.json")) as f:
        entries = json.load(f)

    for entry in entries[:3]:
        feats = extract_features(entry["text"])
        print(entry["date"], json.dumps(feats))
