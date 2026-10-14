"""Build the text-to-prediction pipeline."""

from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def build_pipeline(classifier=None):
    """Return a pipeline: TF-IDF vectorizer followed by a classifier.

    The pipeline takes raw message text and returns 'ham' or 'spam'.
    Uses Naive Bayes unless another classifier is passed in.
    """
    if classifier is None:
        classifier = MultinomialNB()

    return Pipeline(
        [
            ("tfidf", TfidfVectorizer()),  # text -> numbers
            # clone() makes a fresh, untrained copy, so two pipelines built
            # from the same classifier never share (and overwrite) its training.
            ("classifier", clone(classifier)),  # numbers -> ham or spam
        ]
    )
