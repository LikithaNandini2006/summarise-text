import re
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from rapidfuzz import fuzz
except ImportError:
    from fuzzywuzzy import fuzz


def _clean_text(text):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text


def _split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r'(?<=[.!?])\s+', text)
        if sentence.strip()
    ]


def summarize_text(text):
    summary, _accuracy = summarize_text_with_accuracy(text)
    return summary


def summarize_text_with_accuracy(text):
    text = _clean_text(text)

    if not text:
        return "", 0

    sentences = _split_sentences(text)

    if len(sentences) <= 2:
        return text, 100

    tfidf = TfidfVectorizer(stop_words='english')

    try:
        matrix = tfidf.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:2]), 60

    similarity = cosine_similarity(matrix)

    scores = []

    for i in range(len(sentences)):

        cosine_score = similarity[i].sum()

        fuzzy_score = 0

        for j in range(len(sentences)):
            if i != j:
                fuzzy_score += fuzz.ratio(
                    sentences[i],
                    sentences[j]
                )

        final_score = cosine_score + (fuzzy_score / 100)

        scores.append(final_score)

    summary_length = max(
        1,
        int(len(sentences) * 0.5)
    )

    top_sentences = np.argsort(scores)[-summary_length:]

    top_sentences = sorted(top_sentences)

    summary = " ".join(
        [sentences[i] for i in top_sentences]
    )

    max_score = max(scores) if scores else 0

    if max_score <= 0:
        accuracy = 60
    else:
        selected_average = float(np.mean([scores[i] for i in top_sentences]))
        accuracy = round((selected_average / max_score) * 100)
        accuracy = min(100, max(1, accuracy))

    return summary, accuracy
