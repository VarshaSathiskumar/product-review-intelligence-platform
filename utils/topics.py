"""
Recurring-complaint discovery with BERTopic on BGE-M3 embeddings.

BERTopic (UMAP + HDBSCAN) needs a reasonably large corpus, so fit it once on
many complaints (e.g. a whole product category), then aggregate the topic
assignments per product — do not fit one model per product.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

from utils.embeddings import embed_texts


# Filler words that otherwise dominate every review-topic label
_REVIEW_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    "just", "like", "really", "dont", "didnt", "doesnt", "product", "star", "stars",
    "got", "bought", "buy", "use", "used", "one", "would", "good", "great", "get",
    "even", "im", "ive", "thing", "things", "time", "want", "make", "work",
]


def fit_complaint_topics(
    texts: List[str],
    embeddings: Optional[np.ndarray] = None,
    min_topic_size: int = 10,
    reduce_outliers: bool = True,
    random_state: int = 42,
) -> Tuple[BERTopic, np.ndarray, np.ndarray]:
    """
    Fit BERTopic on complaint texts.

    Returns:
        (topic_model, topics, embeddings). With reduce_outliers, HDBSCAN noise
        (-1) is reassigned to the nearest topic.
    """
    if embeddings is None:
        embeddings = embed_texts(texts, batch_size=64, show_progress_bar=True)

    vectorizer = CountVectorizer(stop_words=_REVIEW_STOP_WORDS, ngram_range=(1, 2), min_df=2)
    model = BERTopic(
        umap_model=UMAP(
            n_neighbors=8, n_components=5, min_dist=0.0,
            metric="cosine", random_state=random_state,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=min_topic_size, min_samples=3, metric="euclidean",
            cluster_selection_method="leaf", prediction_data=True,
        ),
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(texts, embeddings)
    topics = np.array(topics)
    if reduce_outliers and (topics == -1).any():
        topics = np.array(
            model.reduce_outliers(texts, topics.tolist(), strategy="embeddings", embeddings=embeddings)
        )
        model.update_topics(texts, topics=topics.tolist(), vectorizer_model=vectorizer)
    return model, topics, embeddings


def topic_keywords(model: BERTopic, topic_id: int, top_n: int = 5) -> str:
    words = model.get_topic(topic_id) or []
    return ", ".join(w for w, _ in words[:top_n])


def summarize_product_topics(
    df: pd.DataFrame, model: BERTopic, top_n_topics: int = 3, top_keywords: int = 5
) -> pd.DataFrame:
    """
    Aggregate topic assignments per product.

    df needs columns: product_id, product_title, product_category, review, topic.
    Returns one row per (product, topic), keeping each product's top_n_topics.
    """
    rows = []
    for pid, grp in df.groupby("product_id"):
        counts = grp["topic"].value_counts().head(top_n_topics)
        meta = grp.iloc[0]
        for topic_id, n in counts.items():
            rows.append({
                "product_id": pid,
                "product_title": meta["product_title"],
                "product_category": meta["product_category"],
                "cluster_id": int(topic_id),
                "complaint_count": int(n),
                "theme_keywords": topic_keywords(model, int(topic_id), top_keywords),
                "sample_review": grp.loc[grp["topic"] == topic_id, "review"].iloc[0],
            })
    return pd.DataFrame(rows)


def summarize_category_topics(
    df: pd.DataFrame, model: BERTopic, top_n_topics: int = 5, top_keywords: int = 8
) -> pd.DataFrame:
    """
    Aggregate topic assignments per product category (recurring complaint themes).

    df needs columns: product_category, review, topic. Returns one row per
    (category, topic) with share of that category's complaints.
    """
    rows = []
    for cat, grp in df.groupby("product_category"):
        counts = grp["topic"].value_counts().head(top_n_topics)
        for topic_id, n in counts.items():
            rows.append({
                "product_category": cat,
                "topic_id": int(topic_id),
                "complaint_count": int(n),
                "share_of_category": round(n / len(grp), 3),
                "theme_keywords": topic_keywords(model, int(topic_id), top_keywords),
                "sample_review": grp.loc[grp["topic"] == topic_id, "review"].iloc[0],
            })
    return pd.DataFrame(rows)
