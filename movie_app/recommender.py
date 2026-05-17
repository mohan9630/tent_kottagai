from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _rating_bucket(rating: float) -> str:
    if rating >= 8:
        return "excellent"
    if rating >= 7:
        return "strong"
    if rating >= 5:
        return "average"
    return "rough"


def _prepare_feature_frame(movies_df: pd.DataFrame) -> pd.DataFrame:
    working_df = movies_df.copy()
    working_df["genres_text"] = working_df["genres"].apply(lambda items: " ".join(items))
    working_df["rating_bucket"] = working_df["rating"].apply(_rating_bucket)
    working_df["content_features"] = (
        working_df["genres_text"].fillna("")
        + " "
        + working_df["genres_text"].fillna("")
        + " "
        + working_df["rating_bucket"].fillna("")
        + " "
        + working_df["overview"].fillna("")
    ).str.strip()
    return working_df


def get_recommendations(
    movies_df: pd.DataFrame,
    movie_id: int,
    top_n: int = 8,
) -> pd.DataFrame:
    if movies_df.empty or movie_id not in movies_df["id"].values or len(movies_df) < 2:
        return pd.DataFrame(columns=list(movies_df.columns) + ["similarity"])

    working_df = _prepare_feature_frame(movies_df)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(working_df["content_features"])

    movie_index = working_df.index[working_df["id"] == movie_id][0]
    similarity_scores = cosine_similarity(tfidf_matrix[movie_index], tfidf_matrix).flatten()

    working_df["similarity"] = similarity_scores
    recommendations = working_df[working_df["id"] != movie_id].sort_values(
        by=["similarity", "rating", "popularity"],
        ascending=[False, False, False],
    )
    return recommendations.head(top_n)


def get_watchlist_recommendations(
    movies_df: pd.DataFrame,
    watchlist_ids: list[int],
    top_n: int = 8,
) -> pd.DataFrame:
    valid_ids = [movie_id for movie_id in watchlist_ids if movie_id in movies_df["id"].values]
    if movies_df.empty or not valid_ids or len(movies_df) <= len(valid_ids):
        return pd.DataFrame(columns=list(movies_df.columns) + ["similarity"])

    working_df = _prepare_feature_frame(movies_df)
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(working_df["content_features"])

    watchlist_indices = working_df.index[working_df["id"].isin(valid_ids)].tolist()
    average_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix[watchlist_indices]).mean(axis=1)

    working_df["similarity"] = average_similarity
    recommendations = working_df[~working_df["id"].isin(valid_ids)].sort_values(
        by=["similarity", "rating", "popularity"],
        ascending=[False, False, False],
    )
    return recommendations.head(top_n)
