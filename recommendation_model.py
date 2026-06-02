"""
Training and inference for the content-based recommender.
Uses TF-IDF vectors + genre/keyword/cast-aware hybrid re-ranking.
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import build_feature_record

load_dotenv()

MODEL_DIR = "model"
MOVIES_PKL = os.path.join(MODEL_DIR, "movies_dict.pkl")
SIMILARITY_PKL = os.path.join(MODEL_DIR, "similarity.pkl")
METADATA_PKL = os.path.join(MODEL_DIR, "movie_metadata.pkl")
VECTORIZER_PKL = os.path.join(MODEL_DIR, "vectorizer.pkl")

# Hybrid re-ranking weights (cosine base + metadata overlap bonuses)
GENRE_MATCH_BOOST = 0.40
KEYWORD_MATCH_BOOST = 0.25
CAST_MATCH_BOOST = 0.15
DIRECTOR_MATCH_BOOST = 0.12
NO_GENRE_PENALTY = 0.20  # multiply score when zero genre overlap


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def hybrid_score(
    cosine_sim: float,
    query_meta: dict,
    candidate_meta: dict,
) -> float:
    """Combine cosine similarity with metadata overlap signals."""
    genre_sim = _jaccard(query_meta["genre_set"], candidate_meta["genre_set"])
    keyword_sim = _jaccard(query_meta["keyword_set"], candidate_meta["keyword_set"])
    cast_sim = _jaccard(query_meta["cast_set"], candidate_meta["cast_set"])
    director_sim = _jaccard(query_meta["director_set"], candidate_meta["director_set"])

    score = float(cosine_sim)
    score *= 1.0 + GENRE_MATCH_BOOST * genre_sim
    score *= 1.0 + KEYWORD_MATCH_BOOST * keyword_sim
    score *= 1.0 + CAST_MATCH_BOOST * cast_sim
    score *= 1.0 + DIRECTOR_MATCH_BOOST * director_sim

    if query_meta["genre_set"] and candidate_meta["genre_set"] and genre_sim == 0:
        score *= NO_GENRE_PENALTY

    return score


def format_similarity_score(score: float) -> str:
    """Format hybrid score as percentage (clamped 0–99 for display)."""
    pct = max(0, min(99, round(float(score) * 100)))
    return f"Similarity: {pct}%"


def train_with_audit(movies_df: pd.DataFrame, audit: bool = True) -> dict:
    """
    Train model from preprocessed movie dataframe.
    Returns audit statistics dict.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    audit_log = {"steps": []}

    def log_step(name: str, count: int, detail: str = ""):
        audit_log["steps"].append({"step": name, "count": count, "detail": detail})
        if audit:
            msg = f"[{name}] rows={count}"
            if detail:
                msg += f" — {detail}"
            print(msg)

    df = movies_df.copy()
    log_step("input_movies", len(df))

    feature_rows = []
    for _, row in df.iterrows():
        feature_rows.append(build_feature_record(row))

    features_df = pd.DataFrame(feature_rows)
    df = pd.concat([df.reset_index(drop=True), features_df], axis=1)
    log_step("after_feature_engineering", len(df))

    tags = df["tags"].tolist()
    max_features = min(12000, max(2000, len(df) * 4))

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        sublinear_tf=True,
    )
    vectors = vectorizer.fit_transform(tags)
    log_step(
        "after_vectorization",
        vectors.shape[0],
        f"shape={vectors.shape}, max_features={max_features}, vocab={len(vectorizer.vocabulary_)}",
    )

    similarity = cosine_similarity(vectors)
    log_step(
        "similarity_matrix",
        similarity.shape[0],
        f"shape={similarity.shape}, min={similarity.min():.4f}, max={similarity.max():.4f}",
    )

    assert len(df) == similarity.shape[0] == similarity.shape[1], (
        f"Dimension mismatch: movies={len(df)}, similarity={similarity.shape}"
    )

    output_df = df[
        [
            "movie_id",
            "title",
            "tags",
            "release_year",
            "rating",
            "genres_display",
            "overview_display",
        ]
    ].copy()

    metadata = []
    for _, row in df.iterrows():
        metadata.append(
            {
                "genre_set": list(row["genre_set"]),
                "keyword_set": list(row["keyword_set"]),
                "cast_set": list(row["cast_set"]),
                "director_set": list(row["director_set"]),
            }
        )

    pickle.dump(output_df.to_dict(), open(MOVIES_PKL, "wb"))
    pickle.dump(similarity, open(SIMILARITY_PKL, "wb"))
    pickle.dump(metadata, open(METADATA_PKL, "wb"))
    pickle.dump(vectorizer, open(VECTORIZER_PKL, "wb"))

    audit_log["final_movie_count"] = len(output_df)
    audit_log["similarity_shape"] = similarity.shape
    return audit_log


class MovieRecommender:
    """Loaded recommender for inference."""

    def __init__(self, movies_df: pd.DataFrame, similarity: np.ndarray, metadata: list):
        self.movies = movies_df
        self.similarity = similarity
        self.metadata = metadata
        self._title_to_pos = {
            title: pos for pos, title in enumerate(movies_df["title"].values)
        }

    @classmethod
    def load(cls):
        movies = pd.DataFrame(pickle.load(open(MOVIES_PKL, "rb")))
        similarity = pickle.load(open(SIMILARITY_PKL, "rb"))
        metadata = pickle.load(open(METADATA_PKL, "rb"))
        if len(movies) != similarity.shape[0] or len(movies) != len(metadata):
            raise ValueError(
                f"Model files mismatch: movies={len(movies)}, "
                f"similarity={similarity.shape}, metadata={len(metadata)}"
            )
        return cls(movies, similarity, metadata)

    def recommend(self, title: str, top_n: int = 8, candidate_pool: int = 50) -> list[dict]:
        if title not in self._title_to_pos:
            raise KeyError(f"Movie not found: {title}")

        query_pos = self._title_to_pos[title]
        query_meta = {
            "genre_set": set(self.metadata[query_pos]["genre_set"]),
            "keyword_set": set(self.metadata[query_pos]["keyword_set"]),
            "cast_set": set(self.metadata[query_pos]["cast_set"]),
            "director_set": set(self.metadata[query_pos]["director_set"]),
        }

        cosine_row = self.similarity[query_pos]
        candidates = np.argsort(cosine_row)[::-1]

        ranked = []
        for pos in candidates:
            if pos == query_pos:
                continue
            cand_meta = {
                "genre_set": set(self.metadata[pos]["genre_set"]),
                "keyword_set": set(self.metadata[pos]["keyword_set"]),
                "cast_set": set(self.metadata[pos]["cast_set"]),
                "director_set": set(self.metadata[pos]["director_set"]),
            }
            hybrid = hybrid_score(float(cosine_row[pos]), query_meta, cand_meta)
            ranked.append((pos, hybrid, float(cosine_row[pos])))
            if len(ranked) >= candidate_pool:
                break

        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:top_n]

        if top:
            max_hybrid = top[0][1]
            if max_hybrid > 0:
                top = [(p, h / max_hybrid, c) for p, h, c in top]

        results = []
        for pos, norm_score, raw_cosine in top:
            row = self.movies.iloc[pos]
            results.append(
                {
                    "position": pos,
                    "title": row["title"],
                    "movie_id": (
                        int(float(row["movie_id"]))
                        if pd.notna(row["movie_id"]) and float(row["movie_id"]) > 0
                        else None
                    ),
                    "similarity_score": norm_score,
                    "cosine_score": raw_cosine,
                    "genres_display": row.get("genres_display", "N/A"),
                    "year": row.get("release_year"),
                    "rating": row.get("rating"),
                    "overview_display": row.get("overview_display"),
                }
            )
        return results


def explain_recommendation(source_title: str, recommended_title: str, recommender: MovieRecommender) -> str:
    """Human-readable explanation for a recommendation pair."""
    q = recommender._title_to_pos[source_title]
    c = recommender._title_to_pos[recommended_title]
    qm = recommender.metadata[q]
    cm = recommender.metadata[c]
    shared_genres = set(qm["genre_set"]) & set(cm["genre_set"])
    shared_kw = set(qm["keyword_set"]) & set(cm["keyword_set"])
    shared_cast = set(qm["cast_set"]) & set(cm["cast_set"])
    shared_dir = set(qm["director_set"]) & set(cm["director_set"])
    parts = []
    if shared_genres:
        parts.append(f"shared genres: {', '.join(sorted(shared_genres))}")
    if shared_kw:
        parts.append(f"shared keywords: {', '.join(sorted(list(shared_kw)[:5]))}")
    if shared_cast:
        parts.append(f"shared cast: {', '.join(sorted(shared_cast))}")
    if shared_dir:
        parts.append(f"shared director(s): {', '.join(sorted(shared_dir))}")
    if not parts:
        parts.append("textual similarity in overview/tags")
    return "; ".join(parts)
