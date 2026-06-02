"""
Full pipeline audit + before/after recommendation comparison.
Run: python audit_pipeline.py
"""

import os
import pickle

import pandas as pd

from recommendation_model import MovieRecommender, explain_recommendation, format_similarity_score

TEST_TITLES = [
    "The Hangover",
    "The Hangover Part II",
    "Interstellar",
    "Batman Begins",
    "Toy Story",
    "Titanic",
]


def audit_dataset_files():
    print("\n" + "=" * 70)
    print("1. DATASET PIPELINE AUDIT")
    print("=" * 70)
    if os.path.exists("data/tmdb_5000_movies.csv"):
        print(f"Legacy movies CSV: {len(pd.read_csv('data/tmdb_5000_movies.csv'))} rows")
    if os.path.exists("data/tmdb_movies.csv"):
        print(f"TMDB API CSV:      {len(pd.read_csv('data/tmdb_movies.csv'))} rows")
    print(
        "\nWhy ~1433 movies with TMDB-only mode:\n"
        "  - fetch_tmdb_data.py requests TMDB_MAX_MOVIES (default 1500) popular titles\n"
        "  - Rows without overview/title are skipped during API fetch\n"
        "  - drop_duplicates(title) and dropna remove more rows\n"
        "  - train.py dropna on genres/keywords/cast/crew removes additional rows\n"
        "\nUse DATA_SOURCE=combined in .env to merge legacy ~4800 + TMDB API catalogs."
    )


def audit_model_files():
    print("\n" + "=" * 70)
    print("2. MODEL / SIMILARITY MATRIX AUDIT")
    print("=" * 70)
    movies = pd.DataFrame(pickle.load(open("model/movies_dict.pkl", "rb")))
    sim = pickle.load(open("model/similarity.pkl", "rb"))
    meta = pickle.load(open("model/movie_metadata.pkl", "rb"))
    print(f"movies_dict.pkl rows:     {len(movies)}")
    print(f"similarity.pkl shape:     {sim.shape}")
    print(f"movie_metadata.pkl rows:  {len(meta)}")
    ok = len(movies) == sim.shape[0] == sim.shape[1] == len(meta)
    print(f"Dimension check:          {'PASS' if ok else 'FAIL'}")
    print(f"Similarity range:         min={sim.min():.4f}, max={sim.max():.4f}")
    if os.path.exists("model/vectorizer.pkl"):
        print("vectorizer.pkl:           present (TF-IDF)")


def print_recommendations(recommender: MovieRecommender, label: str):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    for title in TEST_TITLES:
        if title not in recommender._title_to_pos:
            print(f"\n[MISSING] {title}")
            continue
        src = movies.iloc[recommender._title_to_pos[title]]
        print(f"\n--- {title} ---")
        print(f"Genres: {src.get('genres_display', 'N/A')}")
        recs = recommender.recommend(title, top_n=10)
        for i, rec in enumerate(recs, 1):
            why = explain_recommendation(title, rec["title"], recommender)
            print(
                f"  {i:2}. {format_similarity_score(rec['similarity_score'])} | "
                f"{rec['title']} | {rec['genres_display']}\n"
                f"      Why: {why}"
            )


if __name__ == "__main__":
    audit_dataset_files()
    audit_model_files()
    recommender = MovieRecommender.load()
    movies = recommender.movies
    print_recommendations(recommender, "3. TOP-10 RECOMMENDATIONS (HYBRID TF-IDF + METADATA)")
