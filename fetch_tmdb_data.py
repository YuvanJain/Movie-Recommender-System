"""
Download and cache a movie dataset from the live TMDB API.

Usage:
    python fetch_tmdb_data.py
    python fetch_tmdb_data.py --max-movies 2000
    python fetch_tmdb_data.py --force   # re-fetch even if CSV exists
"""

import argparse
import json
import os

import pandas as pd
from dotenv import load_dotenv

from tmdb_client import TMDBClient, movie_record_from_api

load_dotenv()

OUTPUT_CSV = os.path.join("data", "tmdb_movies.csv")
CACHE_DIR = os.path.join("data", "tmdb_cache")


def fetch_and_build_dataset(max_movies: int, force: bool = False) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_CSV) and not force:
        print(f"Using existing dataset: {OUTPUT_CSV}")
        return pd.read_csv(OUTPUT_CSV)

    client = TMDBClient()
    print(f"Discovering up to {max_movies} popular movies from TMDB...")
    movie_ids = client.discover_movie_ids(max_movies=max_movies)
    print(f"Found {len(movie_ids)} movie IDs. Fetching details (cached in {CACHE_DIR})...")

    rows = []
    for idx, movie_id in enumerate(movie_ids, start=1):
        cache_path = os.path.join(CACHE_DIR, f"{movie_id}.json")
        if os.path.exists(cache_path) and not force:
            with open(cache_path, "r", encoding="utf-8") as f:
                row = json.load(f)
        else:
            row = movie_record_from_api(client, movie_id)
            if row:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(row, f, ensure_ascii=False)
        if row:
            rows.append(row)
        if idx % 50 == 0 or idx == len(movie_ids):
            print(f"  Progress: {idx}/{len(movie_ids)} ({len(rows)} valid movies)")

    if not rows:
        raise RuntimeError("No movies fetched from TMDB. Check your API key and network.")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["title"], keep="first")
    df = df.dropna(subset=["overview", "genres", "keywords", "cast", "crew"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} movies to {OUTPUT_CSV}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch TMDB movie dataset")
    parser.add_argument(
        "--max-movies",
        type=int,
        default=int(os.getenv("TMDB_MAX_MOVIES", "1500")),
        help="Number of popular movies to fetch (default: 1500)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch all movies even if cache/CSV exists",
    )
    args = parser.parse_args()
    fetch_and_build_dataset(max_movies=args.max_movies, force=args.force)


if __name__ == "__main__":
    main()
