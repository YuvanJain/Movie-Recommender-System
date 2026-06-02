import os

import pandas as pd
import requests
from dotenv import load_dotenv

from recommendation_model import train_with_audit

load_dotenv()

DATA_SOURCE = os.getenv("DATA_SOURCE", "combined").lower()
TMDB_CSV_PATH = os.path.join("data", "tmdb_movies.csv")
LEGACY_MOVIES = "data/tmdb_5000_movies.csv"
LEGACY_CREDITS = "data/tmdb_5000_credits.csv"


def download_file(url, local_filename):
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename}...")
        response = requests.get(url, stream=True)
        with open(local_filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {local_filename}")
    else:
        print(f"{local_filename} already exists.")


def genres_to_display_from_json(value):
    import ast

    try:
        names = [g["name"] for g in ast.literal_eval(value)]
        return ", ".join(names) if names else None
    except (ValueError, SyntaxError, TypeError):
        return None


def load_legacy_csv() -> pd.DataFrame:
    movies_url = "https://raw.githubusercontent.com/vamshi121/TMDB-5000-Movie-Dataset/main/tmdb_5000_movies.csv"
    credits_url = "https://raw.githubusercontent.com/harshitcodes/tmdb_movie_data_analysis/master/tmdb-5000-movie-dataset/tmdb_5000_credits.csv"
    download_file(movies_url, LEGACY_MOVIES)
    download_file(credits_url, LEGACY_CREDITS)

    movies = pd.read_csv(LEGACY_MOVIES)
    credits = pd.read_csv(LEGACY_CREDITS)
    # Credits also has movie_id — drop it so merge keeps movies.id as movie_id
    credits = credits.drop(columns=["movie_id"], errors="ignore")
    movies = movies.rename(columns={"id": "movie_id"})
    merged = movies.merge(credits, on="title", how="inner")
    return merged


def load_tmdb_api_csv() -> pd.DataFrame:
    if not os.path.exists(TMDB_CSV_PATH):
        print("TMDB CSV missing — fetching from API...")
        from fetch_tmdb_data import fetch_and_build_dataset

        max_movies = int(os.getenv("TMDB_MAX_MOVIES", "1500"))
        fetch_and_build_dataset(max_movies=max_movies, force=False)
    df = pd.read_csv(TMDB_CSV_PATH)
    return df.rename(columns={"id": "movie_id"}, errors="ignore")


def ensure_movie_id_column(movies: pd.DataFrame) -> pd.DataFrame:
    """Resolve movie_id after legacy/credits merge or concat."""
    if "movie_id" not in movies.columns or movies["movie_id"].isna().mean() > 0.5:
        if "movie_id_x" in movies.columns:
            movies["movie_id"] = movies["movie_id_x"]
        elif "id" in movies.columns:
            movies["movie_id"] = movies["id"]
    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    return movies


def prepare_columns(movies: pd.DataFrame) -> pd.DataFrame:
    movies = ensure_movie_id_column(movies)
    if "release_year" not in movies.columns and "release_date" in movies.columns:
        movies["release_year"] = pd.to_datetime(movies["release_date"], errors="coerce").dt.year
    if "rating" not in movies.columns and "vote_average" in movies.columns:
        movies["rating"] = movies["vote_average"]
    if "genres" in movies.columns:
        if "genres_display" not in movies.columns:
            movies["genres_display"] = None
        missing = movies["genres_display"].isna()
        movies.loc[missing, "genres_display"] = movies.loc[missing, "genres"].apply(
            genres_to_display_from_json
        )
    if "overview_display" not in movies.columns and "overview" in movies.columns:
        movies["overview_display"] = movies["overview"]
    return movies


def load_dataset_with_audit() -> pd.DataFrame:
    """Load dataset based on DATA_SOURCE with row-count audit at each step."""
    frames = []
    source_label = DATA_SOURCE

    if DATA_SOURCE in ("csv", "combined", "legacy"):
        legacy = load_legacy_csv()
        print(f"[load_legacy_csv] rows={len(legacy)}")
        frames.append(("legacy", legacy))

    if DATA_SOURCE in ("tmdb", "combined"):
        tmdb = load_tmdb_api_csv()
        print(f"[load_tmdb_api_csv] rows={len(tmdb)}")
        frames.append(("tmdb", tmdb))

    if not frames:
        raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}")

    if len(frames) == 1:
        movies = frames[0][1]
    else:
        legacy_df = frames[0][1]
        tmdb_df = frames[1][1]
        print(f"[before_combine] legacy={len(legacy_df)}, tmdb={len(tmdb_df)}")
        movies = pd.concat([legacy_df, tmdb_df], ignore_index=True)
        print(f"[after_concat] rows={len(movies)}")
        before_dedupe = len(movies)
        movies = movies.drop_duplicates(subset=["title"], keep="first")
        print(
            f"[after_dedupe_by_title] rows={len(movies)} "
            f"(dropped {before_dedupe - len(movies)} duplicate titles; legacy kept first)"
        )
        source_label = "combined (legacy + tmdb api)"

    movies = prepare_columns(movies)
    required = [
        "movie_id", "title", "overview", "genres", "keywords", "cast", "crew",
        "release_year", "rating", "genres_display", "overview_display",
    ]
    missing_cols = [c for c in required if c not in movies.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after load: {missing_cols}")

    movies = movies[required]
    print(f"[after_column_select] rows={len(movies)}")

    pre_clean_count = len(movies)
    before_dropna = len(movies)
    movies = movies.dropna(
        subset=["title", "overview", "genres", "keywords", "cast", "crew"]
    )
    print(
        f"[after_dropna] rows={len(movies)} "
        f"(dropped {before_dropna - len(movies)} rows with missing core fields)"
    )

    movies = movies[movies["overview"].astype(str).str.strip().str.len() > 20]
    print(f"[after_overview_length_filter] rows={len(movies)}")

    print(f"\nDataset source: {source_label}")
    print(f"Rows before cleaning: {pre_clean_count}")
    print(f"Final training size: {len(movies)}")
    return movies.reset_index(drop=True)


def train_model():
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    print("=" * 60)
    print("MOVIE RECOMMENDER — TRAINING PIPELINE AUDIT")
    print("=" * 60)

    movies = load_dataset_with_audit()
    audit = train_with_audit(movies, audit=True)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final movies in model: {audit['final_movie_count']}")
    print(f"Similarity matrix: {audit['similarity_shape']}")
    print("=" * 60)


if __name__ == "__main__":
    train_model()
