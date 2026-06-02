import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "analytics.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_analytics_db():
    """Create analytics tables if they do not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_title TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_movie TEXT NOT NULL,
                result_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_impressions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_movie TEXT NOT NULL,
                recommended_movie TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def log_search(movie_title: str):
    """Record a movie search (triggered when user requests recommendations)."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO search_events (movie_title, created_at) VALUES (?, ?)",
            (movie_title, datetime.utcnow().isoformat()),
        )
        conn.commit()


def log_recommendation_batch(source_movie: str, recommended_titles: list):
    """Record a recommendation session and each recommended title."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recommendation_sessions (source_movie, result_count, created_at)
            VALUES (?, ?, ?)
            """,
            (source_movie, len(recommended_titles), now),
        )
        conn.executemany(
            """
            INSERT INTO recommendation_impressions
                (source_movie, recommended_movie, created_at)
            VALUES (?, ?, ?)
            """,
            [(source_movie, title, now) for title in recommended_titles],
        )
        conn.commit()


def get_most_searched_movies(limit=10):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_title AS movie, COUNT(*) AS count
            FROM search_events
            GROUP BY movie_title
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_most_recommended_movies(limit=10):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT recommended_movie AS movie, COUNT(*) AS count
            FROM recommendation_impressions
            GROUP BY recommended_movie
            ORDER BY count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_recommendation_counts():
    with get_connection() as conn:
        sessions = conn.execute(
            "SELECT COUNT(*) AS total FROM recommendation_sessions"
        ).fetchone()["total"]
        impressions = conn.execute(
            "SELECT COUNT(*) AS total FROM recommendation_impressions"
        ).fetchone()["total"]
        searches = conn.execute(
            "SELECT COUNT(*) AS total FROM search_events"
        ).fetchone()["total"]
    return {
        "total_searches": searches,
        "total_sessions": sessions,
        "total_impressions": impressions,
    }


def get_genre_counts_from_titles(title_genre_map: dict, titles: list):
    """Count genre occurrences from a list of movie titles."""
    counts = {}
    for title in titles:
        genres = title_genre_map.get(title)
        if not genres or genres == "N/A":
            continue
        for genre in str(genres).split(","):
            genre = genre.strip()
            if genre:
                counts[genre] = counts.get(genre, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)
