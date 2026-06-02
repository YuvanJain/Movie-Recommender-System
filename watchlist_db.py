import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "watchlist.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create watchlist table if it does not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                movie_id INTEGER NOT NULL UNIQUE,
                title TEXT NOT NULL,
                added_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def add_to_watchlist(movie_id: int, title: str) -> bool:
    """
    Add a movie to the watchlist.
    Returns True if added, False if already present.
    """
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
        if existing:
            return False

        conn.execute(
            "INSERT INTO watchlist (movie_id, title, added_at) VALUES (?, ?, ?)",
            (movie_id, title, datetime.utcnow().isoformat()),
        )
        conn.commit()
    return True


def remove_from_watchlist(movie_id: int) -> bool:
    """Remove a movie from the watchlist. Returns True if a row was deleted."""
    with get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_watchlist():
    """Return all watchlist entries, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT movie_id, title, added_at
            FROM watchlist
            ORDER BY added_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def is_in_watchlist(movie_id: int) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE movie_id = ?",
            (movie_id,),
        ).fetchone()
    return row is not None


def get_watchlist_stats():
    """Return watchlist summary statistics."""
    items = get_watchlist()
    return {
        "total_movies": len(items),
        "titles": [item["title"] for item in items],
    }
