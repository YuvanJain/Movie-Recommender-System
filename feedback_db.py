import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join("data", "feedback.db")

FEEDBACK_HELPFUL = "helpful"
FEEDBACK_NOT_HELPFUL = "not_helpful"


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_feedback_db():
    """Create recommendation_feedback table if it does not exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                selected_movie TEXT NOT NULL,
                recommended_movie TEXT NOT NULL,
                feedback_type TEXT NOT NULL
                    CHECK (feedback_type IN ('helpful', 'not_helpful')),
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_feedback(selected_movie: str, recommended_movie: str, feedback_type: str):
    """Store user feedback for a recommendation."""
    if feedback_type not in (FEEDBACK_HELPFUL, FEEDBACK_NOT_HELPFUL):
        raise ValueError("feedback_type must be 'helpful' or 'not_helpful'")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO recommendation_feedback
                (selected_movie, recommended_movie, feedback_type, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                selected_movie,
                recommended_movie,
                feedback_type,
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def get_most_liked_recommendations(limit=10):
    """Return recommended movies with the most helpful feedback."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT recommended_movie AS movie, COUNT(*) AS count
            FROM recommendation_feedback
            WHERE feedback_type = ?
            GROUP BY recommended_movie
            ORDER BY count DESC
            LIMIT ?
            """,
            (FEEDBACK_HELPFUL, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def get_feedback_summary():
    with get_connection() as conn:
        helpful = conn.execute(
            "SELECT COUNT(*) AS c FROM recommendation_feedback WHERE feedback_type = ?",
            (FEEDBACK_HELPFUL,),
        ).fetchone()["c"]
        not_helpful = conn.execute(
            "SELECT COUNT(*) AS c FROM recommendation_feedback WHERE feedback_type = ?",
            (FEEDBACK_NOT_HELPFUL,),
        ).fetchone()["c"]
    return {"helpful": helpful, "not_helpful": not_helpful, "total": helpful + not_helpful}
