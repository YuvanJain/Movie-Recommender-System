import os
import pickle
import ast

import pandas as pd
import streamlit as st

from analytics_db import (
    get_most_searched_movies,
    get_most_recommended_movies,
    get_recommendation_counts,
    get_genre_counts_from_titles,
    get_connection as get_analytics_connection,
)
from feedback_db import get_most_liked_recommendations, get_feedback_summary
from watchlist_db import get_watchlist_stats


@st.cache_data
def load_title_genre_map():
    """Build a title -> genres lookup from the trained model / CSV fallback."""
    try:
        movies_dict = pickle.load(open("model/movies_dict.pkl", "rb"))
        df = pd.DataFrame(movies_dict)
    except FileNotFoundError:
        return {}

    if "genres_display" not in df.columns:
        csv_path = "data/tmdb_movies.csv"
        if not os.path.exists(csv_path):
            csv_path = "data/tmdb_5000_movies.csv"
        if os.path.exists(csv_path):
            raw = pd.read_csv(csv_path, usecols=["title", "genres"])

            def parse_genres(genres_json):
                try:
                    names = [g["name"] for g in ast.literal_eval(genres_json)]
                    return ", ".join(names) if names else None
                except (ValueError, SyntaxError, TypeError):
                    return None

            raw["genres_display"] = raw["genres"].apply(parse_genres)
            df = df.merge(raw[["title", "genres_display"]], on="title", how="left")

    if "genres_display" not in df.columns:
        return {}

    return dict(zip(df["title"], df["genres_display"].fillna("N/A")))


def rows_to_chart_df(rows, label_col="movie", value_col="count"):
    if not rows:
        return pd.DataFrame(columns=[label_col, value_col])
    return pd.DataFrame(rows)


def genre_rows_to_df(genre_counts, limit=12):
    if not genre_counts:
        return pd.DataFrame(columns=["genre", "count"])
    top = genre_counts[:limit]
    return pd.DataFrame(top, columns=["genre", "count"])


def render_statistics_dashboard():
    """Render all analytics bar charts and metrics (shared by main app tab and admin page)."""
    st.subheader("📊 Usage Statistics")
    st.caption(
        "Live stats from your activity — searches, recommendations, feedback, and watchlist. "
        "Use **Get Recommendations** on the Recommender tab to generate data."
    )

    title_genre_map = load_title_genre_map()
    rec_counts = get_recommendation_counts()
    feedback_summary = get_feedback_summary()
    watchlist_stats = get_watchlist_stats()

    st.markdown("#### Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Searches", rec_counts["total_searches"])
    m2.metric("Recommendation Sessions", rec_counts["total_sessions"])
    m3.metric("Movies Recommended", rec_counts["total_impressions"])
    m4.metric("Watchlist Size", watchlist_stats["total_movies"])
    m5.metric("Helpful Votes", feedback_summary["helpful"])

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🔍 Most Searched Movies")
        searched = get_most_searched_movies(10)
        search_df = rows_to_chart_df(searched)
        if search_df.empty:
            st.info("No search data yet. Get recommendations to log a search.")
        else:
            st.bar_chart(search_df.set_index("movie")["count"], color="#3b82f6")

    with col_right:
        st.markdown("#### 🎬 Most Recommended Movies")
        recommended = get_most_recommended_movies(10)
        rec_df = rows_to_chart_df(recommended)
        if rec_df.empty:
            st.info("No recommendation data yet.")
        else:
            st.bar_chart(rec_df.set_index("movie")["count"], color="#22c55e")

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.markdown("#### 👍 Most Liked Recommendations")
        liked = get_most_liked_recommendations(10)
        liked_df = rows_to_chart_df(liked)
        if liked_df.empty:
            st.info("No helpful feedback yet. Click 👍 Helpful on a recommendation card.")
        else:
            st.bar_chart(liked_df.set_index("movie")["count"], color="#a855f7")

    with col_right2:
        st.markdown("#### 📌 Watchlist Statistics")
        wl_count = watchlist_stats["total_movies"]
        st.metric("Movies on Watchlist", wl_count)

        if wl_count == 0:
            st.info("No movies on your watchlist yet.")
        else:
            wl_titles = watchlist_stats["titles"]
            wl_genres = get_genre_counts_from_titles(title_genre_map, wl_titles)
            wl_genre_df = genre_rows_to_df(wl_genres, 8)
            if not wl_genre_df.empty:
                st.caption("Genres in watchlist")
                st.bar_chart(wl_genre_df.set_index("genre")["count"], color="#f59e0b")
            st.caption("Saved titles")
            st.dataframe(
                pd.DataFrame({"Title": wl_titles}),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()

    col_genre, col_counts = st.columns(2)

    with col_genre:
        st.markdown("#### 🎭 Genre Popularity")
        with get_analytics_connection() as conn:
            impression_titles = [
                row["recommended_movie"]
                for row in conn.execute(
                    "SELECT recommended_movie FROM recommendation_impressions"
                ).fetchall()
            ]
            search_titles = [
                row["movie_title"]
                for row in conn.execute("SELECT movie_title FROM search_events").fetchall()
            ]

        all_titles = impression_titles + search_titles + watchlist_stats["titles"]
        genre_counts = get_genre_counts_from_titles(title_genre_map, all_titles)
        genre_df = genre_rows_to_df(genre_counts, 15)

        if genre_df.empty:
            st.info("No genre data yet.")
        else:
            st.bar_chart(genre_df.set_index("genre")["count"], color="#ec4899")

    with col_counts:
        st.markdown("#### 📈 Recommendation Counts")
        counts_df = pd.DataFrame(
            {
                "Metric": [
                    "Searches",
                    "Recommendation sessions",
                    "Movies shown",
                    "Helpful feedback",
                    "Not helpful feedback",
                ],
                "Count": [
                    rec_counts["total_searches"],
                    rec_counts["total_sessions"],
                    rec_counts["total_impressions"],
                    feedback_summary["helpful"],
                    feedback_summary["not_helpful"],
                ],
            }
        )
        st.bar_chart(counts_df.set_index("Metric")["Count"], color="#06b6d4")

        if feedback_summary["total"] > 0:
            feedback_chart = pd.DataFrame(
                {
                    "Type": ["Helpful", "Not Helpful"],
                    "Count": [feedback_summary["helpful"], feedback_summary["not_helpful"]],
                }
            )
            st.caption("Feedback breakdown")
            st.bar_chart(feedback_chart.set_index("Type")["Count"], color="#6366f1")
