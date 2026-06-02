import streamlit as st
import pickle
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import ast
import html
from watchlist_db import init_db, add_to_watchlist, remove_from_watchlist, get_watchlist, is_in_watchlist
from feedback_db import init_feedback_db, save_feedback, FEEDBACK_HELPFUL, FEEDBACK_NOT_HELPFUL
from analytics_db import init_analytics_db, log_search, log_recommendation_batch
from stats_ui import render_statistics_dashboard
from recommendation_model import MovieRecommender, format_similarity_score

load_dotenv()

OVERVIEW_MAX_CHARS = 120
MAX_SEARCH_SUGGESTIONS = 8


@st.cache_data
def get_sorted_titles(titles_tuple):
    """Cache sorted movie titles for fast autocomplete lookups."""
    return sorted(titles_tuple)


def search_movies(query, titles, max_results=MAX_SEARCH_SUGGESTIONS):
    """
    Return ranked autocomplete suggestions: prefix matches first,
    then substring matches, then word-start matches.
    """
    q = query.strip().lower()
    if not q:
        return []

    prefix, contains, word_start = [], [], []
    for title in titles:
        lower_title = title.lower()
        if lower_title.startswith(q):
            prefix.append(title)
        elif q in lower_title:
            contains.append(title)
        elif any(word.startswith(q) for word in lower_title.replace('-', ' ').replace(':', ' ').split()):
            word_start.append(title)

    seen = set()
    results = []
    for group in (prefix, contains, word_start):
        for title in group:
            if title not in seen:
                seen.add(title)
                results.append(title)
                if len(results) >= max_results:
                    return results
    return results


def find_exact_title(query, titles):
    """Return the canonical title when the query matches exactly (case-insensitive)."""
    q = query.strip().lower()
    if not q:
        return None
    for title in titles:
        if title.lower() == q:
            return title
    return None


def resolve_selected_movie(query, suggestions, titles):
    """
    Resolve the movie to recommend from typed text and/or autocomplete selection.
    Exact typed matches take priority; otherwise use the selected suggestion.
    """
    exact = find_exact_title(query, titles)
    if exact:
        return exact
    if suggestions:
        return suggestions[0]
    return None


def safe_movie_id(value):
    """Convert movie_id to int, or None if missing/invalid."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        mid = int(float(value))
        return mid if mid > 0 else None
    except (ValueError, TypeError):
        return None


@st.cache_data
def load_title_to_tmdb_id():
    """Fallback map title -> TMDB id from raw CSV files."""
    mapping = {}
    for path, id_col in [
        ("data/tmdb_5000_movies.csv", "id"),
        ("data/tmdb_movies.csv", "movie_id"),
    ]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["title", id_col])
        for _, row in df.iterrows():
            title = str(row["title"]).strip()
            mid = safe_movie_id(row[id_col])
            if title and mid and title not in mapping:
                mapping[title] = mid
    return mapping


TITLE_TO_TMDB_ID = {}


def resolve_movie_id(movie_id, title):
    """Get a valid TMDB id from movie_id column or title fallback map."""
    mid = safe_movie_id(movie_id)
    if mid is not None:
        return mid
    return TITLE_TO_TMDB_ID.get(title)


def fetch_poster(movie_id, title=None):
    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        return "https://via.placeholder.com/500x750?text=TMDB+API+Key+Missing"

    mid = resolve_movie_id(movie_id, title)
    if mid is None and title:
        try:
            search = requests.get(
                "https://api.themoviedb.org/3/search/movie",
                params={"api_key": api_key, "query": title, "language": "en-US"},
                timeout=15,
            )
            search.raise_for_status()
            results = search.json().get("results", [])
            if results:
                mid = results[0].get("id")
        except requests.RequestException:
            pass

    if mid is None:
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{mid}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("poster_path"):
            return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"
    except requests.RequestException:
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"


@st.cache_data
def load_csv_metadata():
    """
    Fallback loader: if the saved pickle was trained before metadata columns
    were added, pull year/rating/genres/overview from the raw TMDB CSV on disk.
    """
    csv_path = 'data/tmdb_movies.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/tmdb_5000_movies.csv'
    if not os.path.exists(csv_path):
        return None

    raw = pd.read_csv(
        csv_path,
        usecols=['id', 'release_date', 'vote_average', 'genres', 'overview'],
    )
    raw = raw.rename(columns={'id': 'movie_id'})
    raw['release_year'] = pd.to_datetime(raw['release_date'], errors='coerce').dt.year
    raw['rating'] = raw['vote_average']
    raw['overview_display'] = raw['overview']

    def parse_genres(genres_json):
        try:
            names = [g['name'] for g in ast.literal_eval(genres_json)]
            return ', '.join(names) if names else None
        except (ValueError, SyntaxError, TypeError):
            return None

    raw['genres_display'] = raw['genres'].apply(parse_genres)
    return raw[['movie_id', 'release_year', 'rating', 'genres_display', 'overview_display']]


def enrich_movies_with_metadata(movies_df):
    """
    Merge display metadata into the movies dataframe when the pickle
    does not already contain release_year, rating, genres_display, or overview_display.
    """
    metadata_cols = ['release_year', 'rating', 'genres_display', 'overview_display']
    if all(col in movies_df.columns for col in metadata_cols):
        return movies_df

    csv_meta = load_csv_metadata()
    if csv_meta is None:
        return movies_df

    # Only fill in columns that are missing from the pickle
    missing_cols = [col for col in metadata_cols if col not in movies_df.columns]
    merge_cols = ['movie_id'] + missing_cols
    return movies_df.merge(csv_meta[merge_cols], on='movie_id', how='left')


def format_overview_html(overview):
    """
    Render a short overview (~120 chars) with an expandable Read More section
    when the full text is longer.
    """
    if overview is None or (isinstance(overview, float) and pd.isna(overview)):
        return '<div class="movie-overview">N/A</div>'

    text = str(overview).strip()
    if not text:
        return '<div class="movie-overview">N/A</div>'

    safe_text = html.escape(text)

    if len(text) <= OVERVIEW_MAX_CHARS:
        return f'<div class="movie-overview">{safe_text}</div>'

    preview = text[:OVERVIEW_MAX_CHARS]
    if ' ' in preview:
        preview = preview.rsplit(' ', 1)[0]
    preview = preview.rstrip('.,;:')
    safe_preview = html.escape(preview)

    return f'''<div class="movie-overview">
        <details class="overview-toggle">
            <summary class="overview-summary">
                <span class="overview-short">{safe_preview}...</span>
                <span class="overview-read-more-btn">Read More</span>
            </summary>
            <span class="overview-expanded">{safe_text}</span>
        </details>
    </div>'''


def get_movie_details(row):
    """
    Build display fields for a recommendation card.
    Returns 'N/A' for any missing or empty value.
    """
    details = {
        'movie_id': safe_movie_id(row.get('movie_id')),
        'title': row['title'] if pd.notna(row.get('title')) else 'N/A',
        'year': 'N/A',
        'rating': 'N/A',
        'genres': 'N/A',
        'overview_html': 'N/A',
    }

    if 'release_year' in row.index and pd.notna(row['release_year']):
        details['year'] = str(int(row['release_year']))

    if 'rating' in row.index and pd.notna(row['rating']):
        details['rating'] = f"{float(row['rating']):.1f}/10"

    if 'genres_display' in row.index and pd.notna(row['genres_display']):
        genres_text = str(row['genres_display']).strip()
        if genres_text:
            details['genres'] = genres_text

    overview = row['overview_display'] if 'overview_display' in row.index else None
    details['overview_html'] = format_overview_html(overview)

    return details


def recommend(movie):
    """Return top-8 recommendations using TF-IDF cosine + metadata hybrid re-ranking."""
    recs = recommender.recommend(movie, top_n=8)
    recommended_movies = []
    recommended_movies_posters = []

    for rec in recs:
        movie_row = movies[movies["title"] == rec["title"]].iloc[0]
        details = get_movie_details(movie_row)
        details["similarity"] = format_similarity_score(rec["similarity_score"])
        recommended_movies.append(details)
        movie_id = rec.get("movie_id") or details.get("movie_id")
        recommended_movies_posters.append(
            fetch_poster(movie_id, title=details["title"])
        )

    return recommended_movies, recommended_movies_posters


def get_movie_id_by_title(title):
    """Look up movie_id from the loaded dataset by title."""
    matches = movies[movies['title'] == title]
    if matches.empty:
        return resolve_movie_id(None, title)
    mid = safe_movie_id(matches.iloc[0]['movie_id'])
    return mid if mid is not None else resolve_movie_id(None, title)


def render_watchlist_sidebar(selected_title=None):
    """Render watchlist management in the sidebar."""
    with st.sidebar:
        st.markdown("### 📌 My Watchlist")
        watchlist_items = get_watchlist()
        st.caption(f"**{len(watchlist_items)}** saved · **{len(all_titles):,}** movies in catalog")

        if selected_title and selected_title in valid_titles:
            movie_id = get_movie_id_by_title(selected_title)
            if movie_id is not None:
                if is_in_watchlist(movie_id):
                    st.info(f"'{selected_title}' is in your watchlist.")
                elif st.button("➕ Add searched movie", key="sidebar_add_searched", use_container_width=True):
                    if add_to_watchlist(movie_id, selected_title):
                        st.session_state.watchlist_flash = f"Added '{selected_title}'"
                        st.rerun()

        if st.session_state.get("watchlist_flash"):
            st.success(st.session_state.watchlist_flash)
            st.session_state.watchlist_flash = None

        st.divider()

        if not watchlist_items:
            st.caption("Your watchlist is empty. Add movies from recommendations or search.")
        else:
            for item in watchlist_items:
                col_title, col_remove = st.columns([4, 1])
                with col_title:
                    st.markdown(f"**{item['title']}**")
                with col_remove:
                    if st.button("✕", key=f"remove_wl_{item['movie_id']}", help="Remove from watchlist"):
                        remove_from_watchlist(item['movie_id'])
                        st.rerun()


def render_recommendations(source_movie, names, posters):
    """Display recommendation grid with watchlist and feedback actions."""
    st.markdown(
        f'<div class="results-header">Top picks based on <span>{html.escape(source_movie)}</span></div>',
        unsafe_allow_html=True,
    )

    for row in range(2):
        cols = st.columns(4)
        for col_idx in range(4):
            movie_idx = row * 4 + col_idx
            if movie_idx >= len(names):
                continue

            movie = names[movie_idx]
            safe_title = html.escape(movie['title'])

            with cols[col_idx]:
                st.markdown(
                    f'''
                    <div class="movie-card">
                        <div class="similarity-badge">{html.escape(movie['similarity'])}</div>
                        <img src="{posters[movie_idx]}" class="movie-poster" alt="{safe_title}">
                        <div class="movie-title">{safe_title}</div>
                        <div class="movie-meta"><span class="movie-meta-label">Year:</span> {movie['year']}</div>
                        <div class="movie-meta"><span class="movie-meta-label">Rating:</span> {movie['rating']}</div>
                        <div class="movie-meta"><span class="movie-meta-label">Genres:</span> {movie['genres']}</div>
                        {movie['overview_html']}
                    </div>
                    ''',
                    unsafe_allow_html=True,
                )

                movie_id = movie.get('movie_id')
                if movie_id is not None:
                    if is_in_watchlist(movie_id):
                        st.button(
                            "✓ In Watchlist",
                            key=f"add_wl_{movie_idx}_{movie_id}",
                            disabled=True,
                            use_container_width=True,
                        )
                    elif st.button(
                        "➕ Add to Watchlist",
                        key=f"add_wl_{movie_idx}_{movie_id}",
                        use_container_width=True,
                    ):
                        if add_to_watchlist(movie_id, movie['title']):
                            st.session_state.watchlist_flash = f"Added '{movie['title']}'"
                            st.rerun()

                feedback_key = f"{source_movie}::{movie['title']}"
                if feedback_key in st.session_state.feedback_given:
                    st.markdown(
                        f'<p class="feedback-thanks">{st.session_state.feedback_given[feedback_key]}</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    fb_col1, fb_col2 = st.columns(2)
                    with fb_col1:
                        if st.button(
                            "👍 Helpful",
                            key=f"fb_up_{movie_idx}_{movie_id or 0}",
                            use_container_width=True,
                        ):
                            save_feedback(source_movie, movie['title'], FEEDBACK_HELPFUL)
                            st.session_state.feedback_given[feedback_key] = "👍 Thanks — marked helpful"
                            st.rerun()
                    with fb_col2:
                        if st.button(
                            "👎 Not Helpful",
                            key=f"fb_down_{movie_idx}_{movie_id or 0}",
                            use_container_width=True,
                        ):
                            save_feedback(source_movie, movie['title'], FEEDBACK_NOT_HELPFUL)
                            st.session_state.feedback_given[feedback_key] = "👎 Thanks — we'll improve"
                            st.rerun()

# --- UI Setup ---
st.set_page_config(
    page_title="Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Remove top white bar and tighten layout */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
}
div[data-testid="stToolbar"] {
    display: none !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 95% !important;
}
.stApp {
    background: linear-gradient(180deg, #0a0c10 0%, #0E1117 35%, #12151c 100%);
    color: #FAFAFA;
}
/* Hero */
.app-hero {
    text-align: center;
    padding: 0.5rem 1rem 1.75rem 1rem;
    margin-bottom: 0.5rem;
}
.app-hero h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
    background: linear-gradient(90deg, #ffffff, #a8c7fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.app-hero p {
    color: #9ca3af !important;
    font-size: 1.05rem;
    margin: 0.5rem 0 0 0;
}
/* Bordered search container */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(38, 39, 48, 0.6) !important;
    border-color: #3d4450 !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    margin-bottom: 1rem !important;
}
.results-header {
    color: #e5e7eb;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 1.5rem 0 1.25rem 0;
    padding-left: 0.25rem;
}
.results-header span {
    color: #60a5fa;
}
.feedback-thanks {
    text-align: center;
    font-size: 0.8rem;
    color: #86efac !important;
    margin: 0.5rem 0 0 0;
    font-weight: 600;
}
/* Force selectbox label to be light */
.stSelectbox label p {
    color: #FAFAFA !important;
    font-size: 1.2rem !important;
}
/* Search bar dark theme */
.stTextInput label p {
    color: #FAFAFA !important;
    font-size: 1.2rem !important;
}
.stTextInput > div > div > input {
    background-color: #262730 !important;
    color: #FAFAFA !important;
    border: 1px solid #555555 !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    padding: 0.6rem 0.75rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #FFFFFF !important;
    box-shadow: 0 0 0 1px #FFFFFF !important;
}
.stTextInput > div > div > input::placeholder {
    color: #888888 !important;
}
.search-suggestions-label {
    color: #AAAAAA !important;
    font-size: 0.85rem !important;
    margin: 0.35rem 0 0.25rem 0 !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background-color: #262730 !important;
    border-color: #555555 !important;
    border-radius: 8px !important;
    color: #FAFAFA !important;
}
div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
    fill: #FAFAFA !important;
}
.search-no-results {
    color: #AAAAAA !important;
    font-size: 0.85rem;
    margin-top: 0.35rem;
}
/* Primary CTA */
.stMain div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.75rem !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
}
.stMain div.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.55) !important;
}
/* Card row buttons (watchlist + feedback) — high contrast on dark background */
div[data-testid="column"] div.stButton > button {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #d1d5db !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
div[data-testid="column"] div.stButton > button p,
div[data-testid="column"] div.stButton > button span,
div[data-testid="column"] div.stButton > button div {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}
div[data-testid="column"] div.stButton > button:hover {
    background: #f3f4f6 !important;
    color: #000000 !important;
    border-color: #9ca3af !important;
    transform: none !important;
}
div[data-testid="column"] div.stButton > button:hover p,
div[data-testid="column"] div.stButton > button:hover span {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
div[data-testid="column"] div.stButton > button:disabled {
    background: #e5e7eb !important;
    color: #374151 !important;
    border-color: #9ca3af !important;
    opacity: 1 !important;
}
div[data-testid="column"] div.stButton > button:disabled p,
div[data-testid="column"] div.stButton > button:disabled span {
    color: #374151 !important;
    -webkit-text-fill-color: #374151 !important;
}
/* Style the movie cards */
.movie-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 14px;
    padding: 15px;
    box-shadow: 0 12px 28px rgba(0,0,0,0.45);
    height: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 1px solid #e2e8f0;
    position: relative;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(0,0,0,0.5);
}
.similarity-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: white !important;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 4px 8px;
    border-radius: 20px;
    z-index: 1;
    box-shadow: 0 2px 8px rgba(34, 197, 94, 0.4);
}
.movie-title {
    font-size: 1.0rem;
    font-weight: 700;
    text-align: center;
    margin-top: 12px;
    color: #000000 !important;
}
.movie-similarity {
    display: none;
}
.movie-meta {
    font-size: 0.85rem;
    text-align: center;
    margin-top: 6px;
    color: #333333 !important;
    line-height: 1.4;
    width: 100%;
}
.movie-meta-label {
    font-weight: 600;
    color: #555555 !important;
}
/* ENHANCEMENT: Overview text and Read More toggle on each card */
.movie-overview {
    font-size: 0.8rem;
    text-align: center;
    margin-top: 10px;
    color: #444444 !important;
    line-height: 1.45;
    width: 100%;
}
.overview-toggle .overview-expanded {
    display: none;
}
.overview-toggle[open] .overview-expanded {
    display: block;
}
.overview-toggle[open] .overview-short,
.overview-toggle[open] .overview-read-more-btn {
    display: none;
}
.overview-summary {
    cursor: pointer;
    list-style: none;
}
.overview-summary::-webkit-details-marker {
    display: none;
}
.overview-read-more-btn {
    color: #1a1a1a !important;
    font-weight: 700;
    margin-left: 4px;
}
.movie-poster {
    border-radius: 8px;
    width: 100%;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}
h1, h2, h3 {
    color: #FAFAFA !important;
}
section[data-testid="stSidebar"] {
    background-color: #0E1117;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stCaption {
    color: #FAFAFA !important;
}
div[data-testid="stSidebar"] div.stButton > button {
    background-color: #262730 !important;
    color: #FAFAFA !important;
    border: 1px solid #555555 !important;
    font-weight: 600 !important;
    padding: 0.25rem 0.5rem !important;
}
div[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #3A3B45 !important;
    border-color: #888888 !important;
}
/* Statistics tab metrics */
[data-testid="stMetric"] {
    background: rgba(38, 39, 48, 0.8);
    border: 1px solid #3d4450;
    border-radius: 12px;
    padding: 0.75rem;
}
[data-testid="stMetricLabel"] { color: #9ca3af !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; }
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: #262730 !important;
    color: #9ca3af !important;
    border-radius: 8px 8px 0 0;
    padding: 0.6rem 1.25rem;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: #3b82f6 !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

_dataset_label = "TMDB API" if os.getenv("DATA_SOURCE", "tmdb").lower() == "tmdb" else "TMDB 5000 CSV"
st.markdown(f"""
<div class="app-hero">
    <h1>🎬 Movie Recommender</h1>
    <p>Discover films you'll love — powered by content-based AI similarity · Dataset: {_dataset_label}</p>
</div>
""", unsafe_allow_html=True)

# Ensure the model files exist
try:
    recommender = MovieRecommender.load()
    movies = recommender.movies
    movies = enrich_movies_with_metadata(movies)
    TITLE_TO_TMDB_ID.update(load_title_to_tmdb_id())
except (FileNotFoundError, ValueError) as exc:
    # On Streamlit Cloud, the large pickled files aren't in Git. We auto-build them on the fly.
    with st.spinner("Model files not found or outdated. Rebuilding the recommendation model (approx. 30s-1m)..."):
        try:
            from train import train_model
            train_model()
            recommender = MovieRecommender.load()
            movies = recommender.movies
            movies = enrich_movies_with_metadata(movies)
            TITLE_TO_TMDB_ID.update(load_title_to_tmdb_id())
            st.success("Model built successfully!")
        except Exception as train_exc:
            st.error(
                "Model files not found, and auto-training failed. Please run `python train.py` locally or check settings."
            )
            st.caption(f"Load error: {str(exc)}")
            st.caption(f"Training error: {str(train_exc)}")
            st.stop()

init_db()
init_feedback_db()
init_analytics_db()

if "rec_source" not in st.session_state:
    st.session_state.rec_source = None
    st.session_state.rec_names = None
    st.session_state.rec_posters = None
    st.session_state.feedback_given = {}
if "watchlist_flash" not in st.session_state:
    st.session_state.watchlist_flash = None

all_titles = get_sorted_titles(tuple(movies['title'].tolist()))
valid_titles = set(all_titles)

tab_recommender, tab_statistics = st.tabs(["🎬 Recommender", "📊 Statistics"])

selected_movie_name = None

with tab_recommender:
    with st.container(border=True):
        st.markdown("##### 🔍 Find your next favorite")
        search_query = st.text_input(
            "Search for a movie",
            placeholder="Start typing a movie name...",
            key="movie_search",
            label_visibility="collapsed",
        )

        suggestions = search_movies(search_query, all_titles) if search_query.strip() else []

        if search_query.strip():
            if suggestions:
                st.markdown('<p class="search-suggestions-label">Suggestions</p>', unsafe_allow_html=True)
                picked_suggestion = st.selectbox(
                    "Movie suggestions",
                    options=suggestions,
                    label_visibility="collapsed",
                    key="movie_suggestions",
                )
                selected_movie_name = find_exact_title(search_query, all_titles) or picked_suggestion
            else:
                st.markdown(
                    '<p class="search-no-results">No matching movies found. Try a different search.</p>',
                    unsafe_allow_html=True,
                )
                selected_movie_name = find_exact_title(search_query, all_titles)

        recommend_clicked = st.button("✨ Get Recommendations", type="primary")

    if recommend_clicked:
        movie_to_recommend = selected_movie_name or resolve_selected_movie(
            search_query, suggestions, all_titles
        )

        if not movie_to_recommend or movie_to_recommend not in valid_titles:
            st.warning("Please type or select a valid movie from the suggestions.")
            st.session_state.rec_source = None
            st.session_state.rec_names = None
            st.session_state.rec_posters = None
        else:
            with st.spinner("Finding movies you'll love..."):
                names, posters = recommend(movie_to_recommend)
            log_search(movie_to_recommend)
            log_recommendation_batch(movie_to_recommend, [m["title"] for m in names])
            st.session_state.rec_source = movie_to_recommend
            st.session_state.rec_names = names
            st.session_state.rec_posters = posters
            st.session_state.feedback_given = {}

    if st.session_state.rec_names and st.session_state.rec_posters:
        render_recommendations(
            st.session_state.rec_source,
            st.session_state.rec_names,
            st.session_state.rec_posters,
        )

with tab_statistics:
    render_statistics_dashboard()

render_watchlist_sidebar(selected_movie_name)
