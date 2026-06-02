# 🎬 Content-Based Movie Recommender System

A content-based movie recommendation engine built with **Python**, **Streamlit**, and **Machine Learning**. It uses **Cosine Similarity** on genres, keywords, cast, and crew — powered by the **TMDB API** for a larger, up-to-date catalog.

## ✨ Features

- **TMDB API integration** — fetch hundreds or thousands of popular movies directly from [The Movie Database](https://www.themoviedb.org/)
- **Legacy CSV mode** — optional ~5,000 movie static dataset (no API fetch)
- **NLTK** stemming for tag matching
- **Genre-weighted** tags (3× weight on genres)
- Dark Streamlit UI with search, watchlist, feedback, and statistics tabs
- Posters loaded live from TMDB using your API key

## 🛠️ Tech Stack

- Streamlit, Pandas, Scikit-learn, NLTK, Requests, python-dotenv
- SQLite for watchlist, feedback, and analytics
- TMDB API v3 for dataset + posters

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure TMDB API key

1. Create a free account at [themoviedb.org](https://www.themoviedb.org/signup)
2. Open [API Settings](https://www.themoviedb.org/settings/api) and create an API key (v3)
3. Copy `.env.example` to `.env` and set your key:

```bash
copy .env.example .env
```

Edit `.env`:

```env
TMDB_API_KEY=your_actual_api_key_here
DATA_SOURCE=tmdb
TMDB_MAX_MOVIES=1500
```

| Variable | Description |
|----------|-------------|
| `TMDB_API_KEY` | Required for posters and TMDB dataset fetch |
| `DATA_SOURCE` | `tmdb` (live API dataset) or `csv` (legacy GitHub CSVs) |
| `TMDB_MAX_MOVIES` | How many popular movies to download (default `1500`) |
| `TMDB_REQUEST_DELAY` | Pause between API calls in seconds (default `0.25`) |

### 3. Build dataset and train model

**Option A — TMDB API (recommended, larger catalog):**

```bash
# Step 1: Download movies from TMDB (cached in data/tmdb_cache/, saved to data/tmdb_movies.csv)
python fetch_tmdb_data.py

# Optional: fetch more titles
python fetch_tmdb_data.py --max-movies 2500

# Step 2: Train similarity model
python train.py
```

Or let training fetch automatically on first run:

```bash
python train.py
```

**Option B — Legacy CSV (~5000 movies, no API fetch for training data):**

```env
DATA_SOURCE=csv
```

```bash
python train.py
```

### 4. Run the app

```bash
streamlit run app.py
```

Use the sidebar **Admin Dashboard** page or the **Statistics** tab for analytics.

## 📁 Project layout

| File | Purpose |
|------|---------|
| `tmdb_client.py` | TMDB API client |
| `fetch_tmdb_data.py` | Download & cache movies from TMDB |
| `train.py` | Build similarity matrix from dataset |
| `app.py` | Main recommender UI |
| `data/tmdb_movies.csv` | Dataset built from API (generated) |
| `model/movies_dict.pkl` | Trained movie metadata |
| `model/similarity.pkl` | Cosine similarity matrix |

## 🎥 Data source

- **TMDB mode:** Movies discovered via `/discover/movie` (popularity), with details, keywords, and credits per title.
- **CSV mode:** Static [TMDB 5000](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) exports from GitHub.

This product uses the TMDB API but is not endorsed or certified by TMDB.

---
<<<<<<< HEAD

=======
>>>>>>> 8ebd0e9 (Improve movie recommendation system)
