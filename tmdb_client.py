"""
TMDB API v3 client for building the movie recommendation dataset.
"""

import json
import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LANGUAGE = "en-US"


class TMDBClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TMDB_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TMDB_API_KEY is missing. Add it to your .env file. "
                "Get a key at https://www.themoviedb.org/settings/api"
            )
        self.session = requests.Session()
        self._last_request_at = 0.0
        self.request_delay = float(os.getenv("TMDB_REQUEST_DELAY", "0.25"))

    def _throttle(self):
        elapsed = time.time() - self._last_request_at
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_at = time.time()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        self._throttle()
        params = dict(params or {})
        params["api_key"] = self.api_key
        params.setdefault("language", DEFAULT_LANGUAGE)
        url = f"{TMDB_BASE_URL}{path}"
        response = self.session.get(url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(2)
            response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def discover_movie_ids(self, max_movies: int = 1500) -> list[int]:
        """Collect popular movie IDs via TMDB discover (paginated)."""
        movie_ids = []
        page = 1
        while len(movie_ids) < max_movies:
            data = self._get(
                "/discover/movie",
                {
                    "sort_by": "popularity.desc",
                    "include_adult": "false",
                    "include_video": "false",
                    "page": page,
                },
            )
            results = data.get("results", [])
            if not results:
                break
            for movie in results:
                movie_ids.append(movie["id"])
                if len(movie_ids) >= max_movies:
                    break
            if page >= data.get("total_pages", page):
                break
            page += 1
        return movie_ids[:max_movies]

    def get_movie_details(self, movie_id: int) -> dict:
        return self._get(f"/movie/{movie_id}")

    def get_movie_keywords(self, movie_id: int) -> list[dict]:
        data = self._get(f"/movie/{movie_id}/keywords")
        return data.get("keywords", [])

    def get_movie_credits(self, movie_id: int) -> dict:
        return self._get(f"/movie/{movie_id}/credits")


def movie_record_from_api(client: TMDBClient, movie_id: int) -> Optional[dict]:
    """
    Fetch one movie and return a row compatible with train.py / legacy CSV format.
    """
    try:
        details = client.get_movie_details(movie_id)
        keywords_raw = client.get_movie_keywords(movie_id)
        credits = client.get_movie_credits(movie_id)
    except requests.RequestException:
        return None

    title = (details.get("title") or "").strip()
    overview = (details.get("overview") or "").strip()
    if not title or not overview:
        return None

    genres = details.get("genres", [])
    keywords = [{"name": k.get("name", "")} for k in keywords_raw if k.get("name")]
    cast = [{"name": c.get("name", "")} for c in credits.get("cast", []) if c.get("name")]
    crew = [
        {"name": c.get("name", ""), "job": c.get("job", "")}
        for c in credits.get("crew", [])
        if c.get("name")
    ]

    genres_display = ", ".join(g["name"] for g in genres if g.get("name")) or None

    return {
        "movie_id": details.get("id", movie_id),
        "title": title,
        "overview": overview,
        "genres": json.dumps(genres),
        "keywords": json.dumps(keywords),
        "cast": json.dumps(cast),
        "crew": json.dumps(crew),
        "release_date": details.get("release_date") or "",
        "vote_average": details.get("vote_average"),
        "genres_display": genres_display,
        "overview_display": overview,
    }
