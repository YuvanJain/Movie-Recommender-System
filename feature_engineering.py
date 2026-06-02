"""
Feature engineering for content-based movie recommendations.
Builds weighted tag text and structured metadata sets for hybrid scoring.
"""

import ast
import re
from typing import Any

from nltk.stem.porter import PorterStemmer

# Repetition weights — higher = more influence in TF-IDF space
GENRE_WEIGHT = 6
KEYWORD_WEIGHT = 5
CAST_WEIGHT = 3
DIRECTOR_WEIGHT = 4
OVERVIEW_MAX_WORDS = 35

STEMMER = PorterStemmer()

# Common overview words that cause false matches across unrelated films
EXTRA_STOPWORDS = {
    "one", "two", "three", "find", "must", "life", "world", "film",
    "story", "young", "man", "woman", "new", "old", "time", "way",
    "day", "year", "help", "take", "make", "come", "go", "get",
    "see", "know", "want", "think", "look", "back", "first", "last",
    "never", "ever", "together", "against", "around", "through",
    "father", "mother", "son", "daughter", "family", "friend",
    "love", "death", "live", "die", "fight", "save", "discover",
}


def _safe_literal_list(value: Any) -> list:
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def _normalize_token(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    words = [STEMMER.stem(w) for w in text.split() if len(w) > 2]
    words = [w for w in words if w not in EXTRA_STOPWORDS]
    return " ".join(words)


def _extract_names(items: list, key: str = "name", limit: int | None = None) -> list[str]:
    names = []
    for item in items:
        if isinstance(item, dict):
            name = item.get(key) or item.get("name")
        else:
            name = str(item)
        if name:
            names.append(str(name).strip())
        if limit and len(names) >= limit:
            break
    return names


def parse_genres_raw(value: Any) -> list[str]:
    items = _safe_literal_list(value)
    return _extract_names(items)


def parse_keywords_raw(value: Any) -> list[str]:
    items = _safe_literal_list(value)
    return _extract_names(items)


def parse_cast_raw(value: Any, limit: int = 5) -> list[str]:
    items = _safe_literal_list(value)
    return _extract_names(items, limit=limit)


def parse_directors_raw(value: Any) -> list[str]:
    items = _safe_literal_list(value)
    directors = []
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            name = item.get("name")
            if name:
                directors.append(str(name).strip())
    return directors


def token_set_from_names(names: list[str], prefix: str) -> set[str]:
    tokens = set()
    for name in names:
        normalized = _normalize_token(name.replace(" ", ""))
        if normalized:
            tokens.add(f"{prefix}_{normalized}")
    return tokens


def build_feature_record(row) -> dict:
    """
    Build tag string and structured token sets from a raw movie row.
    """
    genres = parse_genres_raw(row.get("genres"))
    keywords = parse_keywords_raw(row.get("keywords"))
    cast = parse_cast_raw(row.get("cast"), limit=5)
    directors = parse_directors_raw(row.get("crew"))

    overview_text = str(row.get("overview", "") or "")
    overview_words = []
    seen = set()
    for word in _normalize_token(overview_text).split():
        if word not in seen:
            seen.add(word)
            overview_words.append(word)
        if len(overview_words) >= OVERVIEW_MAX_WORDS:
            break

    genre_tokens = token_set_from_names(genres, "genre")
    keyword_tokens = token_set_from_names(keywords, "keyword")
    cast_tokens = token_set_from_names(cast, "cast")
    director_tokens = token_set_from_names(directors, "director")

    weighted_parts = (
        overview_words
        + list(genre_tokens) * GENRE_WEIGHT
        + list(keyword_tokens) * KEYWORD_WEIGHT
        + list(cast_tokens) * CAST_WEIGHT
        + list(director_tokens) * DIRECTOR_WEIGHT
    )

    tags = " ".join(weighted_parts).lower()

    return {
        "tags": tags,
        "genre_set": set(genres),
        "keyword_set": set(keywords),
        "cast_set": set(cast),
        "director_set": set(directors),
    }
