# Movie Recommender — Audit & Quality Improvement Report

## Executive summary

Recommendation quality suffered from a **small TMDB-only catalog (~1,426 movies)**, **CountVectorizer bag-of-words** dominated by generic overview text, and **no genre/cast/keyword re-ranking**. The pipeline now uses a **combined dataset (~5,715 movies)**, **TF-IDF vectorization**, **weighted metadata tags**, and **hybrid scoring** while preserving the Streamlit UI.

---

## 1. Dataset pipeline audit

| Step | Row count | Notes |
|------|-----------|-------|
| Legacy CSV + credits merge | 4,809 | Full TMDB 5000 export |
| TMDB API CSV | 1,433 | `TMDB_MAX_MOVIES` popular fetch |
| After concat | 6,242 | Combined sources |
| After dedupe by title | 5,720 | Legacy row kept first (522 dupes removed) |
| After dropna (core fields) | 5,717 | 3 rows missing genres/keywords/cast/crew |
| After overview length filter | **5,715** | Final training set |

### Why only ~1,433 movies before?

`DATA_SOURCE=tmdb` used **only** `fetch_tmdb_data.py`, which:

1. Requests at most `TMDB_MAX_MOVIES` (default **1500**) from `/discover/movie`
2. Skips movies without overview/title during API fetch
3. Drops duplicate titles and `dropna` in fetch script
4. `train.py` `dropna` removed additional rows → **~1,426** in the old model

The legacy CSV has **~4,803** movies; combining both yields **~5,715** unique titles.

**Fix:** Default `DATA_SOURCE=combined` merges legacy + TMDB API catalogs.

---

## 2. Feature engineering audit

### Before (issues)

| Metadata | Used? | Problem |
|----------|-------|---------|
| Overview | Yes | Full text → dominated similarity |
| Genres | Yes (×3) | Underweighted vs long overview |
| Keywords | Yes | Same |
| Cast | Yes (top 3) | Same |
| Director | Yes (crew job filter) | Same |
| Processing | Porter stem only | No dedupe; generic overview words |

### After (`feature_engineering.py`)

- Overview capped to **35 unique stemmed words**; extra stopword list for plot clichés
- Prefixed tokens: `genre_comedy`, `keyword_hangover`, `cast_…`, `director_…`
- Weights: genres **×6**, keywords **×5**, cast **×3**, director **×4**
- Deduplicated tokens before vectorization

---

## 3. Vectorization audit

| Setting | Before | After |
|---------|--------|-------|
| Vectorizer | CountVectorizer | **TfidfVectorizer** |
| max_features | 5000 | **12000** (capped by corpus size) |
| ngram_range | (1,1) | **(1, 2)** |
| min_df | default | **2** |
| max_df | default | **0.92** |
| sublinear_tf | no | **yes** |

**TF-IDF** down-weights common terms (e.g. “life”, “world”) and improves genre/keyword signal vs raw counts.

---

## 4. Similarity matrix audit

| Check | Result |
|-------|--------|
| `len(movies)` | 5715 |
| `similarity.shape` | (5715, 5715) |
| `movie_metadata.pkl` length | 5715 |
| Alignment | **PASS** |
| Cosine range | 0.0 – 1.0 |

New artifacts: `model/vectorizer.pkl`, `model/movie_metadata.pkl`

---

## 5. Similarity scores in UI

- **Before:** Raw cosine on count vectors (often 30–47% for weak matches).
- **After:** **Hybrid score** = cosine × genre/keyword/cast/director boosts; **×0.2 penalty** if zero genre overlap; top match normalized to **99%** for readable relative ranking.
- Display: `Similarity: {0–99}%` from `format_similarity_score()` in `recommendation_model.py`.

---

## 6. Before vs after examples

### The Hangover Part II

**Before (1,426-movie model):** Mostly obscure comedies (Poppea's Hot Nights, Siegfried, foreign titles) at ~31–36%.

**After (5,715-movie hybrid):**

1. The Hangover (99%) — same franchise, cast, director, keywords  
2. Old School (57%) — Todd Phillips comedy  
3. Due Date (47%) — Zach Galifianakis + Todd Phillips  
4. Failure to Launch — shared cast  
5. 22 Jump Street — comedy + keyword overlap  

### Interstellar

**Before:** Guardians of the Galaxy, Lightyear (weak sci-fi matches).

**After:** Silent Running, Apollo 13, The Martian, 2001, Project Hail Mary — space/sci-fi metadata overlap.

### Batman Begins

**After:** The Dark Knight / Rises (99%), then related Batman/superhero titles.

### Titanic

**Before:** Obscure romance titles (GGS, regional films).

**After (expected):** The Notebook, similar romance/drama (re-run `python audit_pipeline.py` for full list).

---

## 7. Files modified

| File | Change |
|------|--------|
| `feature_engineering.py` | **New** — weighted tags + metadata sets |
| `recommendation_model.py` | **New** — TF-IDF training + hybrid `recommend()` |
| `train.py` | Combined dataset, step audit logs |
| `app.py` | Uses `MovieRecommender.load()` (UI unchanged) |
| `audit_pipeline.py` | **New** — full audit + top-10 explanations |
| `.env.example` | `DATA_SOURCE=combined` |
| `AUDIT_REPORT.md` | This report |

---

## 8. How to retrain & validate

```bash
# In .env
DATA_SOURCE=combined
TMDB_API_KEY=your_key

python train.py
python audit_pipeline.py
streamlit run app.py
```

---

## Root causes (concise)

1. **Dataset too small** when using TMDB-only fetch (~1.5k vs ~5.7k combined).  
2. **Overview text dominance** in bag-of-words vectors.  
3. **No structured re-ranking** — cosine-only matched generic words across genres.  
4. **CountVectorizer** amplified frequent plot words vs discriminative genres/keywords.  
5. **Misleading % scores** — low cosine on sparse vectors looked “broken” despite math being correct.
