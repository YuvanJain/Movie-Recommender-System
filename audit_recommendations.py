"""Temporary audit: print top-10 recommendations for benchmark titles."""
import pickle
import pandas as pd

md = pickle.load(open("model/movies_dict.pkl", "rb"))
movies = pd.DataFrame(md)
sim = pickle.load(open("model/similarity.pkl", "rb"))

TEST_TITLES = [
    "The Hangover",
    "The Hangover Part II",
    "Interstellar",
    "Batman Begins",
    "Toy Story",
    "Titanic",
]


def top10(title):
    match = movies[movies["title"] == title]
    if match.empty:
        print(f"MISSING: {title}")
        return
    pos = movies.index.get_loc(match.index[0])
    scores = sorted(list(enumerate(sim[pos])), key=lambda x: x[1], reverse=True)[1:11]
    print(f"\n=== {title} ===")
    src = movies.iloc[pos]
    print(f"Source genres: {src.get('genres_display', '?')}")
    for i, s in scores:
        row = movies.iloc[i]
        pct = round(float(s) * 100)
        genres = row.get("genres_display", "?")
        print(f"  {s:.4f} ({pct}%) | {row['title']} | {genres}")


if __name__ == "__main__":
    print(f"Model size: {len(movies)}, similarity: {sim.shape}")
    for t in TEST_TITLES:
        top10(t)
