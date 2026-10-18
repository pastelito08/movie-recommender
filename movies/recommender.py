from functools import lru_cache

import joblib
from django.conf import settings
from django.db.models import Avg, Count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Movie, Rating


# ---------------------------------------------------------------
# Part 1: Content-based "Similar movies" (TF-IDF + cosine similarity)
# ---------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_index():
    """Build the TF-IDF matrix once and cache it in memory."""
    rows = list(Movie.objects.values_list("movie_id", "genres"))
    movie_ids = [r[0] for r in rows]
    genres = ["" if g == "(no genres listed)" else g for _, g in rows]

    tfidf = TfidfVectorizer(
        tokenizer=lambda x: x.split("|"),
        token_pattern=None,
        lowercase=False,
    )
    matrix = tfidf.fit_transform(genres)
    id_to_row = {mid: i for i, mid in enumerate(movie_ids)}
    return movie_ids, matrix, id_to_row


@lru_cache(maxsize=1)
def _rating_counts():
    """Number of ratings per movie, used to break similarity ties."""
    return dict(
        Movie.objects.annotate(n=Count("ratings")).values_list("movie_id", "n")
    )


def similar_movie_ids(movie_id, n=8):
    movie_ids, matrix, id_to_row = _build_index()
    idx = id_to_row.get(movie_id)
    if idx is None:
        return []

    scores = cosine_similarity(matrix[idx], matrix).ravel()
    scores[idx] = -1  # exclude the movie itself

    counts = _rating_counts()
    candidates = [i for i in scores.argsort()[::-1][:200] if scores[i] > 0]
    candidates.sort(
        key=lambda i: (round(scores[i], 3), counts.get(movie_ids[i], 0)),
        reverse=True,
    )
    return [movie_ids[i] for i in candidates[:n]]


# ---------------------------------------------------------------
# Part 2: Personalized "Recommended for you" (SVD model)
# ---------------------------------------------------------------

ARTIFACTS_DIR = settings.BASE_DIR / "ml" / "artifacts"


def _latest_model_path():
    """Find the highest-numbered svd_vN.pkl file."""
    files = []
    for p in ARTIFACTS_DIR.glob("svd_v*.pkl"):
        num = p.stem.split("_v")[1]
        if num.isdigit():
            files.append((int(num), p))
    return max(files)[1] if files else None


@lru_cache(maxsize=1)
def _load_model():
    """Load the newest trained SVD model once. Returns None if none exists."""
    path = _latest_model_path()
    return joblib.load(path) if path else None


@lru_cache(maxsize=1)
def _movie_stats():
    """{movie_id: (number_of_ratings, average_score)}, cached."""
    rows = Movie.objects.annotate(
        n=Count("ratings"), avg=Avg("ratings__score")
    ).values_list("movie_id", "n", "avg")
    return {mid: (n, avg or 0.0) for mid, n, avg in rows}


def reload_model():
    """Clear caches so the next request loads the newest model and fresh stats."""
    _load_model.cache_clear()
    _movie_stats.cache_clear()
    _rating_counts.cache_clear()


def popular_movie_ids(n=12, exclude=(), min_ratings=30):
    """Popularity baseline using the IMDb weighted-rating formula."""
    stats = _movie_stats()
    total = sum(v for v, _ in stats.values())
    C = sum(v * a for v, a in stats.values()) / total  # global average
    m = min_ratings

    scored = []
    for mid, (v, R) in stats.items():
        if v >= m and mid not in exclude:
            scored.append(((v / (v + m)) * R + (m / (v + m)) * C, mid))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:n]]


def _model_knows_user(model, user_id):
    try:
        model.trainset.to_inner_uid(user_id)
        return True
    except ValueError:
        return False


def recommend_for_user(user, n=12, min_ratings=30):
    """Returns (list_of_movie_ids, source) where source is 'svd' or 'popular'."""
    rated = set(
        Rating.objects.filter(user=user).values_list("movie__movie_id", flat=True)
    )
    model = _load_model()

    # Fallback: no model file, or the model has never seen this user (cold start)
    if model is None or not rated or not _model_knows_user(model, user.id):
        return popular_movie_ids(n, exclude=rated, min_ratings=min_ratings), "popular"

    stats = _movie_stats()
    candidates = [
        mid for mid, (v, _) in stats.items()
        if v >= min_ratings and mid not in rated
    ]
    scored = [(model.predict(user.id, mid).est, mid) for mid in candidates]
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:n]], "svd"