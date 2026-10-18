from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST
from .recommender import recommend_for_user, similar_movie_ids

from .models import Movie, Rating
from .recommender import similar_movie_ids

SCORE_CHOICES = [x / 2 for x in range(1, 11)]  # 0.5, 1.0, ... 5.0


def movie_list(request):
    query = request.GET.get("q", "").strip()

    movies = Movie.objects.annotate(
        avg_score=Avg("ratings__score"),
        num_ratings=Count("ratings"),
    ).order_by("-num_ratings", "title")

    if query:
        movies = movies.filter(title__icontains=query)

    paginator = Paginator(movies, 24)
    page_obj = paginator.get_page(request.GET.get("page"))

    return render(request, "movies/movie_list.html", {
        "page_obj": page_obj,
        "query": query,
    })


def movie_detail(request, movie_id):
    movie = get_object_or_404(Movie, movie_id=movie_id)
    stats = movie.ratings.aggregate(avg=Avg("score"), count=Count("id"))

    user_rating = None
    if request.user.is_authenticated:
        user_rating = Rating.objects.filter(user=request.user, movie=movie).first()

    # Similar movies (content-based: TF-IDF + cosine similarity)
    similar_ids = similar_movie_ids(movie.movie_id, n=8)
    similar_qs = Movie.objects.filter(movie_id__in=similar_ids)
    order = {mid: i for i, mid in enumerate(similar_ids)}
    similar = sorted(similar_qs, key=lambda m: order[m.movie_id])

    return render(request, "movies/movie_detail.html", {
        "movie": movie,
        "stats": stats,
        "user_rating": user_rating,
        "score_choices": SCORE_CHOICES,
        "similar": similar,
    })


@login_required
@require_POST
def rate_movie(request, movie_id):
    movie = get_object_or_404(Movie, movie_id=movie_id)

    try:
        score = float(request.POST.get("score"))
    except (TypeError, ValueError):
        messages.error(request, "Please choose a rating.")
        return redirect("movie_detail", movie_id=movie_id)

    if score not in SCORE_CHOICES:
        messages.error(request, "Invalid rating value.")
        return redirect("movie_detail", movie_id=movie_id)

    Rating.objects.update_or_create(
        user=request.user,
        movie=movie,
        defaults={"score": score},
    )
    messages.success(request, f"You rated {movie.title} {score} stars.")
    return redirect("movie_detail", movie_id=movie_id)

@login_required
def recommendations(request):
    ids, source = recommend_for_user(request.user, n=12)
    qs = Movie.objects.filter(movie_id__in=ids)
    order = {mid: i for i, mid in enumerate(ids)}
    movies = sorted(qs, key=lambda m: order[m.movie_id])

    return render(request, "movies/recommendations.html", {
        "movies": movies,
        "source": source,
    })