import pandas as pd
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from movies.models import Movie, Rating

User = get_user_model()


class Command(BaseCommand):
    help = "Import MovieLens movies, users, and ratings from CSV files"

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            default=str(settings.BASE_DIR / "data"),
            help="Folder containing movies.csv and ratings.csv",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        data_dir = options["data_dir"]

        movies_df = pd.read_csv(f"{data_dir}/movies.csv")
        ratings_df = pd.read_csv(f"{data_dir}/ratings.csv")

        self.import_movies(movies_df)
        self.import_users(ratings_df)
        self.import_ratings(ratings_df)

        self.stdout.write(self.style.SUCCESS("Import finished."))

    def import_movies(self, df):
        existing = set(Movie.objects.values_list("movie_id", flat=True))
        new_movies = [
            Movie(movie_id=row.movieId, title=row.title, genres=row.genres)
            for row in df.itertuples()
            if row.movieId not in existing
        ]
        Movie.objects.bulk_create(new_movies, batch_size=1000)
        self.stdout.write(f"Movies added: {len(new_movies)}")

    def import_users(self, df):
        existing = set(
            User.objects.filter(username__startswith="ml_user_")
            .values_list("username", flat=True)
        )
        new_users = []
        for uid in df["userId"].unique():
            username = f"ml_user_{uid}"
            if username not in existing:
                user = User(username=username)
                user.set_unusable_password()
                new_users.append(user)
        User.objects.bulk_create(new_users, batch_size=1000)
        self.stdout.write(f"Users added: {len(new_users)}")

    def import_ratings(self, df):
        user_map = {
            int(u.username.replace("ml_user_", "")): u.id
            for u in User.objects.filter(username__startswith="ml_user_")
        }
        movie_map = dict(Movie.objects.values_list("movie_id", "id"))

        already = set(Rating.objects.values_list("user_id", "movie_id"))

        new_ratings = []
        for row in df.itertuples():
            user_pk = user_map[row.userId]
            movie_pk = movie_map.get(row.movieId)
            if movie_pk is None or (user_pk, movie_pk) in already:
                continue
            new_ratings.append(
                Rating(user_id=user_pk, movie_id=movie_pk, score=row.rating)
            )

        Rating.objects.bulk_create(new_ratings, batch_size=5000)
        self.stdout.write(f"Ratings added: {len(new_ratings)}")