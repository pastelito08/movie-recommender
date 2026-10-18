import time

import joblib
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from movies.models import Rating


class Command(BaseCommand):
    help = "Train a new SVD model from the ratings in the database"

    def add_arguments(self, parser):
        parser.add_argument("--factors", type=int, default=50)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--lr", type=float, default=0.005)
        parser.add_argument("--reg", type=float, default=0.02)

    def handle(self, *args, **options):
        start = time.time()

        # 1. Export ratings from the database
        rows = Rating.objects.values_list("user_id", "movie__movie_id", "score")
        df = pd.DataFrame(list(rows), columns=["userId", "movieId", "rating"])
        self.stdout.write(f"Loaded {len(df)} ratings from {df.userId.nunique()} users")

        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(df, reader)

        # 2. Quick quality check on a hold-out split
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        params = dict(
            n_factors=options["factors"],
            n_epochs=options["epochs"],
            lr_all=options["lr"],
            reg_all=options["reg"],
            random_state=42,
        )
        check = SVD(**params).fit(trainset)
        rmse = accuracy.rmse(check.test(testset), verbose=False)
        self.stdout.write(f"Hold-out RMSE: {rmse:.4f}")

        # 3. Train the final model on ALL data
        model = SVD(**params).fit(data.build_full_trainset())

        # 4. Save as the next version
        artifacts = settings.BASE_DIR / "ml" / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        versions = [
            int(p.stem.split("_v")[1])
            for p in artifacts.glob("svd_v*.pkl")
            if p.stem.split("_v")[1].isdigit()
        ]
        new_version = max(versions, default=0) + 1
        path = artifacts / f"svd_v{new_version}.pkl"
        joblib.dump(model, path)

        self.stdout.write(self.style.SUCCESS(
            f"Saved {path.name} (RMSE {rmse:.4f}, {time.time() - start:.0f}s)"
        ))