from django.conf import settings
from django.core.validators import MinValueValidator, MaxValueValidator
from django.db import models


class Movie(models.Model):
    movie_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255, blank=True)

    def __str__(self):
        return self.title
    
    @property
    def genre_list(self):
        return self.genres.split("|") if self.genres else []


class Rating(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ratings",
    )
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        related_name="ratings",
    )
    score = models.FloatField(
        validators=[MinValueValidator(0.5), MaxValueValidator(5.0)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "movie")

    def __str__(self):
        return f"{self.user} rated {self.movie} {self.score}"