from django.contrib import admin
from .models import Movie, Rating


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ("movie_id", "title", "genres")
    search_fields = ("title",)


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("user", "movie", "score", "created_at")
    list_filter = ("score",)