# Movie Recommender (Django + Machine Learning)

A movie recommendation website built with **Django** and **scikit-learn / Surprise**. Users browse movies, rate them, and get two kinds of recommendations:

- **Similar movies** on every movie page (content-based: TF-IDF + cosine similarity)
- **Recommended for you** (personalized: SVD matrix factorization trained on user ratings)

New users with no ratings get a **popularity fallback**, and a management command **retrains the model** as new ratings arrive.

---

## Table of contents

1. [Features](#features)
2. [Tech stack](#tech-stack)
3. [How it works](#how-it-works)
4. [Project structure](#project-structure)
5. [Database design](#database-design)
6. [Setup and installation](#setup-and-installation)
7. [Usage](#usage)
8. [URL reference](#url-reference)
9. [Management commands](#management-commands)
10. [The ML notebooks](#the-ml-notebooks)
11. [ML concepts explained](#ml-concepts-explained)
12. [Model evaluation](#model-evaluation)
13. [Design decisions](#design-decisions)
14. [Troubleshooting](#troubleshooting)
15. [Future improvements](#future-improvements)
16. [Dataset and credits](#dataset-and-credits)

---

## Features

| Feature | Description |
|---|---|
| Movie catalogue | ~9,700 movies with search and pagination (24 per page) |
| Ratings | Logged-in users rate movies from 0.5 to 5.0 stars; re-rating updates the existing row |
| Authentication | Signup, login, logout using Django's built-in auth |
| Similar movies | Content-based recommendations from genres (TF-IDF + cosine similarity) |
| Recommended for you | Personalized recommendations from an SVD model |
| Cold-start fallback | Popular movies (IMDb weighted rating) for users the model does not know |
| Model versioning | Each training run saves `svd_v1.pkl`, `svd_v2.pkl`, and so on |
| Retraining command | `python manage.py train_model` rebuilds the model from the database |
| Admin panel | Inspect movies, users, and ratings at `/admin/` |

---

## Tech stack

| Area | Tools |
|---|---|
| Backend | Python 3.12, Django, Django REST Framework (installed for future API use) |
| Database | SQLite (development), easy to swap for PostgreSQL |
| ML | scikit-learn (TF-IDF, cosine similarity), scikit-surprise (SVD), NumPy, Pandas |
| Model storage | joblib (`.pkl` files) |
| Experiments | Jupyter notebooks, Matplotlib, Seaborn |
| Frontend | Django templates with plain HTML and CSS |

---

## How it works

### Big picture

```
                 ┌───────────────────────────┐
 MovieLens CSVs →│ import_movielens command  │→ Movie, User, Rating tables
                 └───────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                    ▼
   Content-based (genres)                Collaborative (ratings)
   TF-IDF → cosine similarity            train_model → SVD → svd_vN.pkl
              │                                    │
              ▼                                    ▼
   "Similar movies" on movie page        "Recommended for you" page
                                         (popularity fallback for new users)
```

### 1. Data layer

`import_movielens` reads `movies.csv` and `ratings.csv` and creates:

- `Movie` rows (with the original MovieLens ID kept in `movie_id`)
- 610 placeholder users named `ml_user_1` to `ml_user_610` (unusable passwords, they only hold training data)
- `Rating` rows linking users to movies

### 2. Similar movies (content-based)

File: `movies/recommender.py`, function `similar_movie_ids`.

1. Each movie's genres (`Action|Sci-Fi`) are converted into a numeric vector with **TF-IDF**. Rare genres get higher weights than common ones.
2. **Cosine similarity** compares the chosen movie's vector with every other movie.
3. The top results are returned. Ties (many movies share identical genres) are broken by popularity, so well-known titles rank first.
4. The TF-IDF matrix is built once and cached with `lru_cache`.

### 3. Recommended for you (collaborative filtering)

File: `movies/recommender.py`, function `recommend_for_user`.

1. The newest `ml/artifacts/svd_vN.pkl` is loaded once and cached.
2. For the logged-in user, the model predicts a rating for every movie they have not rated.
3. Only movies with at least 30 ratings are considered, to avoid unreliable obscure picks.
4. The 12 highest predicted movies are shown.

### 4. Cold-start fallback

If the model file is missing, the user has no ratings, or the user did not exist when the model was trained, the site shows **popular movies** ranked with the IMDb weighted-rating formula:

```
score = (v / (v + m)) × R + (m / (v + m)) × C
```

`R` = movie average, `v` = its number of ratings, `m` = minimum ratings required, `C` = global average rating.

### 5. Retraining

`python manage.py train_model` exports all ratings from the database, evaluates on a hold-out split, trains a final model on all data, and saves it as the next version. The recommender automatically picks the highest version number.

---

## Project structure

```
movie_recommender/
├── manage.py                     # Django command runner
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore
├── db.sqlite3                    # SQLite database (not committed)
│
├── config/                       # Project configuration
│   ├── settings.py               # Apps, database, templates, login redirects
│   ├── urls.py                   # Root URL router
│   ├── asgi.py / wsgi.py         # Server entry points
│
├── accounts/                     # Signup logic
│   ├── views.py                  # signup view
│   └── urls.py
│
├── movies/                       # Main app
│   ├── models.py                 # Movie and Rating tables
│   ├── views.py                  # movie_list, movie_detail, rate_movie, recommendations
│   ├── urls.py
│   ├── admin.py
│   ├── recommender.py            # ALL ML serving logic (similar + personalized)
│   └── management/commands/
│       ├── import_movielens.py   # Loads the dataset into the database
│       └── train_model.py        # Trains and versions the SVD model
│
├── templates/                    # HTML only
│   ├── base.html                 # Shared layout and navbar
│   ├── registration/
│   │   ├── login.html
│   │   └── signup.html
│   └── movies/
│       ├── movie_list.html
│       ├── movie_detail.html
│       └── recommendations.html
│
├── data/                         # MovieLens CSVs (not committed)
│   ├── movies.csv
│   └── ratings.csv
│
└── ml/
    ├── notebooks/                # Experiments (training and evaluation)
    │   ├── 01_explore_data.ipynb
    │   ├── 02_content_based.ipynb
    │   └── 03_collaborative.ipynb
    └── artifacts/                # Saved models (not committed)
        └── svd_v1.pkl
```

**Rule of thumb:** training happens in notebooks or the `train_model` command. The web app only *loads* saved models and predicts.

---

## Database design

### `Movie`

| Field | Type | Notes |
|---|---|---|
| `movie_id` | Integer, unique | Original MovieLens ID, used by the ML model |
| `title` | Char(255) | e.g. `Toy Story (1995)` |
| `genres` | Char(255) | Pipe-separated, e.g. `Adventure\|Animation\|Children` |

Helper property `genre_list` splits the genres into a Python list for templates (no database column).

### `Rating`

| Field | Type | Notes |
|---|---|---|
| `user` | ForeignKey to User | Cascade delete |
| `movie` | ForeignKey to Movie | Cascade delete |
| `score` | Float | Validated between 0.5 and 5.0 |
| `created_at` | DateTime | Auto-set, useful for time-based splits |

`unique_together = (user, movie)` guarantees one rating per user per movie, which keeps the training data clean.

### ID mapping (important)

The SVD model is trained with `userId = Django user id` and `movieId = MovieLens movie_id`. At prediction time the code calls `model.predict(user.id, movie.movie_id)`. Keeping these identifiers consistent is what makes the model usable from Django.

---

## Setup and installation

### Prerequisites

- Python 3.10 or newer (developed on 3.12)
- pip
- Git (optional)

### Steps

**1. Clone and enter the project**

```bash
git clone <your-repo-url>
cd movie_recommender
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

If `scikit-surprise` fails to build on Windows, install Microsoft C++ Build Tools, or use `pip install "numpy<2"` first and retry. If it still fails, see [Troubleshooting](#troubleshooting).

**4. Download the dataset**

Download `ml-latest-small.zip` from <https://grouplens.org/datasets/movielens/latest/>, unzip it, create a `data/` folder next to `manage.py`, and copy `movies.csv` and `ratings.csv` into it.

**5. Create the database**

```bash
python manage.py makemigrations
python manage.py migrate
```

**6. Import the data**

```bash
python manage.py import_movielens
```

Expected output:

```
Movies added: 9742
Users added: 610
Ratings added: 100836
Import finished.
```

**7. Create an admin user**

```bash
python manage.py createsuperuser
```

**8. Train the first model**

```bash
python manage.py train_model
```

This creates `ml/artifacts/svd_v1.pkl`.

**9. Run the server**

```bash
python manage.py runserver
```

Open <http://127.0.0.1:8000/>.

---

## Usage

1. **Browse and search** movies on the home page.
2. **Sign up** for an account (the imported `ml_user_N` accounts cannot log in).
3. **Open a movie** to see its average rating, a rating form, and *Similar movies*.
4. **Rate 15 to 20 movies** across different genres.
5. **Retrain** so the model learns your taste:
   ```bash
   python manage.py train_model
   ```
6. **Restart the server**, then click **For you** in the navbar to see personalized picks.

A brand-new account that has not been included in training will see the popular-movies fallback, which is the expected cold-start behavior.

---

## URL reference

| URL | Name | Description | Login needed |
|---|---|---|---|
| `/` | `movie_list` | Movie grid with search (`?q=`) and pages (`?page=`) | No |
| `/movie/<id>/` | `movie_detail` | One movie, stats, rating form, similar movies | No |
| `/movie/<id>/rate/` | `rate_movie` | Saves a rating (POST only) | Yes |
| `/recommendations/` | `recommendations` | Personalized or popular picks | Yes |
| `/accounts/signup/` | `signup` | Create an account | No |
| `/accounts/login/` | `login` | Log in | No |
| `/accounts/logout/` | `logout` | Log out (POST) | Yes |
| `/admin/` | - | Django admin | Staff |

`<id>` is the MovieLens `movie_id`, not Django's internal primary key.

---

## Management commands

### `import_movielens`

```bash
python manage.py import_movielens
python manage.py import_movielens --data-dir path/to/other/folder
```

Reads the CSVs and fills the database. It is safe to run twice: existing movies, users, and ratings are skipped. The whole import runs in one transaction, so a failure rolls everything back.

### `train_model`

```bash
python manage.py train_model
python manage.py train_model --factors 100 --epochs 30 --lr 0.005 --reg 0.1
```

| Option | Default | Meaning |
|---|---|---|
| `--factors` | 50 | Number of hidden taste factors per user and movie |
| `--epochs` | 20 | Passes over the training data |
| `--lr` | 0.005 | Learning rate for gradient descent |
| `--reg` | 0.02 | Regularization strength (prevents overfitting) |

Steps performed:

1. Export every rating from the database
2. Train on 80% and report **hold-out RMSE** on 20%
3. Retrain on 100% of the data
4. Save as the next version (`svd_v2.pkl`, `svd_v3.pkl`, ...); old versions are kept so you can roll back

After training, restart the server (or call `reload_model()`) so the new model is loaded.

---

## The ML notebooks

| Notebook | What it covers |
|---|---|
| `01_explore_data.ipynb` | Sparsity, rating distribution, power users, long tail, raw vs weighted top movies, genres, popularity baseline |
| `02_content_based.ipynb` | One-hot vs TF-IDF vectors, cosine similarity, `similar_movies()` function, popularity tie-breaker |
| `03_collaborative.ipynb` | User-item matrix, item-based collaborative filtering, baseline and SVD with Surprise, minimum-ratings filter, Precision@K / Recall@K, hyperparameter search, saving the model |

Run notebook cells top to bottom. Select the Python kernel from your project's virtual environment so all packages are available. Never run a `.ipynb` file with `python`.

---

## ML concepts explained

### Sparsity
Only about 1.7% of the possible user-movie pairs have a rating, so about 98.3% of the matrix is empty. Predicting those blanks is the core problem.

### Vectors
A computer cannot compare text like `Action|Sci-Fi`. It can compare numbers, so each movie is turned into a vector, with one position per genre.

### TF-IDF
Term Frequency times Inverse Document Frequency. It weights each genre by how rare it is.

```
IDF = ln((1 + n) / (1 + df)) + 1
```

`n` is the number of movies and `df` is the number of movies containing the genre. Rare genres (Film-Noir) get large weights, common ones (Drama) get small weights. Vectors are then normalized to length 1.

### Cosine similarity
Measures the angle between two vectors.

```
cos(A, B) = (A · B) / (|A| × |B|)
```

`1.0` means identical direction, `0.0` means nothing in common. Example: Jurassic Park vs The Lost World scores 1.0 (same genres), Jurassic Park vs Toy Story scores about 0.22 (only Adventure shared).

TF-IDF represents movies as numbers, and cosine similarity compares them. They are used one after the other.

### Content-based vs collaborative filtering

| | Content-based | Collaborative |
|---|---|---|
| Looks at | Movie features (genres) | Other users' ratings |
| Used for | Similar movies | Recommended for you |
| Needs ratings? | No | Yes |
| Works for new movies? | Yes | No |

### SVD matrix factorization
Splits the huge, mostly empty ratings matrix into two small dense ones:

```
Ratings (users × movies) ≈ User factors (users × k) × Movie factors (k × movies)
```

Each user and movie gets `k` hidden factors (50 here). The predicted rating is:

```
global_average + user_bias + movie_bias + (user_vector · movie_vector)
```

Training uses gradient descent on the known ratings only. The biases absorb the fact that some users rate high and some movies are loved by everyone, so the factors can focus on real taste.

### Cold start
A new user (no ratings) or a new movie (no ratings) gives the model nothing to learn from. This project falls back to popular movies for new users, and content-based similarity works for new movies.

### Long tail
A few movies have hundreds of ratings, thousands have fewer than five. Averages from very few ratings are unreliable, so the recommender requires at least 30 ratings and uses a weighted rating for popularity.

### Baseline
A simple model that any real model must beat. Here: popularity ranking and `BaselineOnly` (global average plus user and movie biases).

---

## Model evaluation

| Metric | Meaning | Better |
|---|---|---|
| **RMSE** | Average rating error in stars: `√mean((predicted − actual)²)` | Lower |
| **MAE** | Average of `|predicted − actual|` | Lower |
| **Precision@K** | Of the K recommended movies, the share the user liked (rating ≥ 3.5) | Higher |
| **Recall@K** | Of all movies the user liked, the share placed in the top K | Higher |

RMSE checks that predictions are sensible. Precision@K reflects what users actually see (a top-10 list), so it is the more relevant metric for a recommender.

### Results (fill in with your own numbers)

| Model | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| Baseline (biases only) | _your value_ | _your value_ | _your value_ |
| SVD (default: 50 factors) | _your value_ | _your value_ | _your value_ |
| SVD (tuned with GridSearchCV) | _your value_ | - | - |

Typical values on MovieLens small: baseline RMSE about 0.87, SVD RMSE about 0.85 to 0.87.

### Limitations of the evaluation

- Metrics are **offline**: they only score movies the user already rated.
- The split is random, not time-based, which is more optimistic than real life.
- Real products confirm quality with **online** tests such as click-through rate and A/B testing.

---

## Design decisions

| Decision | Reason |
|---|---|
| Train outside Django, serve inside | Training is slow and heavy; serving should be fast |
| `lru_cache` for model and TF-IDF matrix | Load once instead of on every request |
| Versioned model files | Allows rollback and comparison over time |
| Popularity fallback | Prevents errors and meaningless output for new users |
| `min_ratings=30` filter | Hides unreliable, obscure movies |
| `unique_together` on Rating | One rating per user per movie keeps training data clean |
| Server-side validation of scores | Never trust browser input |
| POST then redirect after rating | Prevents duplicate submissions on refresh |
| Placeholder `ml_user_N` accounts | Lets the original dataset live in Django's normal tables |

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `Unknown command: 'import_movielens'` or `'train_model'` | File not in `movies/management/commands/`, or missing `__init__.py` in `management/` and `commands/` |
| `TemplateDoesNotExist` | `templates/` must be next to `manage.py`, and `'DIRS': [BASE_DIR / 'templates']` set in `settings.py` |
| `NoReverseMatch` | A URL name used in a template or redirect is missing from `urls.py` |
| `ModuleNotFoundError: sklearn` or `surprise` | Package not installed in the active virtual environment |
| `scikit-surprise` will not install | Install C++ build tools, try `pip install "numpy<2"`, or replace Surprise with a NumPy SVD implementation |
| Signup keeps reloading the form | Password rejected (too short, too common, or similar to username). Read the red message under the field |
| Always see the popular fallback | `ml/artifacts/svd_v*.pkl` missing, or you rated movies after the last training. Run `train_model` and restart the server |
| Recommendations do not change after rating | The model is frozen until retrained. Run `train_model` and restart the server |
| `FileNotFoundError` in a notebook | Notebook paths are relative to `ml/notebooks/` (`../../data/ratings.csv`) |
| `NameError` in a notebook | A previous cell was skipped. Run cells in order |
| `no such table` in a notebook | Wrong path to `db.sqlite3`, or migrations not applied |

---

## Future improvements

- **Hybrid model:** blend SVD and content-based scores
- **"Why recommended?"** explanations ("Because you liked Toy Story")
- **Celery + Redis** for scheduled nightly retraining
- **Tags and release year** as extra content features
- **Sentence embeddings** of plot descriptions for richer similarity
- **Neural collaborative filtering** with PyTorch
- **Implicit feedback** (clicks, watch time) with ALS
- **Time-based train/test split** and more metrics (NDCG, coverage, diversity)
- **PostgreSQL, Docker, and cloud deployment**
- **REST API** endpoints with Django REST Framework
- **Tests** for views, the recommender, and the retraining command

---

## Dataset and credits

This project uses the **MovieLens** dataset (`ml-latest-small`) from GroupLens Research, University of Minnesota.

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1 to 19:19.

MovieLens data is provided for research and education. Do not redistribute the dataset in your repository; instruct users to download it from GroupLens (as in the setup steps above). Check the dataset's README for its full usage terms.

Libraries: Django, scikit-learn, scikit-surprise, Pandas, NumPy, Matplotlib, Seaborn, joblib.

---

## License

Add a license of your choice (for example MIT) before publishing. The MovieLens data has its own terms and is not covered by your project license.