# Movie Recommendation System

A comprehensive machine learning project implementing a movie recommendation system using multiple approaches: Collaborative Filtering, Content-Based Filtering, and Hybrid methods.

## Features

- **Collaborative Filtering**: Uses Non-Negative Matrix Factorization (NMF) to find patterns in user-movie ratings
- **Content-Based Filtering**: Recommends movies based on genre similarity using TF-IDF vectorization
- **Hybrid Approach**: Combines both methods for improved recommendations
- **Streamlit web app**: Dark-themed UI for recommendations and search
- **CLI**: Command-line training and interactive mode

## Project Structure

```
Movie/
├── app.py                      # Streamlit app (entrypoint for deploy)
├── main.py                     # Train model + CLI interactive mode
├── recommendation_engine.py    # Recommendation engine
├── data_processor.py           # Data cleaning
├── generate_dataset.py         # Generate sample data from catalog
├── dataset/
│   └── movie_catalog.csv       # Movie catalog (required for deploy)
├── .streamlit/
│   └── config.toml            # Theme (e.g. slider color)
├── requirements.txt
├── DEPLOY.md                   # Streamlit Community Cloud deploy steps
├── data/                       # Generated or added here
│   ├── movies.csv
│   └── ratings.csv
└── models/                     # Trained model (optional; app can train)
    └── recommendation_model.pkl
```

## Installation

1. **Clone the repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data** (choose one):
   - **Option A:** Generate sample data: `python generate_dataset.py` (uses `dataset/movie_catalog.csv`)
   - **Option B:** Use your own `data/movies.csv` and `data/ratings.csv` (MovieLens format)

## Usage

### Web app (Streamlit)

```bash
streamlit run app.py
```

Then open the URL (e.g. http://localhost:8501). If `data/` is missing, the app can generate it from `dataset/movie_catalog.csv` on first run.

### Deploy on Streamlit Community Cloud

Push the repo to GitHub, then at [share.streamlit.io](https://share.streamlit.io/) create a new app with **Main file path:** `app.py`. See **[DEPLOY.md](DEPLOY.md)** for step-by-step instructions.

### Training the model (CLI)

Run the main script to train and save the model:

```bash
python main.py
```

This will:
- Load and preprocess the data
- Train the collaborative filtering model
- Save the trained model
- Display sample recommendations

### Interactive Mode

For interactive recommendations:

```bash
python main.py --interactive
```

In interactive mode, you can:
- Get recommendations for specific users
- Find similar movies
- Get hybrid recommendations
- Search for movies by title

### Using the API Programmatically

```python
from recommendation_engine import MovieRecommendationEngine

# Initialize engine
engine = MovieRecommendationEngine()

# Load data
engine.load_data('data/movies.csv', 'data/ratings.csv')

# Preprocess
engine.preprocess_data()

# Train model
engine.train_collaborative_filtering(n_components=50)

# Get recommendations
recommendations = engine.get_collaborative_recommendations(user_id=1, n_recommendations=10)
print(recommendations)

# Content-based recommendations
similar_movies = engine.get_content_based_recommendations(movie_id=1, n_recommendations=10)
print(similar_movies)

# Hybrid recommendations
hybrid = engine.get_hybrid_recommendations(user_id=1, n_recommendations=10)
print(hybrid)
```

## Algorithms Explained

### Collaborative Filtering (NMF)

Non-Negative Matrix Factorization decomposes the user-movie rating matrix into two lower-dimensional matrices:
- User features matrix
- Movie features matrix

The model learns latent factors that represent user preferences and movie characteristics. Predictions are made by multiplying these matrices.

### Content-Based Filtering

Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize movie genres. Cosine similarity is then used to find movies with similar genre profiles.

### Hybrid Approach

Combines collaborative and content-based scores with weighted averaging:
```
hybrid_score = α × collaborative_score + (1-α) × content_score
```

Default weight: α = 0.7 (70% collaborative, 30% content-based)

## Data Format

### movies.csv
Expected columns:
- `movieId`: Unique movie identifier
- `title`: Movie title (with year)
- `genres`: Pipe-separated list of genres (e.g., "Action|Adventure|Sci-Fi")

### ratings.csv
Expected columns:
- `userId`: Unique user identifier
- `movieId`: Movie identifier
- `rating`: Rating value (typically 0.5-5.0)
- `timestamp`: (Optional) Rating timestamp

## Customization

### Adjust Model Parameters

In `main.py`, modify the training parameters:

```python
engine.train_collaborative_filtering(
    n_components=50,    # Number of latent factors (increase for more complexity)
    max_iter=200        # Maximum iterations
)
```

### Change Hybrid Weights

```python
recommendations = engine.get_hybrid_recommendations(
    user_id=1,
    collaborative_weight=0.7  # 70% collaborative, 30% content-based
)
```

## Performance Tips

- For large datasets, consider reducing `n_components` or using a subset of data
- Increase `max_iter` for better convergence (but slower training)
- Filter out users/movies with too few ratings to improve quality

## Future Enhancements

Potential improvements:
- Deep learning models (Neural Collaborative Filtering)
- Real-time recommendation updates
- Web interface using Flask/FastAPI
- Integration with movie databases (TMDB, IMDb)
- User-based collaborative filtering
- Matrix factorization with bias terms

## License

This project is provided as-is for educational purposes.

## References

- MovieLens Dataset: https://grouplens.org/datasets/movielens/
- Scikit-learn Documentation: https://scikit-learn.org/
- Recommender Systems Handbook: Ricci, F., Rokach, L., & Shapira, B. (2015)
