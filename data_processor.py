"""
Data Processing Utilities for Movie Recommendation System
"""

import pandas as pd


class DataProcessor:
    """Handles data cleaning for movies and ratings."""

    def clean_movie_data(self, movies_df):
        """Clean and preprocess movie data."""
        df = movies_df.copy()
        df["genres"] = df["genres"].fillna("Unknown")
        df["title"] = df["title"].fillna("Unknown")
        df["year"] = df["title"].str.extract(r"\((\d{4})\)")
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True)
        return df

    def clean_ratings_data(self, ratings_df):
        """Clean and preprocess ratings data."""
        df = ratings_df.copy()
        df = df.drop_duplicates(subset=["userId", "movieId"])
        df = df[(df["rating"] >= 0.5) & (df["rating"] <= 5.0)]
        user_rating_counts = df["userId"].value_counts()
        valid_users = user_rating_counts[user_rating_counts >= 5].index
        df = df[df["userId"].isin(valid_users)]
        movie_rating_counts = df["movieId"].value_counts()
        valid_movies = movie_rating_counts[movie_rating_counts >= 5].index
        df = df[df["movieId"].isin(valid_movies)]
        return df
