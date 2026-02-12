"""
Movie Recommendation Engine
Implements both Collaborative Filtering and Content-Based Filtering
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pickle
import os


class MovieRecommendationEngine:
    """Main recommendation engine combining multiple approaches"""
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.model = None
        self.movie_features = None
        self.tfidf_vectorizer = None
        self.content_similarity = None
        
    def load_data(self, movies_path='data/movies.csv', ratings_path='data/ratings.csv'):
        """Load movie and rating data"""
        try:
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)
            print(f"Loaded {len(self.movies_df)} movies and {len(self.ratings_df)} ratings")
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure data files exist in the 'data' directory")
            return False
    
    def preprocess_data(self):
        """Preprocess and prepare data for recommendation"""
        if self.movies_df is None or self.ratings_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create user-movie rating matrix
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            fill_value=0
        )
        
        print(f"Created user-movie matrix: {self.user_movie_matrix.shape}")
        
        # Prepare content-based features
        self._prepare_content_features()
        
    def _prepare_content_features(self):
        """Prepare content-based features from movie metadata"""
        # Combine genres into a single string
        self.movies_df['genres_combined'] = self.movies_df['genres'].fillna('')
        
        # Create TF-IDF features from genres
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        genre_features = self.tfidf_vectorizer.fit_transform(self.movies_df['genres_combined'])
        
        # Calculate cosine similarity between movies
        self.content_similarity = cosine_similarity(genre_features)
        
        print("Content-based features prepared")
    
    def train_collaborative_filtering(self, n_components=50, max_iter=200):
        """Train Non-Negative Matrix Factorization model for collaborative filtering"""
        if self.user_movie_matrix is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        # Convert to numpy array
        R = self.user_movie_matrix.values
        
        # Initialize and train NMF model
        self.model = NMF(n_components=n_components, max_iter=max_iter, random_state=42)
        self.movie_features = self.model.fit_transform(R)
        
        print(f"Collaborative filtering model trained with {n_components} components")
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using collaborative filtering"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_collaborative_filtering() first.")
        
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found. Returning popular movies instead.")
            return self._get_popular_movies(n_recommendations)
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id].values
        
        # Predict ratings for all movies
        user_features = self.model.transform(user_ratings.reshape(1, -1))
        predicted_ratings = np.dot(user_features, self.model.components_)
        
        # Get top recommendations (excluding already rated movies)
        user_rated_movies = set(self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values)
        movie_ids = self.user_movie_matrix.columns.values
        
        # Create recommendations dataframe
        recommendations = pd.DataFrame({
            'movieId': movie_ids,
            'predicted_rating': predicted_ratings[0]
        })
        
        # Filter out already rated movies
        recommendations = recommendations[~recommendations['movieId'].isin(user_rated_movies)]
        
        # Sort by predicted rating and get top N
        top_recommendations = recommendations.nlargest(n_recommendations, 'predicted_rating')
        
        # Merge with movie details
        result = top_recommendations.merge(self.movies_df, on='movieId', how='left')
        
        return result[['movieId', 'title', 'genres', 'predicted_rating']]
    
    def get_content_based_recommendations(self, movie_id, n_recommendations=10):
        """Get recommendations based on movie content similarity"""
        if self.content_similarity is None:
            raise ValueError("Content features not prepared. Call preprocess_data() first.")
        
        if movie_id not in self.movies_df['movieId'].values:
            print(f"Movie {movie_id} not found.")
            return None
        
        # Get index of the movie
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.content_similarity[movie_idx]))
        
        # Sort by similarity (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movies (excluding the movie itself)
        top_similar = similarity_scores[1:n_recommendations+1]
        
        # Extract movie IDs and similarity scores
        recommended_movies = []
        for idx, score in top_similar:
            movie = self.movies_df.iloc[idx]
            recommended_movies.append({
                'movieId': movie['movieId'],
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': score
            })
        
        return pd.DataFrame(recommended_movies)
    
    def get_hybrid_recommendations(self, user_id, movie_id=None, n_recommendations=10, 
                                   collaborative_weight=0.7):
        """Get hybrid recommendations combining collaborative and content-based filtering"""
        collab_recs = None
        content_recs = None
        
        # Get collaborative recommendations
        try:
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        except Exception as e:
            print(f"Error getting collaborative recommendations: {e}")
        
        # Get content-based recommendations if movie_id provided
        if movie_id:
            try:
                content_recs = self.get_content_based_recommendations(movie_id, n_recommendations * 2)
            except Exception as e:
                print(f"Error getting content-based recommendations: {e}")
        
        # Combine recommendations
        if collab_recs is not None and content_recs is not None:
            # Merge and score
            combined = collab_recs.merge(
                content_recs[['movieId', 'similarity_score']],
                on='movieId',
                how='outer'
            )
            
            # Normalize scores
            if 'predicted_rating' in combined.columns:
                combined['predicted_rating'] = combined['predicted_rating'].fillna(0) / 5.0
            if 'similarity_score' in combined.columns:
                combined['similarity_score'] = combined['similarity_score'].fillna(0)
            
            # Calculate hybrid score
            combined['hybrid_score'] = (
                combined['predicted_rating'].fillna(0) * collaborative_weight +
                combined['similarity_score'].fillna(0) * (1 - collaborative_weight)
            )
            
            # Get top recommendations
            result = combined.nlargest(n_recommendations, 'hybrid_score')
            return result[['movieId', 'title', 'genres', 'hybrid_score']]
        
        elif collab_recs is not None:
            return collab_recs.head(n_recommendations)
        elif content_recs is not None:
            return content_recs.head(n_recommendations)
        else:
            return self._get_popular_movies(n_recommendations)
    
    def _get_popular_movies(self, n_recommendations=10):
        """Get most popular movies by average rating"""
        movie_stats = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']
        
        # Filter movies with at least 50 ratings
        popular = movie_stats[movie_stats['rating_count'] >= 50].nlargest(
            n_recommendations, 'avg_rating'
        )
        
        result = popular.merge(self.movies_df, on='movieId', how='left')
        return result[['movieId', 'title', 'genres', 'avg_rating']]
    
    def save_model(self, filepath='models/recommendation_model.pkl'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'movie_features': self.movie_features,
            'user_movie_matrix': self.user_movie_matrix,
            'movies_df': self.movies_df,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'content_similarity': self.content_similarity
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/recommendation_model.pkl'):
        """Load a previously trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.movie_features = model_data['movie_features']
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.movies_df = model_data['movies_df']
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.content_similarity = model_data['content_similarity']
        
        print(f"Model loaded from {filepath}")
