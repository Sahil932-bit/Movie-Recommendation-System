"""
Main script to train and use the Movie Recommendation System
"""

import os
from recommendation_engine import MovieRecommendationEngine
from data_processor import DataProcessor


def main():
    """Main function to run the recommendation system"""
    
    print("=" * 60)
    print("Movie Recommendation System")
    print("=" * 60)
    
    # Initialize components
    engine = MovieRecommendationEngine()
    processor = DataProcessor()
    
    # Check if data exists
    movies_path = 'data/movies.csv'
    ratings_path = 'data/ratings.csv'
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print("\n[!] Data files not found!")
        print("Please download the MovieLens dataset and place it in the 'data' directory.")
        print("You can download it from: https://grouplens.org/datasets/movielens/")
        print("\nExpected files:")
        print(f"  - {movies_path}")
        print(f"  - {ratings_path}")
        return
    
    # Load data
    print("\n[*] Loading data...")
    if not engine.load_data(movies_path, ratings_path):
        return
    
    # Preprocess data
    print("\n[*] Preprocessing data...")
    engine.preprocess_data()
    
    # Clean data
    print("\n[*] Cleaning data...")
    engine.movies_df = processor.clean_movie_data(engine.movies_df)
    engine.ratings_df = processor.clean_ratings_data(engine.ratings_df)
    
    # Recreate user-movie matrix after cleaning
    engine.user_movie_matrix = engine.ratings_df.pivot_table(
        index='userId',
        columns='movieId',
        values='rating',
        fill_value=0
    )
    
    # Train model
    print("\n[*] Training collaborative filtering model...")
    engine.train_collaborative_filtering(n_components=50, max_iter=200)
    
    # Save model
    print("\n[*] Saving model...")
    engine.save_model('models/recommendation_model.pkl')
    
    # Demo recommendations
    print("\n" + "=" * 60)
    print("Demo: Getting Recommendations")
    print("=" * 60)
    
    # Get a sample user
    sample_user = engine.ratings_df['userId'].iloc[0]
    print(f"\n[*] Recommendations for User ID: {sample_user}")
    print("-" * 60)
    
    # Collaborative filtering recommendations
    print("\n[*] Collaborative Filtering Recommendations:")
    collab_recs = engine.get_collaborative_recommendations(sample_user, n_recommendations=5)
    print(collab_recs.to_string(index=False))
    
    # Content-based recommendations
    sample_movie_id = engine.movies_df['movieId'].iloc[0]
    print(f"\n[*] Content-Based Recommendations (similar to Movie ID: {sample_movie_id}):")
    content_recs = engine.get_content_based_recommendations(sample_movie_id, n_recommendations=5)
    if content_recs is not None:
        print(content_recs.to_string(index=False))
    
    # Hybrid recommendations
    print(f"\n[*] Hybrid Recommendations for User {sample_user}:")
    hybrid_recs = engine.get_hybrid_recommendations(sample_user, n_recommendations=5)
    print(hybrid_recs.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("[OK] Recommendation system ready!")
    print("=" * 60)


def interactive_mode():
    """Interactive mode for getting recommendations"""
    engine = MovieRecommendationEngine()
    
    # Try to load existing model
    if os.path.exists('models/recommendation_model.pkl'):
        print("Loading saved model...")
        engine.load_model('models/recommendation_model.pkl')
    else:
        print("No saved model found. Please run main() first to train the model.")
        return
    
    print("\n" + "=" * 60)
    print("Interactive Recommendation System")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a user (Collaborative Filtering)")
        print("2. Get similar movies (Content-Based)")
        print("3. Get hybrid recommendations")
        print("4. Search for a movie")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            try:
                user_id = int(input("Enter User ID: "))
                n_recs = int(input("Number of recommendations (default 10): ") or "10")
                recs = engine.get_collaborative_recommendations(user_id, n_recs)
                print("\n" + recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '2':
            try:
                movie_id = int(input("Enter Movie ID: "))
                n_recs = int(input("Number of recommendations (default 10): ") or "10")
                recs = engine.get_content_based_recommendations(movie_id, n_recs)
                if recs is not None:
                    print("\n" + recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '3':
            try:
                user_id = int(input("Enter User ID: "))
                movie_id = input("Enter Movie ID (optional, press Enter to skip): ").strip()
                movie_id = int(movie_id) if movie_id else None
                n_recs = int(input("Number of recommendations (default 10): ") or "10")
                recs = engine.get_hybrid_recommendations(user_id, movie_id, n_recs)
                print("\n" + recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            try:
                search_term = input("Enter movie title to search: ").strip()
                matches = engine.movies_df[
                    engine.movies_df['title'].str.contains(search_term, case=False, na=False)
                ].head(10)
                if len(matches) > 0:
                    print("\n" + matches[['movieId', 'title', 'genres']].to_string(index=False))
                else:
                    print("No movies found.")
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '5':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        main()
