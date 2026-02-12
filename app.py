"""
Streamlit interface for Movie Recommendation System
"""

import os
import streamlit as st
import pandas as pd
from recommendation_engine import MovieRecommendationEngine
from data_processor import DataProcessor

# Page config
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — cinema-inspired dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Force Streamlit theme primary (slider track + value label) to green */
    :root, .stApp {
        --primary: #22c55e !important;
        --primary-color: #22c55e !important;
    }
    /* Main container */
    .stApp {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    }
    
    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #252b3b 50%, #1a1f2e 100%);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(212, 175, 55, 0.05);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #d4af37 0%, #f4e4bc 50%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
    }
    .hero-sub {
        font-family: 'Outfit', sans-serif;
        color: #8b949e;
        font-size: 1.05rem;
        font-weight: 300;
    }
    
    /* Section headers */
    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 0.25rem;
    }
    .section-caption {
        color: #8b949e;
        font-size: 0.9rem;
        margin-bottom: 1.25rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%) !important;
        border-right: 1px solid rgba(212, 175, 55, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #e6edf3 !important;
    }
    .sidebar-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #d4af37 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Metric cards in sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(26, 31, 46, 0.9) 0%, rgba(22, 27, 34, 0.95) 100%);
        padding: 0.75rem 1rem;
        border-radius: 10px;
        border: 1px solid rgba(212, 175, 55, 0.15);
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: #8b949e !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #d4af37 !important;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    
    /* Movie cards */
    .movie-card {
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.95) 0%, rgba(13, 17, 23, 0.98) 100%);
        border: 1px solid rgba(212, 175, 55, 0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    .movie-card-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.05rem;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 0.35rem;
    }
    .movie-card-genres {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 0.25rem;
    }
    .movie-card-score {
        font-family: 'Outfit', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #d4af37;
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, #d4af37 0%, #b8962e 100%) !important;
        color: #0d1117 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.5rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(212, 175, 55, 0.4) !important;
    }
    
    /* Inputs */
    .stNumberInput input, .stSelectbox > div, .stTextInput > div > input {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid rgba(212, 175, 55, 0.25) !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
    }
    
    /* Slider - GREEN: Value label, track, rail, thumb - broad selectors */
    [data-testid="stSlider"] [data-testid="stSliderValue"],
    [data-testid="stSlider"] div[data-testid="stSliderValue"],
    [data-testid="stSlider"] .stSliderValue,
    [data-testid="stSlider"] span[aria-hidden="true"],
    [data-testid="stSlider"] [class*="ValueLabel"],
    [data-testid="stSlider"] [class*="valueLabel"],
    [data-testid="stSlider"] [class*="PrivateValueLabel"],
    [data-testid="stSlider"] [class*="thumb"] span,
    [data-testid="stSlider"] span,
    [data-testid="stSlider"] > div > div:last-child,
    [data-testid="stSlider"] label + div div,
    [data-testid="stSlider"] > div > div {
        color: #22c55e !important;
        fill: #22c55e !important;
        font-weight: 600 !important;
    }
    [data-testid="stSlider"] [class*="MuiSlider-track"],
    [data-testid="stSlider"] [class*="track"]:not([class*="thumb"]):not([class*="ValueLabel"]) {
        background-color: #22c55e !important;
        background: #22c55e !important;
        fill: #22c55e !important;
        opacity: 1 !important;
    }
    [data-testid="stSlider"] [class*="MuiSlider-rail"],
    [data-testid="stSlider"] [class*="rail"] {
        background-color: rgba(34, 197, 94, 0.3) !important;
        background: rgba(34, 197, 94, 0.3) !important;
        opacity: 1 !important;
    }
    [data-testid="stSlider"] [class*="MuiSlider-thumb"],
    [data-testid="stSlider"] [class*="thumb"]:not(span) {
        color: #22c55e !important;
        border-color: #22c55e !important;
        background-color: #22c55e !important;
        fill: #22c55e !important;
    }
    [data-testid="stSlider"] > div > label,
    [data-testid="stSlider"] label:first-child {
        color: #8b949e !important;
    }
    /* Global slider fallback - any element with Slider/track/rail/thumb in class */
    .stSlider [class*="track"]:not([class*="thumb"]),
    .stSlider [class*="rail"],
    [class*="MuiSlider-track"],
    [class*="MuiSlider-rail"] {
        background: #22c55e !important;
        background-color: #22c55e !important;
    }
    .stSlider [class*="rail"] {
        background: rgba(34, 197, 94, 0.3) !important;
    }
    .stSlider span,
    .stSlider [class*="ValueLabel"],
    .stSlider [class*="thumb"] span {
        color: #22c55e !important;
    }
    .stSlider [class*="thumb"] {
        color: #22c55e !important;
        background-color: #22c55e !important;
    }
    
    /* DataFrames */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(212, 175, 55, 0.15);
        background: rgba(22, 27, 34, 0.6) !important;
    }
    
    /* Info / success boxes */
    .stAlert {
        border-radius: 10px;
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    
    /* Hide Streamlit branding for cleaner look */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def _ensure_data(movies_path, ratings_path):
    """Generate data from dataset/movie_catalog.csv if data files are missing (for deploy)."""
    if os.path.exists(movies_path) and os.path.exists(ratings_path):
        return True
    try:
        import generate_dataset
        generate_dataset.main()
        return os.path.exists(movies_path) and os.path.exists(ratings_path)
    except Exception:
        return False


@st.cache_resource
def load_engine():
    """Load or build the recommendation engine (cached)."""
    engine = MovieRecommendationEngine()
    # Use paths relative to repo root (works for Streamlit Community Cloud)
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "models", "recommendation_model.pkl")
    movies_path = os.path.join(base, "data", "movies.csv")
    ratings_path = os.path.join(base, "data", "ratings.csv")

    if os.path.exists(model_path):
        try:
            engine.load_model(model_path)
            if engine.ratings_df is None and os.path.exists(ratings_path):
                engine.ratings_df = pd.read_csv(ratings_path)
            return engine, None
        except Exception as e:
            return None, str(e)

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        if not _ensure_data(movies_path, ratings_path):
            return None, (
                "Data not found. Ensure dataset/movie_catalog.csv exists in the repo, "
                "or run 'python generate_dataset.py' locally."
            )

    if not engine.load_data(movies_path, ratings_path):
        return None, "Failed to load data."

    processor = DataProcessor()
    engine.preprocess_data()
    engine.movies_df = processor.clean_movie_data(engine.movies_df)
    engine.ratings_df = processor.clean_ratings_data(engine.ratings_df)
    engine.user_movie_matrix = engine.ratings_df.pivot_table(
        index="userId", columns="movieId", values="rating", fill_value=0
    )
    engine.train_collaborative_filtering(n_components=50, max_iter=300)
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        engine.save_model(model_path)
    except Exception:
        pass  # e.g. read-only filesystem on cloud; cache still works
    return engine, None


def render_movie_cards(df, title_col="title", genres_col="genres", score_col=None):
    """Render recommendation rows as styled cards."""
    score_col = score_col or (df.columns[-1] if len(df.columns) > 2 else None)
    html_cards = []
    for _, row in df.iterrows():
        title = str(row[title_col])
        genres = str(row[genres_col]) if genres_col in row else ""
        score = f"{float(row[score_col]):.2f}" if score_col and score_col in row else ""
        html_cards.append(
            f"""
            <div class="movie-card">
                <div class="movie-card-title">{title}</div>
                <div class="movie-card-genres">{genres}</div>
                {f'<div class="movie-card-score">Score: {score}</div>' if score else ''}
            </div>
            """
        )
    st.markdown("<br>".join(html_cards), unsafe_allow_html=True)


def main():
    # Hero
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🎬 Movie Recommendation System</div>
            <div class="hero-sub">Personalized picks using collaborative & content-based filtering — discover your next watch.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading recommendation engine…"):
        engine, error = load_engine()
    if error:
        st.error(error)
        return

    # Sidebar
    st.sidebar.markdown('<p class="sidebar-title">▸ Navigation</p>', unsafe_allow_html=True)
    st.sidebar.markdown("")
    page = st.sidebar.radio(
        "Choose mode",
        [
            "Home",
            "Recommendations for User",
            "Similar Movies",
            "Hybrid Recommendations",
            "Search Movies",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="sidebar-title">▸ Stats</p>', unsafe_allow_html=True)

    if engine.movies_df is not None:
        st.sidebar.metric("Movies", len(engine.movies_df))
    if engine.ratings_df is not None:
        st.sidebar.metric("Ratings", len(engine.ratings_df))
    if engine.user_movie_matrix is not None:
        st.sidebar.metric("Users", engine.user_movie_matrix.shape[0])

    # ---------- Pages ----------
    if page == "Home":
        st.markdown('<p class="section-title">Welcome</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-caption">Choose a mode from the sidebar to get recommendations or search the catalog.</p>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                """
                **👤 For User**  
                Get recommendations tailored to a user based on similar users' tastes (collaborative filtering).
                """
            )
        with c2:
            st.markdown(
                """
                **🎞️ Similar Movies**  
                Pick a movie and find others with similar genres (content-based).
                """
            )
        with c3:
            st.markdown(
                """
                **✨ Hybrid**  
                Combine user preferences and a reference movie for richer recommendations.
                """
            )
        st.info("Select a mode from the sidebar to start.")

    elif page == "Recommendations for User":
        st.markdown('<p class="section-title">Recommendations for User</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-caption">Collaborative filtering: movies that similar users liked.</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            user_id = st.number_input("User ID", min_value=1, value=1, step=1)
            n_recs = st.slider("Number of recommendations", 5, 25, 10)
        if st.button("Get recommendations", type="primary"):
            with st.spinner("Finding recommendations..."):
                try:
                    recs = engine.get_collaborative_recommendations(user_id, n_recommendations=n_recs)
                    recs_display = recs.rename(columns={"predicted_rating": "Score"})
                    render_movie_cards(recs_display, score_col="Score")
                except Exception as e:
                    st.error(str(e))

    elif page == "Similar Movies":
        st.markdown('<p class="section-title">Similar Movies</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-caption">Content-based: movies with similar genres.</p>',
            unsafe_allow_html=True,
        )
        movies_list = engine.movies_df[["movieId", "title"]].drop_duplicates()
        movie_options = dict(zip(movies_list["title"], movies_list["movieId"]))
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_title = st.selectbox("Select a movie", options=list(movie_options.keys()))
            n_recs = st.slider("Number of similar movies", 5, 25, 10)
        movie_id = movie_options.get(selected_title)
        if movie_id and st.button("Find similar movies", type="primary"):
            with st.spinner("Finding similar movies..."):
                try:
                    recs = engine.get_content_based_recommendations(movie_id, n_recommendations=n_recs)
                    if recs is not None:
                        recs_display = recs.rename(columns={"similarity_score": "Similarity"})
                        render_movie_cards(recs_display, score_col="Similarity")
                    else:
                        st.warning("No similar movies found.")
                except Exception as e:
                    st.error(str(e))

    elif page == "Hybrid Recommendations":
        st.markdown('<p class="section-title">Hybrid Recommendations</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-caption">Combines collaborative and content-based scores.</p>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.number_input("User ID", min_value=1, value=1, step=1, key="hybrid_user")
            n_recs = st.slider("Number of recommendations", 5, 25, 10, key="hybrid_n")
        with col2:
            use_movie = st.checkbox("Also use a reference movie for content boost")
            movie_id = None
            if use_movie:
                movies_list = engine.movies_df[["movieId", "title"]].drop_duplicates()
                movie_options = dict(zip(movies_list["title"], movies_list["movieId"]))
                ref_title = st.selectbox("Reference movie", options=list(movie_options.keys()), key="hybrid_movie")
                movie_id = movie_options.get(ref_title)
        if st.button("Get hybrid recommendations", type="primary"):
            with st.spinner("Computing hybrid recommendations..."):
                try:
                    recs = engine.get_hybrid_recommendations(
                        user_id, movie_id=movie_id, n_recommendations=n_recs
                    )
                    score_col = "hybrid_score" if "hybrid_score" in recs.columns else recs.columns[-1]
                    render_movie_cards(recs, score_col=score_col)
                except Exception as e:
                    st.error(str(e))

    elif page == "Search Movies":
        st.markdown('<p class="section-title">Search Movies</p>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-caption">Search the movie catalog by title.</p>',
            unsafe_allow_html=True,
        )
        search = st.text_input("Search by title", placeholder="e.g. Matrix, Dune, Inception")
        if search:
            matches = engine.movies_df[
                engine.movies_df["title"].str.contains(search, case=False, na=False)
            ]
            if len(matches) > 0:
                st.dataframe(
                    matches[["movieId", "title", "genres"]].head(20),
                    use_container_width=True,
                )
            else:
                st.info("No movies found.")


if __name__ == "__main__":
    main()
