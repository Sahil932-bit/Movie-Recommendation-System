# Deploy on Streamlit Community Cloud

This app is ready to deploy on [Streamlit Community Cloud](https://share.streamlit.io/).

## Quick deploy

1. **Push the repo to GitHub**  
   Ensure these are in the repo:
   - `app.py` (entrypoint)
   - `requirements.txt`
   - `dataset/movie_catalog.csv`
   - `recommendation_engine.py`, `data_processor.py`, `generate_dataset.py`

2. **Go to [share.streamlit.io](https://share.streamlit.io/)**  
   Sign in with GitHub.

3. **New app**
   - Click **New app**.
   - Choose your repo and branch.
   - **Main file path:** `app.py`
   - Click **Deploy**.

4. **First run**  
   If `data/` or `models/` are not in the repo, the app will:
   - Generate `data/movies.csv` and `data/ratings.csv` from `dataset/movie_catalog.csv`
   - Train the model (may take 30–60 seconds)
   - Then serve recommendations

   Later runs use the cached engine, so they are fast.

## Optional: faster startup

To avoid generating data and training on every cold start:

1. **Generate data and train locally:**
   ```bash
   python generate_dataset.py
   python main.py
   ```

2. **Commit the generated files** (if you previously ignored them):
   - Un-ignore in `.gitignore`: remove or comment out `data/` and `models/`.
   - Commit and push:
     ```bash
     git add data/ models/
     git commit -m "Add data and trained model for deploy"
     git push
     ```

3. **Redeploy**  
   The app will load the existing model and skip training.

## Requirements

- Python 3.8+
- Dependencies in `requirements.txt` (Streamlit Cloud installs these automatically)

## Config

- Theme and slider styling: `.streamlit/config.toml`  
  No secrets or API keys are required.
