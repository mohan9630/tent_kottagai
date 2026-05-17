# Movie Recommendation Web App

A Streamlit movie recommendation app built with pure Python. It fetches live data from TMDB, lets users search and filter movies, and recommends similar titles with a content-based filtering pipeline powered by TF-IDF and cosine similarity.

## Features

- Search movies by title or keyword
- Filter by genre, rating, and release year range
- Browse responsive movie cards with poster, synopsis, genre, and TMDB rating
- Generate content-based recommendations from the current catalog view
- Save favourites to a session-based watchlist and view them in a separate tab
- Use live TMDB data with API key management through `.env`

## Project Structure

```text
movie_app/
├── .streamlit/
│   └── config.toml
├── app.py
├── api.py
├── recommender.py
├── utils.py
├── requirements.txt
└── .env
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r movie_app/requirements.txt
```

3. Add your TMDB API key to `movie_app/.env`:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

4. Run the app:

```bash
cd movie_app
streamlit run app.py
```

## Streamlit Community Cloud

If you deploy this app to Streamlit Community Cloud, `.env` is not uploaded from GitHub. Add your TMDB key in the app's Secrets settings as:

```toml
TMDB_API_KEY = "your_tmdb_api_key_here"
```

The app now reads the key from `st.secrets["TMDB_API_KEY"]` when deployed and falls back to `movie_app/.env` for local development.

## How To Get A TMDB API Key

1. Go to [TMDB](https://www.themoviedb.org/signup) and create an account.
2. Open your account settings and navigate to the API section.
3. Request an API key for developer use.
4. Copy the generated key and place it in `movie_app/.env`.

## Notes

- The watchlist is stored in Streamlit session state, so it persists while the app session stays open.
- The app caches TMDB responses briefly to keep the interface snappy while still using live data.
- If TMDB requests fail, the UI shows a friendly error message instead of crashing.
