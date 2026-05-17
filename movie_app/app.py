from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from api import TMDBApiError, TMDBClient
from recommender import get_recommendations, get_watchlist_recommendations
from utils import (
    add_to_watchlist,
    ensure_session_state,
    filter_movies,
    get_watchlist_dataframe,
    movie_card_markup,
    normalize_movies,
    remove_from_watchlist,
    watchlist_ids,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.jpg"

st.set_page_config(
    page_title="Tent Kottagai",
    page_icon="🎪",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_session_state()


@st.cache_data(ttl=1800, show_spinner=False)
def load_genres() -> dict[int, str]:
    client = TMDBClient()
    return client.get_genres()


@st.cache_data(ttl=900, show_spinner=False)
def load_movies(
    query: str,
    genre_ids: tuple[int, ...],
    rating_min: float,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    client = TMDBClient()
    if query.strip():
        raw_movies = client.search_and_discover(
            query=query.strip(),
            genre_ids=list(genre_ids) or None,
            rating_min=rating_min,
            year_range=(year_start, year_end),
            pages=2,
        )
    else:
        raw_movies = client.discover_movies(
            genre_ids=list(genre_ids) or None,
            rating_min=rating_min,
            year_range=(year_start, year_end),
            pages=3,
        )

    genre_map = load_genres()
    return normalize_movies(raw_movies, genre_map)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(252, 163, 17, 0.16), transparent 30%),
                radial-gradient(circle at top right, rgba(76, 201, 240, 0.10), transparent 26%),
                linear-gradient(180deg, #090b10 0%, #0f141c 100%);
        }
        .hero {
            padding: 1.75rem 1.75rem 1.5rem 1.75rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(26, 33, 48, 0.88), rgba(11, 15, 22, 0.92));
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
            margin-bottom: 1rem;
        }
        .hero-kicker {
            color: #fca311;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            font-size: 0.8rem;
            font-weight: 700;
        }
        .hero-title {
            font-size: 2.6rem;
            line-height: 1.1;
            font-weight: 700;
            color: #f8fafc;
            margin: 0.4rem 0 0.55rem 0;
        }
        .hero-copy {
            color: #cbd5e1;
            max-width: 720px;
            font-size: 1rem;
            line-height: 1.7;
            margin: 0;
        }
        .stat-strip {
            display: flex;
            gap: 0.85rem;
            flex-wrap: wrap;
            margin: 0.8rem 0 1.4rem 0;
        }
        .stat-chip {
            padding: 0.65rem 0.85rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #f8fafc;
            font-size: 0.92rem;
        }
        .movie-card {
            background: linear-gradient(180deg, rgba(16, 22, 31, 0.96), rgba(10, 14, 21, 0.96));
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 22px;
            overflow: hidden;
            min-height: 640px;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
        }
        .movie-poster {
            width: 100%;
            aspect-ratio: 2 / 3;
            object-fit: cover;
            display: block;
            background: #151922;
        }
        .movie-card-body {
            padding: 1rem 1rem 0.9rem 1rem;
        }
        .movie-title {
            color: #f8fafc;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
            min-height: 3rem;
        }
        .movie-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-bottom: 0.75rem;
        }
        .movie-meta span {
            font-size: 0.82rem;
            color: #dbe4f0;
            background: rgba(255, 255, 255, 0.07);
            border-radius: 999px;
            padding: 0.25rem 0.55rem;
        }
        .movie-overview {
            color: #bac7d6;
            font-size: 0.93rem;
            line-height: 1.6;
            min-height: 7.2rem;
            margin: 0 0 0.85rem 0;
        }
        .movie-link {
            color: #8ecae6;
            text-decoration: none;
            font-weight: 600;
        }
        .section-title {
            color: #f8fafc;
            font-size: 1.35rem;
            font-weight: 700;
            margin: 0.2rem 0 0.9rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_movie_grid(movies_df: pd.DataFrame, key_prefix: str, columns_per_row: int = 4) -> None:
    if movies_df.empty:
        st.info("No movies matched this view yet. Try broadening the filters or search text.")
        return

    saved_ids = set(watchlist_ids())
    for start in range(0, len(movies_df), columns_per_row):
        row = movies_df.iloc[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, (_, movie) in zip(columns, row.iterrows()):
            movie_id = int(movie["id"])
            with column:
                st.markdown(movie_card_markup(movie), unsafe_allow_html=True)
                action_columns = st.columns(2)
                with action_columns[0]:
                    if st.button("Find Similar", key=f"{key_prefix}-similar-{movie_id}", use_container_width=True):
                        st.session_state.selected_movie_id = movie_id
                        st.rerun()
                with action_columns[1]:
                    if movie_id in saved_ids:
                        if st.button("Remove", key=f"{key_prefix}-remove-{movie_id}", use_container_width=True):
                            remove_from_watchlist(movie_id)
                            st.rerun()
                    else:
                        if st.button("Save", key=f"{key_prefix}-save-{movie_id}", use_container_width=True):
                            add_to_watchlist(movie)
                            st.rerun()


inject_styles()

current_year = datetime.now().year
year_limits = (1950, current_year)

st.image(str(BANNER_PATH), use_container_width=True)
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Movie Discovery Studio</div>
      <div class="hero-title">Tent Kottagai</div>
      <p class="hero-copy">
        explore the genre you want, find movies similar to your favourites, and build a watchlist for later. 
    """,
    unsafe_allow_html=True,
)

try:
    with st.spinner("Connecting to TMDB and loading genres..."):
        genre_map = load_genres()
except TMDBApiError as exc:
    st.error(str(exc))
    st.stop()

all_genres = sorted(genre_map.values())
genre_name_to_id = {name: genre_id for genre_id, name in genre_map.items()}

query = st.text_input("Search by movie title or keyword", placeholder="Try: dystopian sci-fi, space, batman...")

with st.sidebar:
    st.image(str(LOGO_PATH), use_container_width=True)
    st.header("Filters")
    selected_genres = st.multiselect("Genre", options=all_genres)
    rating_range = st.slider("Rating", min_value=1.0, max_value=10.0, value=(1.0, 10.0), step=0.1)
    release_year_range = st.slider(
        "Release year",
        min_value=year_limits[0],
        max_value=year_limits[1],
        value=year_limits,
        step=1,
    )
    st.caption("The app fetches fresh TMDB data and then applies local filtering for the final view.")

selected_genre_ids = tuple(genre_name_to_id[name] for name in selected_genres)

try:
    with st.spinner("Fetching live movie data from TMDB..."):
        movies_df = load_movies(
            query=query,
            genre_ids=selected_genre_ids,
            rating_min=rating_range[0],
            year_start=release_year_range[0],
            year_end=release_year_range[1],
        )
except TMDBApiError as exc:
    st.error(str(exc))
    st.stop()

filtered_df = filter_movies(
    movies_df=movies_df,
    query=query,
    selected_genres=selected_genres,
    rating_range=rating_range,
    year_range=release_year_range,
)

watchlist_df = get_watchlist_dataframe()
active_title = "Current Browse Results"
total_movies = len(filtered_df)
watchlist_total = len(watchlist_df)

st.markdown(
    f"""
    <div class="stat-strip">
      <div class="stat-chip">{active_title}: <strong>{total_movies}</strong></div>
      <div class="stat-chip">Saved To Watchlist: <strong>{watchlist_total}</strong></div>
      <div class="stat-chip">Recommendation Engine: <strong>TF-IDF + Cosine Similarity</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)

browse_tab, watchlist_tab = st.tabs(["Discover", "Watchlist"])

with browse_tab:
    left_column, right_column = st.columns([2.3, 1.1], gap="large")

    with left_column:
        st.markdown('<div class="section-title">Movie Catalog</div>', unsafe_allow_html=True)
        render_movie_grid(filtered_df.head(24), key_prefix="browse")

    with right_column:
        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        if filtered_df.empty:
            st.info("Load some movies first and the recommendation panel will activate automatically.")
        else:
            title_lookup = {row["title"]: int(row["id"]) for _, row in filtered_df.iterrows()}
            default_index = 0
            if st.session_state.selected_movie_id in filtered_df["id"].values:
                selected_title = filtered_df.loc[
                    filtered_df["id"] == st.session_state.selected_movie_id, "title"
                ].iloc[0]
                default_index = list(title_lookup.keys()).index(selected_title)

            selected_title = st.selectbox(
                "Recommend movies similar to",
                options=list(title_lookup.keys()),
                index=default_index,
            )
            selected_movie_id = title_lookup[selected_title]
            st.session_state.selected_movie_id = selected_movie_id
            recommended_df = get_recommendations(filtered_df, selected_movie_id, top_n=8)
            if recommended_df.empty:
                st.info("Not enough movie context yet to generate recommendations.")
            else:
                render_movie_grid(recommended_df, key_prefix="recommend", columns_per_row=2)

with watchlist_tab:
    st.markdown('<div class="section-title">Saved Favourites / Watchlist</div>', unsafe_allow_html=True)
    if watchlist_df.empty:
        st.info("Save a few titles from the Discover tab to build your watchlist.")
    else:
        render_movie_grid(watchlist_df, key_prefix="watchlist")
        st.markdown('<div class="section-title">Because You Saved These</div>', unsafe_allow_html=True)
        recommendation_source_df = (
            pd.concat([movies_df, watchlist_df], ignore_index=True).drop_duplicates(subset="id")
            if not movies_df.empty
            else watchlist_df
        )
        watchlist_recommendations = get_watchlist_recommendations(
            recommendation_source_df,
            watchlist_ids(),
            top_n=8,
        )
        if watchlist_recommendations.empty:
            st.info("Add more movies or widen the filters to generate watchlist-based recommendations.")
        else:
            render_movie_grid(watchlist_recommendations, key_prefix="watch-recommend")
