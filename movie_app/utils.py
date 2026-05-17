from __future__ import annotations

import html
from typing import Any
from urllib.parse import quote

import pandas as pd
import streamlit as st


POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://www.themoviedb.org/movie"
POSTER_PLACEHOLDER_URL = "data:image/svg+xml;utf8," + quote(
    """
    <svg xmlns="http://www.w3.org/2000/svg" width="500" height="750" viewBox="0 0 500 750">
      <rect width="500" height="750" fill="#2e1a00"/>
      <rect x="24" y="24" width="452" height="702" rx="28" fill="#1a0a00" stroke="#E8A400" stroke-width="4"/>
      <circle cx="250" cy="245" r="72" fill="#E8A400" opacity="0.25"/>
      <path d="M190 330h120v18H190zm-35 48h190v18H155zm30 48h130v18H185z" fill="#f5deb3" opacity="0.88"/>
      <text x="250" y="545" text-anchor="middle" font-size="38" font-family="Georgia, serif" fill="#E8A400">
        Poster
      </text>
      <text x="250" y="590" text-anchor="middle" font-size="32" font-family="Georgia, serif" fill="#f5deb3">
        Coming Soon
      </text>
    </svg>
    """
)


def poster_url(path: str | None) -> str | None:
    if not path:
        return None
    return f"{POSTER_BASE_URL}{path}"


def clean_year(value: Any) -> int | None:
    if not value:
        return None
    text = str(value)
    if len(text) < 4:
        return None
    try:
        return int(text[:4])
    except ValueError:
        return None


def normalize_movies(movies: list[dict[str, Any]], genre_map: dict[int, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for movie in movies:
        movie_id = movie.get("id")
        if not movie_id or movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)

        genre_names = movie.get("genre_names")
        if not genre_names:
            genre_names = [genre_map[item] for item in movie.get("genre_ids", []) if item in genre_map]

        rating = float(movie.get("vote_average") or 0.0)
        year = clean_year(movie.get("release_date"))

        rows.append(
            {
                "id": int(movie_id),
                "title": movie.get("title") or "Untitled",
                "overview": (movie.get("overview") or "Synopsis not available.").strip(),
                "genres": genre_names,
                "genre_text": ", ".join(genre_names) if genre_names else "Unknown",
                "rating": round(rating, 1),
                "popularity": float(movie.get("popularity") or 0.0),
                "release_year": year or 0,
                "poster_url": poster_url(movie.get("poster_path")),
                "tmdb_url": f"{TMDB_MOVIE_URL}/{movie_id}",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "title",
                "overview",
                "genres",
                "genre_text",
                "rating",
                "popularity",
                "release_year",
                "poster_url",
                "tmdb_url",
            ]
        )

    return df.sort_values(by=["popularity", "rating"], ascending=[False, False]).reset_index(drop=True)


def filter_movies(
    movies_df: pd.DataFrame,
    query: str = "",
    selected_genres: list[str] | None = None,
    rating_range: tuple[float, float] = (1.0, 10.0),
    year_range: tuple[int, int] = (1900, 2100),
) -> pd.DataFrame:
    if movies_df.empty:
        return movies_df

    filtered_df = movies_df.copy()
    if query:
        search_text = query.strip().lower()
        filtered_df = filtered_df[
            filtered_df["title"].str.lower().str.contains(search_text, na=False)
            | filtered_df["overview"].str.lower().str.contains(search_text, na=False)
            | filtered_df["genre_text"].str.lower().str.contains(search_text, na=False)
        ]

    if selected_genres:
        selected_set = set(selected_genres)
        filtered_df = filtered_df[
            filtered_df["genres"].apply(lambda items: bool(selected_set.intersection(items)))
        ]

    filtered_df = filtered_df[
        filtered_df["rating"].between(rating_range[0], rating_range[1], inclusive="both")
    ]
    filtered_df = filtered_df[
        filtered_df["release_year"].between(year_range[0], year_range[1], inclusive="both")
    ]
    return filtered_df.reset_index(drop=True)


def truncated(text: str, max_length: int = 180) -> str:
    value = (text or "").strip()
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3].rstrip()}..."


def movie_card_markup(movie: pd.Series) -> str:
    poster = movie.get("poster_url") or POSTER_PLACEHOLDER_URL
    title = html.escape(movie.get("title", "Untitled"))
    genre_text = html.escape(movie.get("genre_text", "Unknown"))
    overview = html.escape(truncated(movie.get("overview", "")))
    rating = movie.get("rating", 0.0)
    year = movie.get("release_year", "N/A")
    tmdb_url = movie.get("tmdb_url", "#")

    return f"""
    <div class="movie-card">
      <img src="{poster}" alt="{title} poster" class="movie-poster" onerror="this.onerror=null;this.src='{POSTER_PLACEHOLDER_URL}';" />
      <div class="movie-card-body">
        <div class="movie-title">{title}</div>
        <div class="movie-meta">
          <span>{genre_text}</span>
          <span>{rating:.1f}/10</span>
          <span>{year}</span>
        </div>
        <p class="movie-overview">{overview}</p>
        <a class="movie-link" href="{tmdb_url}" target="_blank">Open on TMDB</a>
      </div>
    </div>
    """


def ensure_session_state() -> None:
    st.session_state.setdefault("watchlist", {})
    st.session_state.setdefault("selected_movie_id", None)


def add_to_watchlist(movie: pd.Series) -> None:
    st.session_state.watchlist[int(movie["id"])] = movie.to_dict()


def remove_from_watchlist(movie_id: int) -> None:
    st.session_state.watchlist.pop(int(movie_id), None)


def watchlist_ids() -> list[int]:
    return list(st.session_state.watchlist.keys())


def get_watchlist_dataframe() -> pd.DataFrame:
    records = list(st.session_state.watchlist.values())
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values(by=["rating", "title"], ascending=[False, True])
