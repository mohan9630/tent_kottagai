from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TMDB_BASE_URL = "https://api.themoviedb.org/3"


class TMDBApiError(RuntimeError):
    """Raised when TMDB credentials are missing or the API request fails."""


class TMDBClient:
    def __init__(self, api_key: str | None = None, language: str = "en-US") -> None:
        env_path = Path(__file__).resolve().parent / ".env"
        load_dotenv(dotenv_path=env_path)
        self.api_key = api_key or self._load_api_key()
        if not self.api_key:
            raise TMDBApiError(
                "TMDB API key not found. Add TMDB_API_KEY to movie_app/.env for local runs, "
                "or set TMDB_API_KEY in your Streamlit app Secrets when deployed."
            )

        self.language = language
        self.session = requests.Session()
        # Some local environments inject network settings that cause keyed TMDB
        # requests to be reset. Using a direct session keeps the API calls stable.
        self.session.trust_env = False
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Connection": "close",
                "User-Agent": "CineScope/1.0",
            }
        )
        retries = Retry(
            total=2,
            connect=2,
            read=2,
            backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _load_api_key(self) -> str | None:
        try:
            if "TMDB_API_KEY" in st.secrets:
                return str(st.secrets["TMDB_API_KEY"]).strip()
        except Exception:
            pass

        value = os.getenv("TMDB_API_KEY", "").strip()
        return value or None

    def _error_message(self, payload: dict[str, Any] | None, fallback: str) -> str:
        if not payload:
            return fallback

        status_message = payload.get("status_message")
        if status_message:
            return f"TMDB request failed: {status_message}"
        return fallback

    def _get_with_urllib(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        query = urlencode(params)
        request = Request(
            f"{url}?{query}",
            headers={
                "Accept": "application/json",
                "Connection": "close",
                "User-Agent": "CineScope/1.0",
            },
        )
        try:
            with urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = None
            raise TMDBApiError(self._error_message(payload, f"TMDB request failed: HTTP {exc.code}")) from exc
        except URLError as exc:
            raise TMDBApiError(
                "Unable to reach TMDB right now. Please check your network connection and try again."
            ) from exc

    def _get(self, endpoint: str, **params: Any) -> dict[str, Any]:
        request_params = {
            "api_key": self.api_key,
            "language": self.language,
            **params,
        }
        url = f"{TMDB_BASE_URL}/{endpoint.lstrip('/')}"

        try:
            response = self.session.get(url, params=request_params, timeout=20)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            payload = None
            if exc.response is not None:
                try:
                    payload = exc.response.json()
                except ValueError:
                    payload = None
            raise TMDBApiError(self._error_message(payload, "TMDB request failed.")) from exc
        except requests.RequestException as exc:
            try:
                return self._get_with_urllib(url, request_params)
            except TMDBApiError as fallback_exc:
                raise fallback_exc from exc

    def get_genres(self) -> dict[int, str]:
        data = self._get("genre/movie/list")
        return {genre["id"]: genre["name"] for genre in data.get("genres", [])}

    def search_movies(self, query: str, pages: int = 1) -> list[dict[str, Any]]:
        movies: list[dict[str, Any]] = []
        for page in range(1, max(1, pages) + 1):
            data = self._get(
                "search/movie",
                query=query,
                page=page,
                include_adult="false",
            )
            movies.extend(data.get("results", []))
        return movies

    def search_keywords(self, query: str, limit: int = 3) -> list[int]:
        data = self._get("search/keyword", query=query, page=1)
        keyword_ids = [item["id"] for item in data.get("results", [])[:limit]]
        return keyword_ids

    def discover_movies(
        self,
        genre_ids: list[int] | None = None,
        rating_min: float | None = None,
        year_range: tuple[int, int] | None = None,
        keyword_ids: list[int] | None = None,
        pages: int = 1,
        sort_by: str = "popularity.desc",
    ) -> list[dict[str, Any]]:
        movies: list[dict[str, Any]] = []
        for page in range(1, max(1, pages) + 1):
            params: dict[str, Any] = {
                "include_adult": "false",
                "include_video": "false",
                "page": page,
                "sort_by": sort_by,
                "vote_count.gte": 100,
            }
            if genre_ids:
                params["with_genres"] = ",".join(str(item) for item in genre_ids)
            if rating_min is not None:
                params["vote_average.gte"] = rating_min
            if year_range:
                params["primary_release_date.gte"] = f"{year_range[0]}-01-01"
                params["primary_release_date.lte"] = f"{year_range[1]}-12-31"
            if keyword_ids:
                params["with_keywords"] = ",".join(str(item) for item in keyword_ids)

            data = self._get("discover/movie", **params)
            movies.extend(data.get("results", []))
        return movies

    def search_and_discover(
        self,
        query: str,
        genre_ids: list[int] | None = None,
        rating_min: float | None = None,
        year_range: tuple[int, int] | None = None,
        pages: int = 2,
    ) -> list[dict[str, Any]]:
        movies = self.search_movies(query=query, pages=pages)
        try:
            keyword_ids = self.search_keywords(query)
        except TMDBApiError:
            keyword_ids = []

        for keyword_id in keyword_ids:
            try:
                movies.extend(
                    self.discover_movies(
                        genre_ids=genre_ids,
                        rating_min=rating_min,
                        year_range=year_range,
                        keyword_ids=[keyword_id],
                        pages=1,
                    )
                )
            except TMDBApiError:
                continue
        return movies
