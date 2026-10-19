"""
Unit tests for data preprocessing, quality checks, and mapping generation.
"""

import pandas as pd
import pytest

from src.data_loader import detect_columns
from src.preprocessing import DataPreprocessor


@pytest.fixture
def sample_raw_data():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2, 3],
            "movieId": [10, 20, 20, 30, 10],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
            "timestamp": [1000, 1005, 1010, 1015, 1020],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30, 40],
            "title": ["Toy Story (1995)", "Jumanji (1995)", "Grumpier Old Men (1995)", "Waiting to Exhale (1995)"],
            "genres": ["Animation|Children", "Adventure|Children", "Comedy|Romance", "Comedy|Drama"],
        }
    )
    return ratings, movies


def test_detect_columns(sample_raw_data):
    ratings, movies = sample_raw_data
    schema = detect_columns(ratings, movies)
    assert schema["user_col"] == "userId"
    assert schema["movie_col"] == "movieId"
    assert schema["rating_col"] == "rating"
    assert schema["title_col"] == "title"
    assert schema["genre_col"] == "genres"


def test_clean_ratings_duplicate_and_bounds():
    messy_ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2],
            "movieId": [10, 10, 20, 30],
            "rating": [3.0, 4.5, 7.0, -1.0],  # 7.0 and -1.0 are invalid bounds (> 5.0 and < 0.5)
            "timestamp": [100, 200, 150, 120],
        }
    )
    schema = {"user_col": "userId", "movie_col": "movieId", "rating_col": "rating", "timestamp_col": "timestamp"}
    preprocessor = DataPreprocessor()
    cleaned = preprocessor.clean_ratings(messy_ratings, schema)

    # Only valid bound is 3.0 and 4.5 for movie 10; timestamp 200 is latest, so 4.5 is kept
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["rating"] == 4.5
    assert preprocessor.cleaning_report["out_of_bounds_ratings"] == 2


def test_fit_mappings(sample_raw_data):
    ratings, movies = sample_raw_data
    schema = detect_columns(ratings, movies)
    preprocessor = DataPreprocessor()
    preprocessor.fit_mappings(ratings, movies, schema)

    # 3 unique users: 1, 2, 3
    assert len(preprocessor.user_to_index) == 3
    # 4 unique movies: 10, 20, 30, 40
    assert len(preprocessor.movie_to_index) == 4

    # Bi-directional consistency
    for u, idx in preprocessor.user_to_index.items():
        assert preprocessor.index_to_user[idx] == u
    for m, idx in preprocessor.movie_to_index.items():
        assert preprocessor.index_to_movie[idx] == m


def test_transform_and_split(sample_raw_data):
    ratings, movies = sample_raw_data
    schema = detect_columns(ratings, movies)
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    preprocessor.fit_mappings(ratings, movies, schema)
    train_df, test_df = preprocessor.transform_and_split(ratings)

    assert len(train_df) + len(test_df) == len(ratings)
    assert "user_idx" in train_df.columns
    assert "movie_idx" in train_df.columns
    assert "rating" in train_df.columns
