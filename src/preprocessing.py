"""
Preprocessing and Data Cleaning module.
Performs data validation, duplicate resolution, ID encoding, 
and leakage-free train/test splitting for Matrix Factorization.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    MAPPINGS_FILE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    MIN_RATING,
    MAX_RATING,
)
from src.data_loader import load_raw_data


class DataPreprocessor:
    """
    Handles data validation, cleaning, encoding, and train/test splitting.
    """

    def __init__(self, random_state: int = RANDOM_STATE, test_size: float = TEST_SIZE):
        self.random_state = random_state
        self.test_size = test_size

        self.user_to_index: Dict[Any, int] = {}
        self.movie_to_index: Dict[Any, int] = {}
        self.index_to_user: Dict[int, Any] = {}
        self.index_to_movie: Dict[int, Any] = {}
        self.movie_metadata: Dict[Any, Dict[str, str]] = {}
        self.schema: Dict[str, str] = {}
        self.cleaning_report: Dict[str, Any] = {}

    def clean_ratings(self, ratings_df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        """
        Validate ratings data types, remove missing values, filter rating bounds,
        and resolve duplicate user-movie interactions.
        """
        u_col = schema["user_col"]
        m_col = schema["movie_col"]
        r_col = schema["rating_col"]
        t_col = schema.get("timestamp_col")

        initial_count = len(ratings_df)

        # 1. Check & drop nulls
        df = ratings_df[[c for c in [u_col, m_col, r_col, t_col] if c is not None]].dropna().copy()
        null_dropped = initial_count - len(df)

        # 2. Enforce types
        df[r_col] = pd.to_numeric(df[r_col], errors="coerce")
        df = df.dropna(subset=[r_col])

        # 3. Filter valid rating bounds
        valid_mask = (df[r_col] >= MIN_RATING) & (df[r_col] <= MAX_RATING)
        out_of_bounds = int((~valid_mask).sum())
        df = df[valid_mask].copy()

        # 4. Handle duplicate user-movie interactions
        # Decision: If a user rated the same movie multiple times, keep the latest rating based on timestamp.
        # If timestamp is unavailable, aggregate by taking the mean rating.
        exact_duplicates = int(df.duplicated(subset=[u_col, m_col, r_col]).sum())
        user_movie_dups = int(df.duplicated(subset=[u_col, m_col], keep=False).sum())

        if t_col and t_col in df.columns:
            df = df.sort_values(by=t_col).drop_duplicates(subset=[u_col, m_col], keep="last")
        else:
            df = df.groupby([u_col, m_col], as_index=False).agg({r_col: "mean"})

        final_count = len(df)

        self.cleaning_report = {
            "initial_rows": initial_count,
            "null_rows_removed": null_dropped,
            "out_of_bounds_ratings": out_of_bounds,
            "exact_duplicate_rows": exact_duplicates,
            "duplicate_user_movie_pairs": user_movie_dups,
            "resolved_clean_rows": final_count,
        }

        return df

    def fit_mappings(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, schema: Dict[str, str]) -> None:
        """
        Create 0-indexed continuous mappings for users and movies.
        Extracts movie metadata (titles, genres).
        """
        self.schema = schema
        u_col = schema["user_col"]
        m_col = schema["movie_col"]
        m_col_meta = schema.get("movie_col_movies", m_col)
        title_col = schema.get("title_col")
        genre_col = schema.get("genre_col")

        # Get unique user IDs and movie IDs
        unique_users = sorted(ratings_df[u_col].unique().tolist())
        # Also include any movie IDs present in movies_df to ensure full coverage
        unique_movies = sorted(
            list(set(ratings_df[m_col].unique().tolist()).union(set(movies_df[m_col_meta].unique().tolist())))
        )

        self.user_to_index = {uid: idx for idx, uid in enumerate(unique_users)}
        self.index_to_user = {idx: uid for idx, uid in enumerate(unique_users)}

        self.movie_to_index = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.index_to_movie = {idx: mid for idx, mid in enumerate(unique_movies)}

        # Build movie metadata lookup
        self.movie_metadata = {}
        for _, row in movies_df.iterrows():
            m_id = row[m_col_meta]
            title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else f"Movie {m_id}"
            genre = str(row[genre_col]) if genre_col and pd.notna(row[genre_col]) else "Unknown"
            self.movie_metadata[m_id] = {"title": title, "genres": genre}

    def transform_and_split(
        self, clean_ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply ID encoding to ratings and perform an 80/20 train/test split.
        Returns train_df and test_df with columns: user_idx, movie_idx, rating, raw_user_id, raw_movie_id.
        """
        u_col = self.schema["user_col"]
        m_col = self.schema["movie_col"]
        r_col = self.schema["rating_col"]

        encoded_df = pd.DataFrame(
            {
                "user_idx": clean_ratings_df[u_col].map(self.user_to_index),
                "movie_idx": clean_ratings_df[m_col].map(self.movie_to_index),
                "rating": clean_ratings_df[r_col].astype(float),
                "user_id": clean_ratings_df[u_col],
                "movie_id": clean_ratings_df[m_col],
            }
        )

        # 80/20 Split with random_state
        train_df, test_df = train_test_split(
            encoded_df, test_size=self.test_size, random_state=self.random_state
        )

        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def save_artifacts(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Save train.csv, test.csv, and mappings.json to data/processed.
        """
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(TRAIN_FILE, index=False)
        test_df.to_csv(TEST_FILE, index=False)

        # Convert keys to string for JSON serialization
        mappings_data = {
            "user_to_index": {str(k): v for k, v in self.user_to_index.items()},
            "movie_to_index": {str(k): v for k, v in self.movie_to_index.items()},
            "index_to_user": {str(k): v for k, v in self.index_to_user.items()},
            "index_to_movie": {str(k): v for k, v in self.index_to_movie.items()},
            "movie_metadata": {str(k): v for k, v in self.movie_metadata.items()},
            "cleaning_report": self.cleaning_report,
            "schema": self.schema,
            "num_users": len(self.user_to_index),
            "num_movies": len(self.movie_to_index),
        }

        with open(MAPPINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(mappings_data, f, indent=2)

        print(f"Saved train set ({len(train_df)} rows) -> {TRAIN_FILE}")
        print(f"Saved test set ({len(test_df)} rows) -> {TEST_FILE}")
        print(f"Saved mappings and metadata -> {MAPPINGS_FILE}")

    @staticmethod
    def load_mappings() -> Dict[str, Any]:
        """
        Load processed mappings from JSON.
        """
        if not MAPPINGS_FILE.exists():
            raise FileNotFoundError(f"Mappings file {MAPPINGS_FILE} not found. Run preprocessing first.")
        with open(MAPPINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)


def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run complete raw data ingestion, cleaning, mapping, and splitting pipeline.
    """
    print("--- Running Preprocessing Pipeline ---")
    ratings_raw, movies_raw, schema = load_raw_data()
    preprocessor = DataPreprocessor()
    
    clean_ratings = preprocessor.clean_ratings(ratings_raw, schema)
    preprocessor.fit_mappings(clean_ratings, movies_raw, schema)
    train_df, test_df = preprocessor.transform_and_split(clean_ratings)
    preprocessor.save_artifacts(train_df, test_df)

    return train_df, test_df, preprocessor.cleaning_report


if __name__ == "__main__":
    t_df, te_df, report = run_preprocessing_pipeline()
    print("Preprocessing report:", report)
