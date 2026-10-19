"""
Chronological ("future prediction") train/test split.

The default pipeline in preprocessing.py splits ratings at random, which lets the
model train on a user's 2018 ratings while predicting their 2005 ones. That
measures matrix completion -- filling gaps in a known rating matrix -- not
forecasting. This module instead orders each user's ratings by timestamp and
holds out their most recent ones, so the model is only ever shown a user's past
when predicting their future.

The user/movie index mappings are reused from the standard pipeline so both
evaluations address the same catalog.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from src.config import (
    TEST_SIZE,
    TRAIN_TEMPORAL_FILE,
    TEST_TEMPORAL_FILE,
    PROCESSED_DATA_DIR,
)
from src.data_loader import load_raw_data
from src.preprocessing import DataPreprocessor


def split_user_chronologically(
    group: pd.DataFrame, time_col: str, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split one user's ratings by time: earliest (1 - test_size) to train, latest to test.

    Every user keeps at least one training and one test rating, so no user is
    dropped from either side of the evaluation.
    """
    ordered = group.sort_values(time_col, kind="mergesort")
    n = len(ordered)
    n_train = int(round(n * (1.0 - test_size)))
    n_train = max(1, min(n_train, n - 1))
    return ordered.iloc[:n_train], ordered.iloc[n_train:]


def build_temporal_split(test_size: float = TEST_SIZE) -> Dict[str, Any]:
    """
    Build and persist a chronological split, returning a report of its properties.
    """
    ratings_raw, movies_raw, schema = load_raw_data()
    time_col = schema.get("timestamp_col")
    if not time_col:
        raise ValueError(
            "A timestamp column is required for a temporal split but none was found."
        )

    pre = DataPreprocessor()
    clean = pre.clean_ratings(ratings_raw, schema)
    pre.fit_mappings(clean, movies_raw, schema)

    u_col, m_col, r_col = schema["user_col"], schema["movie_col"], schema["rating_col"]

    train_parts, test_parts = [], []
    for _, group in clean.groupby(u_col, sort=False):
        tr, te = split_user_chronologically(group, time_col, test_size)
        train_parts.append(tr)
        test_parts.append(te)

    raw_train = pd.concat(train_parts, ignore_index=True)
    raw_test = pd.concat(test_parts, ignore_index=True)

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "user_idx": df[u_col].map(pre.user_to_index),
                "movie_idx": df[m_col].map(pre.movie_to_index),
                "rating": df[r_col].astype(float),
                "user_id": df[u_col],
                "movie_id": df[m_col],
            }
        ).reset_index(drop=True)

    train_df, test_df = encode(raw_train), encode(raw_test)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_TEMPORAL_FILE, index=False)
    test_df.to_csv(TEST_TEMPORAL_FILE, index=False)

    # A temporal split surfaces cold-start cases that a random split hides: items
    # whose first-ever rating falls after the cutoff have no learned factors.
    seen_movies = set(train_df["movie_idx"])
    unseen = int((~test_df["movie_idx"].isin(seen_movies)).sum())

    report = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "users": int(clean[u_col].nunique()),
        "test_ratings_on_unseen_movies": unseen,
        "unseen_movie_percentage": round(unseen / len(test_df) * 100.0, 2),
        "train_period": [
            str(pd.to_datetime(raw_train[time_col].min(), unit="s").date()),
            str(pd.to_datetime(raw_train[time_col].max(), unit="s").date()),
        ],
        "test_period": [
            str(pd.to_datetime(raw_test[time_col].min(), unit="s").date()),
            str(pd.to_datetime(raw_test[time_col].max(), unit="s").date()),
        ],
        "num_users": pre.num_users if hasattr(pre, "num_users") else len(pre.user_to_index),
        "num_movies": len(pre.movie_to_index),
    }
    return report


def load_temporal_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted chronological split, building it if absent."""
    if not (TRAIN_TEMPORAL_FILE.exists() and TEST_TEMPORAL_FILE.exists()):
        build_temporal_split()
    return pd.read_csv(TRAIN_TEMPORAL_FILE), pd.read_csv(TEST_TEMPORAL_FILE)


if __name__ == "__main__":
    print(build_temporal_split())
