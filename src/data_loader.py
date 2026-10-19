"""
Data Loader module for Movie Recommendation System.
Automatically identifies dataset schemas, loads ratings and movie metadata,
and generates summary diagnostics.
"""

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np

from src.config import RAW_DATA_DIR, RATINGS_FILE, MOVIES_FILE


def detect_columns(ratings_df: pd.DataFrame, movies_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Automatically detect the relevant column names across ratings and movies DataFrames
    without assuming exact casing or naming conventions.
    """
    r_cols = {c.lower().replace("_", ""): c for c in ratings_df.columns}
    
    # Identify user id column
    user_candidates = ["userid", "user", "uid", "user_id"]
    user_col = next((r_cols[k] for k in user_candidates if k in r_cols), ratings_df.columns[0])

    # Identify movie id column
    movie_candidates = ["movieid", "itemid", "item", "movie", "movie_id"]
    movie_col = next((r_cols[k] for k in movie_candidates if k in r_cols), ratings_df.columns[1])

    # Identify rating column
    rating_candidates = ["rating", "ratings", "rate", "score"]
    rating_col = next((r_cols[k] for k in rating_candidates if k in r_cols), ratings_df.columns[2])

    # Identify timestamp column
    timestamp_candidates = ["timestamp", "time", "date"]
    timestamp_col = next((r_cols[k] for k in timestamp_candidates if k in r_cols), None)

    title_col = None
    genre_col = None

    if movies_df is not None:
        m_cols = {c.lower().replace("_", ""): c for c in movies_df.columns}
        title_candidates = ["title", "movietitle", "name"]
        title_col = next((m_cols[k] for k in title_candidates if k in m_cols), None)

        genre_candidates = ["genres", "genre", "category", "categories"]
        genre_col = next((m_cols[k] for k in genre_candidates if k in m_cols), None)

        # Confirm movie id column in movies dataframe
        movie_col_movies = next((m_cols[k] for k in movie_candidates if k in m_cols), movies_df.columns[0])
    else:
        movie_col_movies = movie_col

    return {
        "user_col": user_col,
        "movie_col": movie_col,
        "movie_col_movies": movie_col_movies,
        "rating_col": rating_col,
        "timestamp_col": timestamp_col,
        "title_col": title_col,
        "genre_col": genre_col,
    }


def load_raw_data(data_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Safely load raw ratings and movies datasets from data_dir.
    Handles CSV files or MovieLens 100K pipe-separated u.data / u.item files.
    """
    raw_dir = Path(data_dir) if data_dir else RAW_DATA_DIR

    ratings_path = raw_dir / "ratings.csv"
    movies_path = raw_dir / "movies.csv"

    # Check for ml-latest-small subdirectory if top-level missing
    if not ratings_path.exists():
        sub_ratings = list(raw_dir.glob("**/ratings.csv"))
        if sub_ratings:
            ratings_path = sub_ratings[0]
            
    if not movies_path.exists():
        sub_movies = list(raw_dir.glob("**/movies.csv"))
        if sub_movies:
            movies_path = sub_movies[0]

    # Handle standard MovieLens 100K format (u.data and u.item) if present
    u_data = raw_dir / "u.data"
    u_item = raw_dir / "u.item"
    if not ratings_path.exists() and u_data.exists():
        ratings_df = pd.read_csv(u_data, sep="\t", names=["userId", "movieId", "rating", "timestamp"])
        if u_item.exists():
            movies_df = pd.read_csv(u_item, sep="|", encoding="latin-1", header=None)
            movies_df = movies_df[[0, 1]]
            movies_df.columns = ["movieId", "title"]
            movies_df["genres"] = "Unknown"
        else:
            movies_df = pd.DataFrame(columns=["movieId", "title", "genres"])
    else:
        if not ratings_path.exists():
            raise FileNotFoundError(f"Ratings file could not be found in {raw_dir}")
        ratings_df = pd.read_csv(ratings_path)
        
        if movies_path.exists():
            movies_df = pd.read_csv(movies_path)
        else:
            movies_df = pd.DataFrame(columns=["movieId", "title", "genres"])

    schema = detect_columns(ratings_df, movies_df)
    return ratings_df, movies_df, schema


def summarize_dataset(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, schema: Dict[str, str]) -> Dict[str, Any]:
    """
    Compute key summary statistics and diagnostics for the loaded datasets.
    """
    user_col = schema["user_col"]
    movie_col = schema["movie_col"]
    rating_col = schema["rating_col"]

    total_ratings = len(ratings_df)
    n_users = int(ratings_df[user_col].nunique())
    n_movies = int(ratings_df[movie_col].nunique())
    min_rating = float(ratings_df[rating_col].min())
    max_rating = float(ratings_df[rating_col].max())
    mean_rating = float(ratings_df[rating_col].mean())
    median_rating = float(ratings_df[rating_col].median())
    std_rating = float(ratings_df[rating_col].std())

    possible_interactions = n_users * n_movies
    sparsity = (1.0 - (total_ratings / possible_interactions)) * 100.0 if possible_interactions > 0 else 0.0

    summary = {
        "total_ratings": total_ratings,
        "unique_users": n_users,
        "unique_movies": n_movies,
        "min_rating": min_rating,
        "max_rating": max_rating,
        "mean_rating": round(mean_rating, 4),
        "median_rating": round(median_rating, 4),
        "std_rating": round(std_rating, 4),
        "sparsity_percentage": round(sparsity, 4),
        "ratings_shape": list(ratings_df.shape),
        "movies_shape": list(movies_df.shape),
        "schema": schema,
    }
    return summary


def print_dataset_info(ratings_df: pd.DataFrame, movies_df: pd.DataFrame, schema: Dict[str, str]) -> None:
    """
    Print formatted summary diagnostics to console.
    """
    summary = summarize_dataset(ratings_df, movies_df, schema)
    print("=" * 60)
    print("DATASET DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Total Ratings Recorded: {summary['total_ratings']:,}")
    print(f"Unique Users:           {summary['unique_users']:,}")
    print(f"Unique Movies Rated:    {summary['unique_movies']:,}")
    print(f"Rating Scale Range:     {summary['min_rating']} to {summary['max_rating']}")
    print(f"Mean Rating:            {summary['mean_rating']:.2f} (Std: {summary['std_rating']:.2f})")
    print(f"Matrix Sparsity:        {summary['sparsity_percentage']:.2f}%")
    print(f"Ratings DF Shape:       {summary['ratings_shape']}")
    print(f"Movies DF Shape:        {summary['movies_shape']}")
    print("-" * 60)
    print("Detected Column Schema:")
    for k, v in schema.items():
        print(f"  - {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    r_df, m_df, s = load_raw_data()
    print_dataset_info(r_df, m_df, s)
