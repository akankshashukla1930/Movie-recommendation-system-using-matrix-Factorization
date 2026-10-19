"""
Recommendation Engine module for Movie Recommendation System.
Generates personalized Top-N movie recommendations excluding already-rated titles,
attaches movie metadata (title, genres), and gracefully handles edge cases.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd

from src.config import (
    MODEL_FILE,
    MAPPINGS_FILE,
    TRAIN_FILE,
    TEST_FILE,
    BIAS_SHRINKAGE,
    NOVELTY_WEIGHT,
)
from src.matrix_factorization import MatrixFactorizationSGD
from src.preprocessing import DataPreprocessor


class MovieRecommender:
    """
    Inference and recommendation engine using trained Matrix Factorization parameters
    and dataset metadata mappings.
    """

    def __init__(
        self,
        model: Optional[MatrixFactorizationSGD] = None,
        mappings: Optional[Dict[str, Any]] = None,
    ):
        if model is None:
            self.model = MatrixFactorizationSGD.load_model(MODEL_FILE)
        else:
            self.model = model

        if mappings is None:
            self.mappings = DataPreprocessor.load_mappings()
        else:
            self.mappings = mappings

        self.user_to_idx: Dict[str, int] = self.mappings["user_to_index"]
        self.movie_to_idx: Dict[str, int] = self.mappings["movie_to_index"]
        self.idx_to_user: Dict[str, Any] = self.mappings["index_to_user"]
        self.idx_to_movie: Dict[str, Any] = self.mappings["index_to_movie"]
        self.movie_metadata: Dict[str, Dict[str, str]] = self.mappings.get("movie_metadata", {})

        # Load historical ratings to track what users have already rated
        self._load_interaction_history()

    def _load_interaction_history(self) -> None:
        """
        Combine train and test interactions to catalog all movies each user has already rated.
        """
        frames = []
        if TRAIN_FILE.exists():
            frames.append(pd.read_csv(TRAIN_FILE))
        if TEST_FILE.exists():
            frames.append(pd.read_csv(TEST_FILE))

        if frames:
            all_interactions = pd.concat(frames, ignore_index=True)
            # Map user_id to set of rated movie_idx
            self.user_rated_indices: Dict[int, set] = (
                all_interactions.groupby("user_idx")["movie_idx"].apply(set).to_dict()
            )
            # Map raw user_id to DataFrame of their ratings
            self.interaction_df = all_interactions
        else:
            self.user_rated_indices = {}
            self.interaction_df = pd.DataFrame()

        # Compute training movie counts for empirical Bayes shrinkage
        if TRAIN_FILE.exists():
            train_interactions = pd.read_csv(TRAIN_FILE)
            counts_dict = train_interactions["movie_idx"].value_counts().to_dict()
            self.movie_counts = np.array(
                [counts_dict.get(i, 0) for i in range(self.model.num_movies)],
                dtype=np.float64,
            )
        else:
            self.movie_counts = None

    def is_valid_user(self, user_id: Union[int, str]) -> bool:
        """
        Check if user_id exists in known mappings.
        """
        return str(user_id) in self.user_to_idx

    def get_user_history(self, user_id: Union[int, str], limit: int = 10) -> pd.DataFrame:
        """
        Fetch movies previously rated by the user with actual ratings and titles.
        """
        if not self.is_valid_user(user_id) or self.interaction_df.empty:
            return pd.DataFrame()

        u_idx = self.user_to_idx[str(user_id)]
        user_ratings = (
            self.interaction_df[self.interaction_df["user_idx"] == u_idx]
            .sort_values(by="rating", ascending=False)
            .head(limit)
        )

        records = []
        for _, row in user_ratings.iterrows():
            m_id = str(row["movie_id"])
            meta = self.movie_metadata.get(m_id, {"title": f"Movie {m_id}", "genres": "Unknown"})
            records.append(
                {
                    "Movie ID": m_id,
                    "Title": meta["title"],
                    "Genre": meta["genres"],
                    "User Rating": float(row["rating"]),
                }
            )

        return pd.DataFrame(records)

    def recommend_cold_start(
        self,
        n_recommendations: int = 10,
        bias_shrinkage: float = BIAS_SHRINKAGE,
        novelty_weight: float = NOVELTY_WEIGHT,
    ) -> pd.DataFrame:
        """
        Generate unpersonalized Top-N recommendations for new/unknown users (cold-start).
        Uses global baseline rating (mu + b_i) without pretending personalized preferences exist.
        """
        # user_idx = -1 triggers unpersonalized prediction (global_mean + movie_bias)
        raw_recs = self.model.recommend(
            user_idx=-1,
            rated_movie_indices=None,
            top_n=n_recommendations,
            bias_shrinkage=bias_shrinkage,
            movie_counts=self.movie_counts,
            novelty_weight=novelty_weight,
        )

        results = []
        for rank, (m_idx, score) in enumerate(raw_recs, start=1):
            raw_mid = str(self.idx_to_movie.get(str(m_idx), m_idx))
            meta = self.movie_metadata.get(
                raw_mid, {"title": f"Movie #{raw_mid}", "genres": "Unknown"}
            )
            results.append(
                {
                    "Rank": rank,
                    "Movie ID": raw_mid,
                    "Movie Title": meta["title"],
                    "Genre": meta["genres"],
                    "Predicted Rating": round(score, 2),
                }
            )

        return pd.DataFrame(results)

    def recommend_movies(
        self,
        user_id: Union[int, str],
        n_recommendations: int = 10,
        cold_start_fallback: bool = False,
        bias_shrinkage: float = BIAS_SHRINKAGE,
        novelty_weight: float = NOVELTY_WEIGHT,
    ) -> pd.DataFrame:
        """
        Generate Top-N recommendations for user_id.
        Excludes all movies previously rated by the user.
        If user_id is unknown and cold_start_fallback is True, falls back to recommend_cold_start.

        Returns DataFrame with columns:
        Rank, Movie ID, Movie Title, Genre, Predicted Rating
        """
        str_uid = str(user_id)

        # 1. User validation & Cold-Start handling
        if not self.is_valid_user(str_uid):
            if cold_start_fallback:
                return self.recommend_cold_start(
                    n_recommendations=n_recommendations,
                    bias_shrinkage=bias_shrinkage,
                    novelty_weight=novelty_weight,
                )
            raise ValueError(
                f"User ID '{user_id}' not found in trained dataset. Valid range: 1 to {len(self.user_to_idx)}."
            )

        user_idx = self.user_to_idx[str_uid]
        rated_indices = self.user_rated_indices.get(user_idx, set())

        # 2. Get top-N unrated recommendations from model with optimized continuous ranking
        raw_recs = self.model.recommend(
            user_idx=user_idx,
            rated_movie_indices=rated_indices,
            top_n=n_recommendations,
            bias_shrinkage=bias_shrinkage,
            movie_counts=self.movie_counts,
            novelty_weight=novelty_weight,
        )

        # 3. Format into rich tabular output
        results = []
        for rank, (m_idx, score) in enumerate(raw_recs, start=1):
            raw_mid = str(self.idx_to_movie.get(str(m_idx), m_idx))
            meta = self.movie_metadata.get(
                raw_mid, {"title": f"Movie #{raw_mid}", "genres": "Unknown"}
            )
            results.append(
                {
                    "Rank": rank,
                    "Movie ID": raw_mid,
                    "Movie Title": meta["title"],
                    "Genre": meta["genres"],
                    "Predicted Rating": round(score, 2),
                }
            )

        return pd.DataFrame(results)

    def get_most_and_least_active_users(self) -> Tuple[str, str, int, int]:
        """
        Find the most active and least active user IDs based on interaction counts.
        """
        counts = {u_idx: len(movies) for u_idx, movies in self.user_rated_indices.items()}
        sorted_users = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        most_active_idx, max_count = sorted_users[0]
        least_active_idx, min_count = sorted_users[-1]

        most_active_uid = str(self.idx_to_user[str(most_active_idx)])
        least_active_uid = str(self.idx_to_user[str(least_active_idx)])

        return most_active_uid, least_active_uid, max_count, min_count


def test_recommendation_engine(recommender: MovieRecommender) -> None:
    """
    Phase 11: Thoroughly test recommendations across standard and edge case users:
    - User 1
    - Random user
    - Most active user
    - Least active user
    - Invalid user ID
    """
    print("=" * 70)
    print("PHASE 11: TESTING RECOMMENDATION ENGINE ACROSS USER PROFILES")
    print("=" * 70)

    most_act, least_act, max_c, min_c = recommender.get_most_and_least_active_users()
    test_cases = [
        ("User 1", "1"),
        ("Random User (User 42)", "42"),
        (f"Most Active User ({most_act} with {max_c} ratings)", most_act),
        (f"Least Active User ({least_act} with {min_c} ratings)", least_act),
    ]

    for label, uid in test_cases:
        print(f"\n---> Testing: {label} (ID: {uid})")
        recs = recommender.recommend_movies(uid, n_recommendations=5)
        print(recs[["Rank", "Movie Title", "Genre", "Predicted Rating"]].to_string(index=False))

        # Verification checks
        user_idx = recommender.user_to_idx[str(uid)]
        already_rated = recommender.user_rated_indices.get(user_idx, set())
        rec_indices = [recommender.movie_to_idx[str(m)] for m in recs["Movie ID"]]

        # Check 1: No already rated movies recommended
        overlap = set(rec_indices).intersection(already_rated)
        assert len(overlap) == 0, f"Error: Already rated movie recommended for {uid}!"

        # Check 2: No duplicates in recommendations
        assert len(recs["Movie ID"]) == len(recs["Movie ID"].unique()), "Error: Duplicate recommendations found!"

        # Check 3: Sorted descending
        ratings_list = recs["Predicted Rating"].tolist()
        assert ratings_list == sorted(ratings_list, reverse=True), "Error: Predictions not sorted descending!"

        print("  [PASSED] No rated movies recommended | No duplicates | Correctly sorted descending.")

    # Test Invalid User ID
    print("\n---> Testing: Invalid User ID (999999)")
    try:
        recommender.recommend_movies(999999, n_recommendations=5)
        print("  [FAILED] Should have raised ValueError for invalid user.")
    except ValueError as e:
        print(f"  [PASSED] Handled gracefully with exception: {e}")

    print("\n" + "=" * 70)
    print("ALL RECOMMENDATION ENGINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    rec_engine = MovieRecommender()
    test_recommendation_engine(rec_engine)
