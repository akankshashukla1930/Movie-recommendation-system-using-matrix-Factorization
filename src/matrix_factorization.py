"""
Matrix Factorization with Stochastic Gradient Descent (SGD) implemented from scratch in NumPy.
Includes latent factors for users and movies, user/movie bias terms, global baseline mean,
and L2 regularization.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    N_FACTORS,
    LEARNING_RATE,
    REGULARIZATION,
    EPOCHS,
    RANDOM_STATE,
    MIN_RATING,
    MAX_RATING,
    INIT_MEAN,
    INIT_STD,
    MODEL_FILE,
    FIGURES_DIR,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
)


class MatrixFactorizationSGD:
    """
    Collaborative Filtering Matrix Factorization trained via Stochastic Gradient Descent.

    Formula:
        r_hat(u, i) = mu + b_u + b_i + P_u . Q_i
    """

    def __init__(
        self,
        n_factors: int = N_FACTORS,
        learning_rate: float = LEARNING_RATE,
        regularization: float = REGULARIZATION,
        epochs: int = EPOCHS,
        random_state: int = RANDOM_STATE,
        min_rating: float = MIN_RATING,
        max_rating: float = MAX_RATING,
        patience: int = EARLY_STOPPING_PATIENCE,
        min_delta: float = MIN_DELTA,
    ):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.random_state = random_state
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.patience = patience
        self.min_delta = min_delta

        # Learned parameters
        self.global_mean: float = 0.0
        self.user_bias: Optional[np.ndarray] = None
        self.movie_bias: Optional[np.ndarray] = None
        self.user_factors: Optional[np.ndarray] = None
        self.movie_factors: Optional[np.ndarray] = None

        # Dimensions
        self.num_users: int = 0
        self.num_movies: int = 0

        # Training history
        self.rmse_history: List[float] = []
        self.val_rmse_history: List[float] = []
        self.best_epoch: Optional[int] = None
        self.best_val_rmse: Optional[float] = None
        self.stopped_early: bool = False

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        num_users: Optional[int] = None,
        num_movies: Optional[int] = None,
        verbose: bool = True,
    ) -> "MatrixFactorizationSGD":
        """
        Fit latent factors and biases using SGD over the training interactions.
        """
        rng = np.random.default_rng(self.random_state)

        user_indices = train_df["user_idx"].to_numpy(dtype=np.int32)
        movie_indices = train_df["movie_idx"].to_numpy(dtype=np.int32)
        ratings = train_df["rating"].to_numpy(dtype=np.float64)

        self.num_users = int(max(user_indices.max() + 1, self.num_users, num_users or 0))
        self.num_movies = int(max(movie_indices.max() + 1, self.num_movies, num_movies or 0))

        # 1. Global Mean
        self.global_mean = float(np.mean(ratings))

        # 2. Initialize Biases to zero
        self.user_bias = np.zeros(self.num_users, dtype=np.float64)
        self.movie_bias = np.zeros(self.num_movies, dtype=np.float64)

        # 3. Initialize Latent Matrices normally
        self.user_factors = rng.normal(
            INIT_MEAN, INIT_STD, size=(self.num_users, self.n_factors)
        ).astype(np.float64)
        self.movie_factors = rng.normal(
            INIT_MEAN, INIT_STD, size=(self.num_movies, self.n_factors)
        ).astype(np.float64)

        n_samples = len(ratings)
        self.rmse_history = []
        self.val_rmse_history = []
        self.best_epoch = None
        self.best_val_rmse = None
        self.stopped_early = False

        # Early stopping bookkeeping
        best_snapshot: Optional[Dict[str, Any]] = None
        epochs_without_improvement = 0

        val_u = val_i = val_r = None
        if val_df is not None:
            val_u = val_df["user_idx"].to_numpy(dtype=np.int32)
            val_i = val_df["movie_idx"].to_numpy(dtype=np.int32)
            val_r = val_df["rating"].to_numpy(dtype=np.float64)

        if verbose:
            print("=" * 65)
            print(f"TRAINING MATRIX FACTORIZATION MODEL (SGD)")
            print(f"Factors: {self.n_factors} | LR: {self.learning_rate} | Reg: {self.regularization} | Epochs: {self.epochs}")
            print(f"Users: {self.num_users:,} | Movies: {self.num_movies:,} | Samples: {n_samples:,}")
            print("=" * 65)

        sample_order = np.arange(n_samples)

        for epoch in range(1, self.epochs + 1):
            # Shuffle dataset every epoch for true SGD
            rng.shuffle(sample_order)

            # Optimization Loop
            for idx in sample_order:
                u = user_indices[idx]
                i = movie_indices[idx]
                r = ratings[idx]

                # Current prediction
                pred = (
                    self.global_mean
                    + self.user_bias[u]
                    + self.movie_bias[i]
                    + np.dot(self.user_factors[u], self.movie_factors[i])
                )
                err = r - pred

                # SGD updates with L2 regularization
                self.user_bias[u] += self.learning_rate * (
                    err - self.regularization * self.user_bias[u]
                )
                self.movie_bias[i] += self.learning_rate * (
                    err - self.regularization * self.movie_bias[i]
                )

                u_factors_temp = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (
                    err * self.movie_factors[i]
                    - self.regularization * self.user_factors[u]
                )
                self.movie_factors[i] += self.learning_rate * (
                    err * u_factors_temp
                    - self.regularization * self.movie_factors[i]
                )

            # Calculate train epoch RMSE
            train_rmse = self.calculate_rmse(user_indices, movie_indices, ratings)
            self.rmse_history.append(train_rmse)

            val_str = ""
            if val_df is not None:
                val_rmse = self.calculate_rmse(val_u, val_i, val_r)
                self.val_rmse_history.append(val_rmse)
                val_str = f" | Val RMSE: {val_rmse:.4f}"

                # Track the best epoch and snapshot its weights
                if self.best_val_rmse is None or val_rmse < self.best_val_rmse - self.min_delta:
                    self.best_val_rmse = val_rmse
                    self.best_epoch = epoch
                    epochs_without_improvement = 0
                    best_snapshot = {
                        "user_bias": self.user_bias.copy(),
                        "movie_bias": self.movie_bias.copy(),
                        "user_factors": self.user_factors.copy(),
                        "movie_factors": self.movie_factors.copy(),
                    }
                    val_str += "  <- best"
                else:
                    epochs_without_improvement += 1

            if verbose:
                print(f"Epoch {epoch:3d}/{self.epochs} | Train RMSE: {train_rmse:.4f}{val_str}")

            if val_df is not None and epochs_without_improvement >= self.patience:
                self.stopped_early = True
                if verbose:
                    print(
                        f"\nEarly stopping: no validation improvement for {self.patience} "
                        f"consecutive epochs (best was epoch {self.best_epoch})."
                    )
                break

        # Roll back to the best-validation weights instead of keeping the last,
        # overfit, epoch.
        if best_snapshot is not None and self.best_epoch != len(self.rmse_history):
            self.user_bias = best_snapshot["user_bias"]
            self.movie_bias = best_snapshot["movie_bias"]
            self.user_factors = best_snapshot["user_factors"]
            self.movie_factors = best_snapshot["movie_factors"]
            if verbose:
                print(
                    f"Restored weights from epoch {self.best_epoch} "
                    f"(Val RMSE: {self.best_val_rmse:.4f})."
                )

        if verbose:
            print("=" * 65)

        return self

    def predict(self, user_idx: int, movie_idx: int) -> float:
        """
        Predict rating for a specific user and movie index.
        Gracefully handles out-of-bounds / unseen indices.
        """
        pred = self.global_mean

        known_user = (
            self.user_bias is not None
            and 0 <= user_idx < len(self.user_bias)
        )
        known_movie = (
            self.movie_bias is not None
            and 0 <= movie_idx < len(self.movie_bias)
        )

        if known_user:
            pred += self.user_bias[user_idx]
        if known_movie:
            pred += self.movie_bias[movie_idx]

        if known_user and known_movie:
            pred += float(np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx]))

        # Clip predictions to rating scale boundaries
        return float(np.clip(pred, self.min_rating, self.max_rating))

    def predict_all_movies(
        self,
        user_idx: int,
        unclipped: bool = False,
        bias_shrinkage: float = 0.0,
        movie_counts: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Vectorized prediction for a given user across ALL known movies.
        Formula: r_u = mu + b_u + b_all + (Q . P_u)
        If unclipped is False, clips predictions to [min_rating, max_rating].
        If bias_shrinkage > 0 and movie_counts is provided, applies Bayesian damping
        to movie biases: b_i * (n_i / (n_i + beta)) to mitigate noise on low-interaction movies.
        """
        if self.user_bias is None or self.movie_bias is None:
            raise ValueError("Model has not been trained yet.")

        known_user = 0 <= user_idx < len(self.user_bias)
        u_bias = self.user_bias[user_idx] if known_user else 0.0

        if known_user:
            u_latent = self.user_factors[user_idx]
            latent_scores = np.dot(self.movie_factors, u_latent)
        else:
            latent_scores = np.zeros(self.num_movies, dtype=np.float64)

        if bias_shrinkage > 0.0 and movie_counts is not None and len(movie_counts) == self.num_movies:
            shrinkage = movie_counts / (movie_counts + bias_shrinkage)
            eff_movie_bias = self.movie_bias * shrinkage
        else:
            eff_movie_bias = self.movie_bias

        all_predictions = self.global_mean + u_bias + eff_movie_bias + latent_scores
        if unclipped:
            return all_predictions
        return np.clip(all_predictions, self.min_rating, self.max_rating)

    def calculate_rmse(
        self,
        user_indices: np.ndarray,
        movie_indices: np.ndarray,
        actual_ratings: np.ndarray,
    ) -> float:
        """
        Fast vectorized RMSE computation for a set of interactions.
        """
        preds = np.empty(len(actual_ratings), dtype=np.float64)
        for i in range(len(actual_ratings)):
            preds[i] = self.predict(user_indices[i], movie_indices[i])

        mse = np.mean((actual_ratings - preds) ** 2)
        return float(np.sqrt(mse))

    def recommend(
        self,
        user_idx: int,
        rated_movie_indices: Optional[set] = None,
        top_n: int = 10,
        bias_shrinkage: float = 0.0,
        movie_counts: Optional[np.ndarray] = None,
        novelty_weight: float = 0.0,
    ) -> List[Tuple[int, float]]:
        """
        Recommend Top-N movies for user_idx, strictly excluding movies in rated_movie_indices.
        Ranks candidates using continuous unclipped scores to prevent tie-breaking artifacts,
        and clips final returned predicted ratings to [min_rating, max_rating].

        If novelty_weight > 0 and movie_counts is provided, ranking scores are
        discounted by each movie's popularity. Without it every user's list is
        dominated by the same few blockbusters, which keeps catalog coverage low.
        The returned predicted ratings are unaffected -- the penalty only reorders
        candidates.

        Returns list of tuples: [(movie_idx, predicted_rating), ...]
        """
        # Get unclipped ranking scores
        ranking_scores = self.predict_all_movies(
            user_idx,
            unclipped=True,
            bias_shrinkage=bias_shrinkage,
            movie_counts=movie_counts,
        )

        # Keep the true predicted ratings for display, rank on the penalised copy.
        display_scores = ranking_scores
        if novelty_weight > 0.0 and movie_counts is not None and len(movie_counts) == self.num_movies:
            popularity = np.log1p(movie_counts)
            max_pop = popularity.max()
            if max_pop > 0:
                ranking_scores = ranking_scores - novelty_weight * (popularity / max_pop)

        # Mask already rated movies
        if rated_movie_indices:
            valid_indices = [
                i for i in range(len(ranking_scores)) if i not in rated_movie_indices
            ]
        else:
            valid_indices = list(range(len(ranking_scores)))

        valid_scores = ranking_scores[valid_indices]
        valid_display = display_scores[valid_indices]

        # Get top-N indices based on continuous ranking scores
        if len(valid_scores) <= top_n:
            top_order = np.argsort(-valid_scores)
        else:
            top_order = np.argpartition(-valid_scores, top_n)[:top_n]
            top_order = top_order[np.argsort(-valid_scores[top_order])]

        # Clip returned predicted ratings to [min_rating, max_rating] for standard scale display
        recommendations = [
            (
                int(valid_indices[idx]),
                float(np.clip(valid_display[idx], self.min_rating, self.max_rating)),
            )
            for idx in top_order
        ]
        return recommendations

    def plot_learning_curve(self, save_path: Optional[Path] = None) -> Path:
        """
        Plot training and validation RMSE per epoch, marking the best epoch so
        any divergence between the two curves is visible at a glance.
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        epochs_range = list(range(1, len(self.rmse_history) + 1))

        ax.plot(
            epochs_range,
            self.rmse_history,
            marker="o",
            color="#1f77b4",
            linewidth=2.5,
            markersize=5,
            label="Training RMSE",
        )

        if self.val_rmse_history:
            ax.plot(
                epochs_range[: len(self.val_rmse_history)],
                self.val_rmse_history,
                marker="s",
                color="#d62728",
                linewidth=2.5,
                markersize=5,
                label="Validation RMSE",
            )

            if self.best_epoch is not None:
                ax.axvline(
                    self.best_epoch,
                    color="#2ca02c",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"Best epoch ({self.best_epoch})",
                )
                ax.annotate(
                    f"Best Val RMSE: {self.best_val_rmse:.4f}",
                    xy=(self.best_epoch, self.best_val_rmse),
                    xytext=(20, 25),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                    fontweight="bold",
                )

        ax.set_title("Matrix Factorization (SGD) Learning Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Root Mean Squared Error (RMSE)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(frameon=True)

        plt.tight_layout()
        out = save_path if save_path else (FIGURES_DIR / "06_learning_curve.png")
        plt.savefig(out, dpi=300)
        plt.close()
        return out

    def save_model(self, file_path: Optional[Path] = None) -> Path:
        """
        Serialize all trained parameters and model state using joblib.
        """
        path = Path(file_path) if file_path else MODEL_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "n_factors": self.n_factors,
            "learning_rate": self.learning_rate,
            "regularization": self.regularization,
            "epochs": self.epochs,
            "random_state": self.random_state,
            "global_mean": self.global_mean,
            "user_bias": self.user_bias,
            "movie_bias": self.movie_bias,
            "user_factors": self.user_factors,
            "movie_factors": self.movie_factors,
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "rmse_history": self.rmse_history,
            "val_rmse_history": self.val_rmse_history,
            "best_epoch": self.best_epoch,
            "best_val_rmse": self.best_val_rmse,
            "stopped_early": self.stopped_early,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "min_rating": self.min_rating,
            "max_rating": self.max_rating,
        }

        joblib.dump(state, path)
        print(f"Model successfully serialized to: {path}")
        return path

    @classmethod
    def load_model(cls, file_path: Optional[Path] = None) -> "MatrixFactorizationSGD":
        """
        Load trained model instance from disk.
        """
        path = Path(file_path) if file_path else MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"Model file {path} not found. Please train first.")

        state = joblib.load(path)
        instance = cls(
            n_factors=state["n_factors"],
            learning_rate=state["learning_rate"],
            regularization=state["regularization"],
            epochs=state["epochs"],
            random_state=state["random_state"],
            min_rating=state.get("min_rating", MIN_RATING),
            max_rating=state.get("max_rating", MAX_RATING),
            patience=state.get("patience", EARLY_STOPPING_PATIENCE),
            min_delta=state.get("min_delta", MIN_DELTA),
        )

        instance.global_mean = state["global_mean"]
        instance.user_bias = state["user_bias"]
        instance.movie_bias = state["movie_bias"]
        instance.user_factors = state["user_factors"]
        instance.movie_factors = state["movie_factors"]
        instance.num_users = state["num_users"]
        instance.num_movies = state["num_movies"]
        instance.rmse_history = state.get("rmse_history", [])
        instance.val_rmse_history = state.get("val_rmse_history", [])
        instance.best_epoch = state.get("best_epoch")
        instance.best_val_rmse = state.get("best_val_rmse")
        instance.stopped_early = state.get("stopped_early", False)

        return instance
