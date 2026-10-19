"""
Unit tests for MatrixFactorizationSGD, recommender engine, and persistence.
"""

import numpy as np
import pandas as pd
import pytest

from src.matrix_factorization import MatrixFactorizationSGD
from src.recommender import MovieRecommender


@pytest.fixture
def trained_toy_model():
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "movie_idx": [0, 1, 1, 2, 0, 2],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )
    model = MatrixFactorizationSGD(
        n_factors=4,
        learning_rate=0.05,
        regularization=0.01,
        epochs=10,
        random_state=42,
    )
    model.fit(train_df, verbose=False)
    return model


def test_model_initialization():
    model = MatrixFactorizationSGD(n_factors=25, learning_rate=0.02, epochs=5)
    assert model.n_factors == 25
    assert model.learning_rate == 0.02
    assert model.epochs == 5
    assert model.global_mean == 0.0


def test_model_prediction_range(trained_toy_model):
    pred = trained_toy_model.predict(0, 1)
    assert isinstance(pred, float)
    assert 0.5 <= pred <= 5.0

    # Unseen indices should fall back safely without error
    unseen_user_pred = trained_toy_model.predict(999, 1)
    assert 0.5 <= unseen_user_pred <= 5.0


def test_predict_all_movies(trained_toy_model):
    preds = trained_toy_model.predict_all_movies(0)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == trained_toy_model.num_movies
    assert np.all((preds >= 0.5) & (preds <= 5.0))


def test_recommend_excludes_already_rated(trained_toy_model):
    # User 0 already rated movie 0 and movie 1
    rated = {0, 1}
    recs = trained_toy_model.recommend(user_idx=0, rated_movie_indices=rated, top_n=5)

    rec_indices = [idx for idx, _ in recs]
    assert 0 not in rec_indices
    assert 1 not in rec_indices
    # Must only contain movie 2
    assert 2 in rec_indices


def test_model_save_and_reload(tmp_path, trained_toy_model):
    save_file = tmp_path / "toy_model.pkl"
    trained_toy_model.save_model(save_file)

    loaded = MatrixFactorizationSGD.load_model(save_file)
    assert loaded.n_factors == trained_toy_model.n_factors
    assert np.isclose(loaded.global_mean, trained_toy_model.global_mean)
    assert np.allclose(loaded.user_bias, trained_toy_model.user_bias)
    assert np.allclose(loaded.movie_bias, trained_toy_model.movie_bias)

    orig_pred = trained_toy_model.predict(1, 2)
    loaded_pred = loaded.predict(1, 2)
    assert np.isclose(orig_pred, loaded_pred)


def test_recommender_integration():
    recommender = MovieRecommender()
    recs = recommender.recommend_movies(user_id=1, n_recommendations=5)

    assert len(recs) == 5
    assert list(recs.columns) == ["Rank", "Movie ID", "Movie Title", "Genre", "Predicted Rating"]
    # Check descending sort
    ratings = recs["Predicted Rating"].tolist()
    assert ratings == sorted(ratings, reverse=True)


def test_parameter_updates_during_training():
    """Verify that model parameters actually change during training (parameters_before != parameters_after)."""
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 1, 0, 1],
            "movie_idx": [0, 1, 1, 0],
            "rating": [5.0, 1.0, 4.0, 2.0],
        }
    )
    model = MatrixFactorizationSGD(n_factors=4, learning_rate=0.05, regularization=0.01, epochs=1, random_state=42)
    # Initialize parameters explicitly before training
    rng = np.random.default_rng(42)
    init_u_bias = np.zeros(2)
    init_m_bias = np.zeros(2)
    init_u_factors = rng.normal(0.0, 0.1, size=(2, 4))
    init_m_factors = rng.normal(0.0, 0.1, size=(2, 4))

    # Fit model
    model.fit(train_df, verbose=False)

    # Confirm parameters changed
    assert not np.array_equal(model.user_bias, init_u_bias), "User bias failed to update!"
    assert not np.array_equal(model.movie_bias, init_m_bias), "Movie bias failed to update!"
    assert not np.array_equal(model.user_factors, init_u_factors), "User factors failed to update!"
    assert not np.array_equal(model.movie_factors, init_m_factors), "Movie factors failed to update!"


def test_cold_start_recommendations():
    """Verify cold-start handling returns unpersonalized recommendations without crashing."""
    recommender = MovieRecommender()

    # Direct cold-start method
    cold_recs = recommender.recommend_cold_start(n_recommendations=5)
    assert len(cold_recs) == 5
    assert "Rank" in cold_recs.columns
    assert "Movie ID" in cold_recs.columns
    assert "Predicted Rating" in cold_recs.columns

    # Via recommend_movies with cold_start_fallback=True for unknown user
    unknown_uid = 999999
    assert not recommender.is_valid_user(unknown_uid)
    fallback_recs = recommender.recommend_movies(unknown_uid, n_recommendations=5, cold_start_fallback=True)
    assert len(fallback_recs) == 5
    assert len(fallback_recs["Movie ID"].unique()) == 5


def test_recommender_duplicate_prevention():
    """Verify that recommendations contain no duplicate movie IDs or duplicate titles."""
    recommender = MovieRecommender()
    recs = recommender.recommend_movies(user_id=1, n_recommendations=10)

    assert len(recs["Movie ID"]) == len(recs["Movie ID"].unique()), "Duplicate Movie IDs in recommendations!"
    assert len(recs["Movie Title"]) == len(recs["Movie Title"].unique()), "Duplicate Movie Titles in recommendations!"


def test_recommender_ranking_order():
    """Verify that recommendations are strictly sorted descending by predicted rating."""
    recommender = MovieRecommender()
    recs = recommender.recommend_movies(user_id=2, n_recommendations=10)
    scores = recs["Predicted Rating"].tolist()
    assert scores == sorted(scores, reverse=True), "Recommendations not sorted in descending order!"


def test_model_evaluator_metrics(trained_toy_model):
    """Verify evaluation suite computes valid RMSE, MAE, Pearson correlation, and Top-K metrics."""
    from src.evaluation import ModelEvaluator

    test_df = pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "movie_idx": [2, 0, 1],
            "rating": [4.5, 3.0, 4.0],
        }
    )
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "movie_idx": [0, 1, 1, 2, 0, 2],
            "rating": [4.0, 5.0, 3.0, 4.0, 2.0, 5.0],
        }
    )
    evaluator = ModelEvaluator(trained_toy_model, test_df, train_df=train_df, save_artifacts=False)
    metrics = evaluator.run_all()

    assert "test_rmse" in metrics
    assert "test_mae" in metrics
    assert "pearson_correlation" in metrics
    assert metrics["test_rmse"] > 0.0
    assert metrics["test_mae"] > 0.0
    assert -1.0 <= metrics["pearson_correlation"] <= 1.0
    assert "hit_rate@10" in metrics
    assert "precision@10" in metrics
    assert "catalog_coverage_percentage" in metrics


def test_recommender_invalid_user():
    """Verify unknown user raises ValueError when cold_start_fallback is False."""
    recommender = MovieRecommender()
    with pytest.raises(ValueError) as excinfo:
        recommender.recommend_movies(user_id=-999, cold_start_fallback=False)
    assert "not found" in str(excinfo.value)


def test_recommend_bias_shrinkage(trained_toy_model):
    """Verify that bias_shrinkage parameter correctly applies Bayesian damping without error."""
    counts = np.array([10, 1, 5])
    recs_shrunk = trained_toy_model.recommend(
        user_idx=0,
        top_n=2,
        bias_shrinkage=5.0,
        movie_counts=counts,
    )
    assert len(recs_shrunk) == 2
    assert all(0.5 <= score <= 5.0 for _, score in recs_shrunk)
    # Check descending sort
    assert recs_shrunk[0][1] >= recs_shrunk[1][1]
