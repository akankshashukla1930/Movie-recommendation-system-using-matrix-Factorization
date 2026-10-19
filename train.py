"""
Master End-to-End Training & Evaluation Pipeline for Movie Recommendation System.
Runs data loading, preprocessing, Matrix Factorization SGD training,
model serialization, evaluation metric computation, and recommendation engine tests.
"""

import sys
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_FILE,
    TEST_FILE,
    MODEL_FILE,
    N_FACTORS,
    LEARNING_RATE,
    REGULARIZATION,
    EPOCHS,
    RANDOM_STATE,
    VAL_SIZE,
    EARLY_STOPPING_PATIENCE,
)
from src.preprocessing import run_preprocessing_pipeline, DataPreprocessor
from src.matrix_factorization import MatrixFactorizationSGD
from src.evaluation import ModelEvaluator
from src.recommender import MovieRecommender, test_recommendation_engine


def run_pipeline() -> None:
    start_time = time.time()
    print("*" * 75)
    print("STARTING END-TO-END TRAINING PIPELINE: MOVIE RECOMMENDATION SYSTEM")
    print("*" * 75)

    # 1. Preprocessing and Data Ingestion
    print("\n[STEP 1/6] Running Data Validation and Preprocessing...")
    train_df, test_df, report = run_preprocessing_pipeline()
    mappings = DataPreprocessor.load_mappings()
    num_users = mappings["num_users"]
    num_movies = mappings["num_movies"]
    print(f"Data ready: {len(train_df):,} training samples, {len(test_df):,} test samples.")
    print(f"Catalog: {num_users:,} users, {num_movies:,} movies.")

    # 2. Matrix Factorization Training with SGD
    print("\n[STEP 2/6] Initializing & Training Matrix Factorization Model with SGD...")

    # Carve a validation set out of TRAIN only. The test set is never used to
    # pick hyperparameters or the stopping epoch, so its metrics stay unbiased.
    fit_df, val_df = train_test_split(
        train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    print(
        f"Early-stopping split: {len(fit_df):,} fit / {len(val_df):,} validation "
        f"(test set of {len(test_df):,} stays held out)."
    )

    search_model = MatrixFactorizationSGD(
        n_factors=N_FACTORS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        epochs=EPOCHS,
        random_state=RANDOM_STATE,
        patience=EARLY_STOPPING_PATIENCE,
    )
    search_model.fit(
        train_df=fit_df,
        val_df=val_df,
        num_users=num_users,
        num_movies=num_movies,
        verbose=True,
    )

    best_epoch = search_model.best_epoch or EPOCHS
    print(
        f"\nValidation selected epoch {best_epoch} "
        f"(Val RMSE: {search_model.best_val_rmse:.4f})."
    )

    # Refit on the FULL training set for the number of epochs validation chose,
    # so the shipped model does not lose the held-out validation ratings.
    print(f"\nRefitting on the full training set for {best_epoch} epochs...")
    model = MatrixFactorizationSGD(
        n_factors=N_FACTORS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        epochs=best_epoch,
        random_state=RANDOM_STATE,
    )
    model.fit(
        train_df=train_df,
        val_df=None,
        num_users=num_users,
        num_movies=num_movies,
        verbose=True,
    )
    # Keep the validation curve from the search run for plotting/inspection.
    model.val_rmse_history = search_model.val_rmse_history
    model.best_epoch = search_model.best_epoch
    model.best_val_rmse = search_model.best_val_rmse

    # 3. Plot and Save Learning Curve
    print("\n[STEP 3/6] Saving Training Convergence Curve...")
    fig_path = search_model.plot_learning_curve()
    print(f"Convergence plot saved to: {fig_path}")

    # 4. Model Serialization
    print("\n[STEP 4/6] Serializing Trained Model...")
    saved_model_path = model.save_model(MODEL_FILE)
    print(f"Model saved to: {saved_model_path}")

    # Verify reload
    print("Verifying model reloading...")
    reloaded_model = MatrixFactorizationSGD.load_model(MODEL_FILE)
    test_sample_pred = reloaded_model.predict(0, 0)
    print(f"Reload test verification prediction for (u=0, i=0): {test_sample_pred:.3f}")

    # 5. Model Evaluation
    print("\n[STEP 5/6] Evaluating Model on Held-Out Test Set...")
    evaluator = ModelEvaluator(model=reloaded_model, test_df=test_df, train_df=train_df)
    metrics = evaluator.run_all()

    # 6. Recommendation Verification
    print("\n[STEP 6/6] Testing Recommendation Engine...")
    recommender = MovieRecommender(model=reloaded_model)
    test_recommendation_engine(recommender)

    elapsed = time.time() - start_time
    print("\n" + "*" * 75)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(
        f"Final Test RMSE: {metrics['test_rmse']:.4f} | Test MAE: {metrics['test_mae']:.4f} | "
        f"Hit Rate@10: {metrics.get('hit_rate@10', 0):.4f} | NDCG@10: {metrics.get('ndcg@10', 0):.4f}"
    )
    print("*" * 75)


if __name__ == "__main__":
    run_pipeline()
