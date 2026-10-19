"""
Configuration module for Movie Recommendation System Using Matrix Factorization.
Stores directory paths, dataset settings, model hyperparameters, and constants.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific file paths
RATINGS_FILE = RAW_DATA_DIR / "ratings.csv"
MOVIES_FILE = RAW_DATA_DIR / "movies.csv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
MAPPINGS_FILE = PROCESSED_DATA_DIR / "mappings.json"

# Temporal ("future prediction") evaluation. Kept entirely separate from the
# random-split files above so both evaluations can be reported side by side.
TRAIN_TEMPORAL_FILE = PROCESSED_DATA_DIR / "train_temporal.csv"
TEST_TEMPORAL_FILE = PROCESSED_DATA_DIR / "test_temporal.csv"

# Models and artifacts
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "matrix_factorization_model.pkl"
MODEL_TEMPORAL_FILE = MODELS_DIR / "matrix_factorization_temporal.pkl"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_RESULTS_DIR = REPORTS_DIR / "model_results"
METRICS_FILE = MODEL_RESULTS_DIR / "metrics.json"
METRICS_TEMPORAL_FILE = MODEL_RESULTS_DIR / "metrics_temporal.json"

# Random state for reproducibility
RANDOM_STATE = 42

# Train/Test split ratio
TEST_SIZE = 0.20

# Validation split ratio, carved out of the TRAINING set only.
# The test set stays fully held out so reported metrics remain unbiased.
VAL_SIZE = 0.125

# Matrix Factorization Default Hyperparameters.
# N_FACTORS / REGULARIZATION selected by grid search on the validation split
# (best val RMSE 0.8556; see reports/model_results/sweep_results.json).
N_FACTORS = 100
LEARNING_RATE = 0.01
REGULARIZATION = 0.12
EPOCHS = 100
INIT_MEAN = 0.0
INIT_STD = 0.1

# Early stopping: epochs to wait for a validation improvement before halting.
# Best-epoch weights are restored, so EPOCHS acts as an upper bound only.
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 1e-4

# Rating scale constraints
MIN_RATING = 0.5
MAX_RATING = 5.0

# Recommendation Ranking Optimization
BIAS_SHRINKAGE = 10.0

# Popularity penalty applied when ranking recommendations. 0.0 ranks purely by
# predicted rating, which concentrates every user's list on the same blockbusters
# and keeps catalog coverage low. Tuned on the validation split.
NOVELTY_WEIGHT = 0.0
