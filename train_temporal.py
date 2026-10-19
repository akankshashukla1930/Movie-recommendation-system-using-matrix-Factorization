"""
Temporal ("future prediction") training and evaluation pipeline.

Trains the same Matrix Factorization model as train.py, but on a chronological
split: each user's earliest ratings are used for training and their most recent
ones are held out. This answers "what will this user like next?" rather than
"what did this user think of a film we already know they watched?".

Writes its own metrics file and leaves the random-split artifacts untouched, so
both evaluations can be compared side by side.
"""

import json
import time
from typing import Tuple

import pandas as pd

from src.config import (
    N_FACTORS,
    LEARNING_RATE,
    REGULARIZATION,
    EPOCHS,
    RANDOM_STATE,
    VAL_SIZE,
    EARLY_STOPPING_PATIENCE,
    MODEL_TEMPORAL_FILE,
    METRICS_TEMPORAL_FILE,
    METRICS_FILE,
    MODEL_RESULTS_DIR,
)
from src.temporal_split import build_temporal_split, load_temporal_split
from src.matrix_factorization import MatrixFactorizationSGD
from src.evaluation import ModelEvaluator


def split_validation_by_recency(
    train_df: pd.DataFrame, val_size: float = VAL_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out each user's most recent training ratings for validation.

    The rows of the temporal training file are already chronological within each
    user, so the tail of each user's block is their most recent activity. Using a
    random validation split here would reintroduce exactly the look-ahead this
    pipeline exists to avoid.
    """
    fit_parts, val_parts = [], []
    for _, group in train_df.groupby("user_idx", sort=False):
        n = len(group)
        n_val = int(round(n * val_size))
        n_val = max(1, min(n_val, n - 1)) if n > 1 else 0
        if n_val == 0:
            fit_parts.append(group)
            continue
        fit_parts.append(group.iloc[: n - n_val])
        val_parts.append(group.iloc[n - n_val :])

    fit_df = pd.concat(fit_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else train_df.iloc[:0]
    return fit_df, val_df


def run_pipeline() -> None:
    start = time.time()
    print("*" * 75)
    print("TEMPORAL PIPELINE: PREDICTING EACH USER'S FUTURE RATINGS")
    print("*" * 75)

    print("\n[STEP 1/5] Building chronological split...")
    report = build_temporal_split()
    train_df, test_df = load_temporal_split()
    num_users, num_movies = report["num_users"], report["num_movies"]
    print(f"Train: {report['train_rows']:,} ratings  ({report['train_period'][0]} -> {report['train_period'][1]})")
    print(f"Test : {report['test_rows']:,} ratings  (each user's most recent 20%)")
    print(
        f"Cold-start exposure: {report['test_ratings_on_unseen_movies']:,} test ratings "
        f"({report['unseen_movie_percentage']}%) are for movies with no training data."
    )

    print("\n[STEP 2/5] Training with recency-based early stopping...")
    fit_df, val_df = split_validation_by_recency(train_df)
    print(f"Fit: {len(fit_df):,} | Validation (each user's latest): {len(val_df):,}")

    search = MatrixFactorizationSGD(
        n_factors=N_FACTORS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        epochs=EPOCHS,
        random_state=RANDOM_STATE,
        patience=EARLY_STOPPING_PATIENCE,
    )
    search.fit(fit_df, val_df, num_users, num_movies, verbose=True)
    best_epoch = search.best_epoch or EPOCHS
    print(f"\nValidation selected epoch {best_epoch} (Val RMSE {search.best_val_rmse:.4f}).")

    print(f"\n[STEP 3/5] Refitting on the full temporal training set for {best_epoch} epochs...")
    model = MatrixFactorizationSGD(
        n_factors=N_FACTORS,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        epochs=best_epoch,
        random_state=RANDOM_STATE,
    )
    model.fit(train_df, None, num_users, num_movies, verbose=True)
    model.val_rmse_history = search.val_rmse_history
    model.best_epoch = search.best_epoch
    model.best_val_rmse = search.best_val_rmse

    print("\n[STEP 4/5] Saving model and learning curve...")
    model.save_model(MODEL_TEMPORAL_FILE)
    curve = search.plot_learning_curve(save_path=MODEL_RESULTS_DIR / "learning_curve_temporal.png")
    print(f"Learning curve saved to: {curve}")

    print("\n[STEP 5/5] Evaluating on each user's held-out future ratings...")
    evaluator = ModelEvaluator(
        model=model,
        test_df=test_df,
        train_df=train_df,
        save_artifacts=False,
    )
    metrics = evaluator.run_all()
    metrics["split_type"] = "temporal (per-user chronological)"
    metrics.update(
        {
            "train_period": report["train_period"],
            "test_period": report["test_period"],
            "test_ratings_on_unseen_movies": report["test_ratings_on_unseen_movies"],
            "unseen_movie_percentage": report["unseen_movie_percentage"],
            "selected_epoch": best_epoch,
        }
    )
    METRICS_TEMPORAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_TEMPORAL_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Temporal metrics written to: {METRICS_TEMPORAL_FILE}")

    write_comparison(metrics)

    print("\n" + "*" * 75)
    print(f"TEMPORAL PIPELINE COMPLETE IN {time.time() - start:.2f} SECONDS")
    print(
        f"Future-prediction Test RMSE: {metrics['test_rmse']:.4f} | "
        f"MAE: {metrics['test_mae']:.4f} | NDCG@10: {metrics.get('ndcg@10', 0):.4f}"
    )
    print("*" * 75)


def write_comparison(temporal: dict) -> None:
    """Write a side-by-side report of the random and temporal evaluations."""
    if not METRICS_FILE.exists():
        print("Skipping comparison report: random-split metrics not found (run train.py first).")
        return

    random_metrics = json.loads(METRICS_FILE.read_text())
    rows = [
        ("Test RMSE", "test_rmse", "{:.4f}"),
        ("Test MAE", "test_mae", "{:.4f}"),
        ("Pearson correlation", "pearson_correlation", "{:.4f}"),
        ("Improvement over mean baseline", "rmse_improvement_percentage", "{:.2f}%"),
        ("Hit Rate@10", "hit_rate@10", "{:.4f}"),
        ("NDCG@10", "ndcg@10", "{:.4f}"),
    ]

    lines = [
        "# Random vs Temporal Evaluation",
        "",
        "Two different questions, measured on the same model and hyperparameters.",
        "",
        "- **Random split** (`train.py`): ratings are shuffled, so a user's later ratings",
        "  can train the model that predicts their earlier ones. This measures *matrix",
        "  completion* -- filling gaps in a rating matrix -- and is the standard MovieLens",
        "  benchmark setup.",
        "- **Temporal split** (`train_temporal.py`): each user's earliest 80% of ratings",
        "  train the model and their most recent 20% are held out. Nothing from the future",
        "  is ever visible. This measures *forecasting*.",
        "",
        "| Metric | Random split | Temporal split |",
        "| :--- | :---: | :---: |",
    ]
    for label, key, fmt in rows:
        r = random_metrics.get(key)
        t = temporal.get(key)
        rs = fmt.format(r) if isinstance(r, (int, float)) else "N/A"
        ts = fmt.format(t) if isinstance(t, (int, float)) else "N/A"
        lines.append(f"| {label} | {rs} | {ts} |")

    lines += [
        "",
        "## Why the temporal numbers are worse",
        "",
        f"- **{temporal.get('unseen_movie_percentage')}% of temporal test ratings are for movies with no",
        "  training data at all.** Their latent factors are still at their initial values, so",
        "  the model can only fall back on the global mean. A random split hides this because",
        "  almost every movie appears somewhere in the training data.",
        "- **Tastes drift.** Predicting what someone enjoyed years before the training window",
        "  is easier than predicting what they will enjoy next.",
        "",
        "The temporal figure is the honest estimate of live recommendation quality; the",
        "random figure is the number comparable to published MovieLens results.",
        "",
    ]
    out = MODEL_RESULTS_DIR / "split_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Comparison report written to: {out}")


if __name__ == "__main__":
    run_pipeline()
