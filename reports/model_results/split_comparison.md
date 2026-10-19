# Random vs Temporal Evaluation

Two different questions, measured on the same model and hyperparameters.

- **Random split** (`train.py`): ratings are shuffled, so a user's later ratings
  can train the model that predicts their earlier ones. This measures *matrix
  completion* -- filling gaps in a rating matrix -- and is the standard MovieLens
  benchmark setup.
- **Temporal split** (`train_temporal.py`): each user's earliest 80% of ratings
  train the model and their most recent 20% are held out. Nothing from the future
  is ever visible. This measures *forecasting*.

| Metric | Random split | Temporal split |
| :--- | :---: | :---: |
| Test RMSE | 0.8523 | 0.8940 |
| Test MAE | 0.6524 | 0.6862 |
| Pearson correlation | 0.5787 | 0.5491 |
| Improvement over mean baseline | 18.30% | 16.41% |
| Hit Rate@10 | 0.4649 | 0.2741 |
| NDCG@10 | 0.1057 | 0.0537 |

## Why the temporal numbers are worse

- **8.41% of temporal test ratings are for movies with no
  training data at all.** Their latent factors are still at their initial values, so
  the model can only fall back on the global mean. A random split hides this because
  almost every movie appears somewhere in the training data.
- **Tastes drift.** Predicting what someone enjoyed years before the training window
  is easier than predicting what they will enjoy next.

The temporal figure is the honest estimate of live recommendation quality; the
random figure is the number comparable to published MovieLens results.
