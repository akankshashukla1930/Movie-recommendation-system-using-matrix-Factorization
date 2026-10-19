# Model Evaluation Report: Matrix Factorization Recommender

## Executive Summary
Quantitative benchmark evaluation for the custom Matrix Factorization with SGD Collaborative Filtering model evaluated on the held-out 20% test partition (20,168 interactions).

## Performance Metrics (Rating Calibration)

| Metric | Score | Naive Global Mean Baseline | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **0.8523** | 1.0432 | **+18.30%** |
| **Mean Absolute Error (MAE)** | **0.6524** | 0.8253 | - |
| **Pearson Correlation** | **0.5787** | 0.0000 | - |
| **Residual Error Mean (Bias)** | **-0.0248** | - | Unbiased ($\approx 0$) |
| **Residual Error Std** | **0.8519** | - | - |

## Top-K Recommendation Quality & Catalog Coverage

> [!NOTE]
> Rating error metrics (RMSE/MAE) measure rating calibration, NOT recommendation hit accuracy. Top-K ranking metrics evaluate whether the model surfaces genuine relevant items (rating $\ge 4.0$) in the user's top recommended list.

| Ranking Metric (Top-10) | Score | Description |
| :--- | :---: | :--- |
| **Hit Rate@10** | **0.4649** | Proportion of test users receiving $\ge 1$ relevant item in top 10 |
| **Precision@10** | **0.0824** | Average fraction of top 10 recommendations that are relevant |
| **Recall@10** | **0.0723** | Average fraction of user's relevant test items captured in top 10 |
| **NDCG@10** | **0.1057** | Normalized Discounted Cumulative Gain accounting for rank position |
| **Catalog Coverage** | **4.36%** | Percentage of total catalog items recommended across all test users |

## Analytical Interpretation
1. **RMSE Score (0.8523)**:
   The model achieves competitive accuracy on the MovieLens benchmark, beating the unpersonalized baseline by over 18.3%.
2. **Residual Distribution**:
   Errors are centered symmetrically around 0.0 with a bell-shaped distribution, proving that the user and item biases successfully eliminated systematic over- or under-estimation.
3. **Realistic Real-World Accuracy**:
   In 5-star rating scales with high user subjectivity and sparse interactions, an RMSE near 0.85–0.90 is typical state-of-the-art for pure collaborative filtering models.
