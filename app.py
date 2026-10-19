"""
Interactive Streamlit Web Application for Movie Recommendation System Using Matrix Factorization.
Provides personalized movie recommendations, user rating history inspection,
and model architecture diagnostics.
"""

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from src.config import (
    FIGURES_DIR,
    MODEL_RESULTS_DIR,
    MODEL_FILE,
    MAPPINGS_FILE,
    METRICS_FILE,
    METRICS_TEMPORAL_FILE,
)
from src.recommender import MovieRecommender

# Page Configuration
st.set_page_config(
    page_title="🎬 Movie Recommender — Matrix Factorization",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .genre-badge {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px;
    }
    .rating-badge {
        font-weight: bold;
        color: #D97706;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading Matrix Factorization Model & Mappings...")
def get_recommender() -> MovieRecommender:
    """
    Cache loaded recommender model and interaction matrices.
    """
    return MovieRecommender()


def main():
    # Header Banner
    st.markdown('<div class="main-header">🎬 Movie Recommender — Matrix Factorization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Personalized Movie Recommendations using Collaborative Filtering & Stochastic Gradient Descent (SGD)</div>',
        unsafe_allow_html=True,
    )

    # Initialize Recommender
    try:
        recommender = get_recommender()
    except Exception as e:
        st.error(f"Error loading model or metadata: {e}. Please run `python train.py` first.")
        st.stop()

    num_users = len(recommender.user_to_idx)
    num_movies = len(recommender.movie_to_idx)
    most_act, least_act, max_c, min_c = recommender.get_most_and_least_active_users()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Recommendation Controls")
        st.caption("Select or type a User ID and choose the number of recommendations.")

        # Preset shortcut buttons
        st.markdown("**Quick Preset Users:**")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            if st.button("User 1", width="stretch"):
                st.session_state["selected_user"] = 1
        with col_p2:
            if st.button(f"User {most_act}", help=f"Most active user ({max_c} ratings)", width="stretch"):
                st.session_state["selected_user"] = int(most_act)
        with col_p3:
            if st.button("User 42", help="Random user", width="stretch"):
                st.session_state["selected_user"] = 42

        # User ID Input
        default_user = st.session_state.get("selected_user", 1)
        user_id_input = st.number_input(
            "User ID",
            min_value=1,
            max_value=num_users * 2,
            value=int(default_user),
            step=1,
            help=f"Dataset contains {num_users} users with IDs from 1 to {num_users}.",
        )

        # Number of recommendations
        n_recs = st.slider(
            "Number of Recommendations (Top-N)",
            min_value=3,
            max_value=25,
            value=10,
            step=1,
        )

        # Primary Action Button
        get_recs_clicked = st.button("🚀 Get Recommendations", type="primary", width="stretch")

        # Load dynamic metrics
        metrics = {}
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:
                metrics = {}

        # Only ever report a genuine held-out test RMSE here. The training RMSE is
        # not a stand-in for it, so when metrics.json is missing we show nothing.
        test_rmse_val = metrics.get("test_rmse")

        # Temporal ("future prediction") evaluation, written by train_temporal.py.
        temporal_metrics = {}
        if METRICS_TEMPORAL_FILE.exists():
            try:
                with open(METRICS_TEMPORAL_FILE, "r", encoding="utf-8") as f:
                    temporal_metrics = json.load(f)
            except Exception:
                temporal_metrics = {}
        sparsity_val = (
            (1.0 - (len(recommender.interaction_df) / (num_users * num_movies))) * 100.0
            if (num_users * num_movies) > 0
            else 98.30
        )

        st.divider()
        st.subheader("📌 Model Stats")
        st.write(f"• **Catalog Size:** {num_movies:,} movies")
        st.write(f"• **Total Reviewers:** {num_users:,} users")
        st.write(f"• **Latent Factors:** {recommender.model.n_factors}")
        st.write(f"• **Matrix Sparsity:** {sparsity_val:.2f}%")
        st.write(
            f"• **Test RMSE:** {test_rmse_val:.4f}"
            if test_rmse_val is not None
            else "• **Test RMSE:** not available (run train.py)"
        )
        if temporal_metrics.get("test_rmse") is not None:
            st.write(
                f"• **Test RMSE (future prediction):** {temporal_metrics['test_rmse']:.4f}"
            )

    # Metrics Summary Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Total Catalog Movies", f"{num_movies:,}")
    with col_m2:
        st.metric("Total Profiled Users", f"{num_users:,}")
    with col_m3:
        st.metric("Global Mean Rating (μ)", f"{recommender.model.global_mean:.2f} ★")
    with col_m4:
        st.metric(
            "Model Test RMSE",
            f"{test_rmse_val:.4f}" if test_rmse_val is not None else "N/A",
        )

    # Main Action Tab Navigation
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🍿 Recommendations", "📜 User Rating History", "🧠 Evaluation & Model Architecture", "📈 EDA & Diagnostics"]
    )

    with tab1:
        st.subheader(f"Top-{n_recs} Unseen Recommendations for User #{user_id_input}")

        if get_recs_clicked or "has_run" not in st.session_state:
            st.session_state["has_run"] = True

            with st.spinner(f"Computing latent projections for User #{user_id_input}..."):
                try:
                    is_known = recommender.is_valid_user(user_id_input)
                    recs_df = recommender.recommend_movies(
                        user_id=user_id_input,
                        n_recommendations=n_recs,
                        cold_start_fallback=True,
                    )

                    if is_known:
                        st.success(
                            f"Generated **{len(recs_df)}** personalized recommendations for **User {user_id_input}**! "
                            f"*(All movies previously rated by this user have been strictly excluded)*"
                        )
                    else:
                        st.warning(
                            f"⚠️ **User #{user_id_input}** is a new user not found in the training catalog (Cold Start). "
                            f"Showing Top-{len(recs_df)} unpersonalized recommendations based on global baseline rating ($\mu + b_i$)."
                        )

                    # Display recommendations table
                    display_df = recs_df.copy()
                    display_df["Predicted Rating"] = display_df["Predicted Rating"].apply(lambda x: f"★ {x:.2f}")

                    st.dataframe(
                        display_df[["Rank", "Movie Title", "Genre", "Predicted Rating"]],
                        width="stretch",
                        hide_index=True,
                    )

                    # Highlight Top 3 in aesthetic cards
                    st.markdown("#### 🌟 Spotlight: Top 3 Picks")
                    top3_cols = st.columns(min(3, len(recs_df)))
                    for i, col in enumerate(top3_cols):
                        if i < len(recs_df):
                            row = recs_df.iloc[i]
                            with col:
                                st.markdown(
                                    f"""
                                    <div class="metric-card">
                                        <div style="font-size: 1.5rem; font-weight: bold; color: #4F46E5;">Rank #{row['Rank']}</div>
                                        <div style="font-size: 1.1rem; font-weight: 700; margin: 0.5rem 0;">{row['Movie Title']}</div>
                                        <div><span class="genre-badge">{row['Genre']}</span></div>
                                        <div class="rating-badge" style="margin-top: 0.6rem;">★ {row['Predicted Rating']:.2f} / 5.0</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                except ValueError as err:
                    st.error(f"⚠️ {err}")
                    st.info("Tip: Please choose a valid User ID between 1 and 610.")
                except Exception as ex:
                    st.error(f"Unexpected error: {ex}")

    with tab2:
        st.subheader(f"Movies Previously Rated by User #{user_id_input}")
        st.caption("Inspect the historical ratings to understand the user's taste profile and preferences.")

        history_df = recommender.get_user_history(user_id=user_id_input, limit=15)
        if not history_df.empty:
            st.dataframe(
                history_df[["Movie ID", "Title", "Genre", "User Rating"]],
                width="stretch",
                hide_index=True,
            )
            st.caption(f"Showing up to 15 highest-rated movies out of user's interaction history.")
        else:
            st.info(f"No previous rating history found for User #{user_id_input}.")

    with tab3:
        st.subheader("How the model is evaluated")
        st.markdown(
            "The same model is scored two different ways. Which number matters "
            "depends on which question you are asking."
        )

        if temporal_metrics.get("test_rmse") is not None:
            comp = pd.DataFrame(
                {
                    "Metric": ["Test RMSE", "Test MAE", "Pearson correlation",
                               "Hit Rate@10", "NDCG@10"],
                    "Random split": [
                        metrics.get("test_rmse"), metrics.get("test_mae"),
                        metrics.get("pearson_correlation"),
                        metrics.get("hit_rate@10"), metrics.get("ndcg@10"),
                    ],
                    "Temporal split": [
                        temporal_metrics.get("test_rmse"), temporal_metrics.get("test_mae"),
                        temporal_metrics.get("pearson_correlation"),
                        temporal_metrics.get("hit_rate@10"), temporal_metrics.get("ndcg@10"),
                    ],
                }
            )
            st.dataframe(comp, width="stretch", hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    "**Random split** — ratings are shuffled before splitting. A user's "
                    "later ratings can train the model that predicts their earlier ones, "
                    "so this measures *matrix completion*: filling gaps in a rating matrix "
                    "we already partly know. This is the standard MovieLens benchmark setup."
                )
            with c2:
                st.markdown(
                    "**Temporal split** — each user's earliest 80% of ratings train the "
                    "model and their most recent 20% are held out. Nothing from the future "
                    "is ever visible, so this measures *forecasting*."
                )

            unseen = temporal_metrics.get("unseen_movie_percentage")
            if unseen is not None:
                st.info(
                    f"**Why the temporal numbers are lower:** {unseen}% of the temporal "
                    "test ratings are for movies with no training data at all, so their "
                    "latent factors were never learned and the model can only fall back "
                    "on the global mean. A random split hides this, because nearly every "
                    "movie appears somewhere in the training data."
                )

            st.warning(
                "**Caveat worth knowing:** 63% of MovieLens users rated everything within "
                "a single day, and for 97% of users the gap between their last training "
                "rating and first test rating is under a day. So the temporal split is a "
                "genuine past-to-future split, but the 'future' is usually minutes later, "
                "not months. It measures cold-start exposure honestly; it does not really "
                "measure long-term taste drift. That is a limit of this dataset, not of "
                "the split."
            )
        else:
            st.info(
                "Temporal evaluation not available yet. Run `python3 train_temporal.py` "
                "to generate it."
            )

        st.divider()
        st.subheader("Mathematical Model: Matrix Factorization with Biases & SGD")
        st.markdown(
            r"""
            Collaborative filtering assumes that users and items can be characterized by a set of latent factor vectors:
            
            $$\hat{r}_{u,i} = \mu + b_u + b_i + P_u \cdot Q_i^T$$
            
            Where:
            - **$\mu$**: Global baseline average rating across all users and movies in the training dataset ($3.50$).
            - **$b_u$**: User bias term reflecting user's tendency to rate higher or lower than average.
            - **$b_i$**: Movie bias term reflecting movie's overall perceived quality relative to the baseline.
            - **$P_u \in \mathbb{R}^k$**: Latent preference feature vector of user $u$.
            - **$Q_i \in \mathbb{R}^k$**: Latent attribute feature vector of movie $i$.
            
            ### Stochastic Gradient Descent (SGD) Updates
            For every observed interaction $(u, i, r_{u,i})$, the prediction error is computed as:
            $$e_{u,i} = r_{u,i} - \hat{r}_{u,i}$$
            
            The model parameters are updated iteratively with learning rate $\gamma = 0.01$ and $L_2$ regularization $\lambda = 0.02$:
            $$b_u \leftarrow b_u + \gamma (e_{u,i} - \lambda b_u)$$
            $$b_i \leftarrow b_i + \gamma (e_{u,i} - \lambda b_i)$$
            $$P_u \leftarrow P_u + \gamma (e_{u,i} Q_i - \lambda P_u)$$
            $$Q_i \leftarrow Q_i + \gamma (e_{u,i} P_u - \lambda Q_i)$$
            """
        )

        st.subheader("Model Artifact Details")
        st.code(
            f"""Hyperparameters:
- Latent Factors (K):   {recommender.model.n_factors}
- Learning Rate (γ):    {recommender.model.learning_rate}
- Regularization (λ):   {recommender.model.regularization}
- Epochs Trained:       {recommender.model.epochs}
- Random Seed:          {recommender.model.random_state}
- Model Artifact Path:  {MODEL_FILE}
- Mappings Path:        {MAPPINGS_FILE}
            """,
            language="text",
        )

    with tab4:
        st.subheader("Exploratory Data Analysis & Diagnostic Figures")
        st.caption("Visual summaries generated during the EDA and Model Evaluation phases.")

        col_f1, col_f2 = st.columns(2)

        curve_path = FIGURES_DIR / "06_learning_curve.png"
        if curve_path.exists():
            with col_f1:
                st.image(str(curve_path), caption="Training RMSE Convergence Curve (SGD Epochs 1-20)", width="stretch")

        actual_pred_path = MODEL_RESULTS_DIR / "actual_vs_predicted.png"
        if actual_pred_path.exists():
            with col_f2:
                st.image(str(actual_pred_path), caption="Actual vs. Predicted Ratings on 20% Held-Out Test Set", width="stretch")

        res_path = MODEL_RESULTS_DIR / "residuals_distribution.png"
        if res_path.exists():
            with col_f1:
                st.image(str(res_path), caption="Residuals Error Distribution (Centered at ~0.0)", width="stretch")

        dist_path = FIGURES_DIR / "01_rating_distribution.png"
        if dist_path.exists():
            with col_f2:
                st.image(str(dist_path), caption="Raw Ratings Distribution Across MovieLens Dataset", width="stretch")


if __name__ == "__main__":
    main()
