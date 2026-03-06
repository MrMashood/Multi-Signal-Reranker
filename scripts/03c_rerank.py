"""
Step 3c: Reranking with ALS Embeddings + NDCG@K Evaluation
===========================================================
Loads trained ALS embeddings and uses dot product scores to
rerank items within each session. Evaluates with NDCG@K against:
  - Baseline A: original shown position (what the user actually saw)
  - Baseline B: random reranking

Input  : data/features/training_data.parquet
         data/model/user_vectors.npy
         data/model/item_vectors.npy
         data/matrix/user_index.parquet
         data/matrix/item_index.parquet

Output : data/eval/rerank_results.parquet   (per-session scores)
         data/eval/evaluation_summary.json  (NDCG@K comparison)

Run:
    python scripts/03c_rerank.py
"""

import os
import json
import numpy as np
import pandas as pd

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "model")
MATRIX_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "matrix")
EVAL_DIR    = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

K_VALUES    = [5, 10, 20]   # evaluate NDCG at these cutoffs
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# NDCG IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(labels: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at K.

    DCG@K = Σ (2^label - 1) / log2(rank + 1)   for rank 1..K

    Higher label at a higher rank = more gain.
    The log2 discount penalizes relevant items appearing late in the list.

    Args:
        labels: relevance labels in ranked order (position 0 = rank 1)
        k:      cutoff
    Returns:
        DCG@K score
    """
    labels = np.array(labels[:k], dtype=float)
    if len(labels) == 0:
        return 0.0
    gains     = 2 ** labels - 1                        # gain per position
    discounts = np.log2(np.arange(2, len(labels) + 2)) # log2(2), log2(3), ...
    return float(np.sum(gains / discounts))


def ndcg_at_k(labels: np.ndarray, k: int) -> float:
    """
    Normalized DCG@K.

    NDCG@K = DCG@K / IDCG@K

    IDCG = DCG of the ideal ranking (labels sorted descending).
    Normalizing by IDCG puts the score in [0, 1]:
      1.0 = perfect ranking
      0.0 = worst possible ranking

    Args:
        labels: relevance labels in the ORDER produced by the ranker
        k:      cutoff
    Returns:
        NDCG@K in [0, 1]
    """
    ideal_labels = np.sort(labels)[::-1]   # best possible order
    idcg = dcg_at_k(ideal_labels, k)
    if idcg == 0:
        return 0.0                          # no relevant items → undefined, return 0
    return dcg_at_k(labels, k) / idcg


def compute_ndcg_for_sessions(sessions_df: pd.DataFrame,
                               score_col: str,
                               k_values: list) -> dict:
    """
    Compute mean NDCG@K across all sessions for a given ranking strategy.

    For each session:
      1. Sort items by score_col descending → this is the ranker's order
      2. Read off relevance_label in that order
      3. Compute NDCG@K

    Average across all sessions = final metric.

    Args:
        sessions_df : DataFrame with columns [session_id, relevance_label, score_col]
        score_col   : column to rank by (higher = better)
        k_values    : list of K cutoffs to evaluate

    Returns:
        dict of {k: mean_ndcg}
    """
    results = {k: [] for k in k_values}

    for _, group in sessions_df.groupby("session_id"):
        # Sort by the ranker's score
        ranked = group.sort_values(score_col, ascending=False)
        labels = ranked["relevance_label"].values

        for k in k_values:
            results[k].append(ndcg_at_k(labels, k))

    return {k: float(np.mean(v)) for k, v in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# RERANKING
# ─────────────────────────────────────────────────────────────────────────────

def compute_als_scores(df: pd.DataFrame,
                       user_vectors: np.ndarray,
                       item_vectors: np.ndarray,
                       user_index: pd.DataFrame,
                       item_index: pd.DataFrame) -> pd.Series:
    """
    Compute ALS dot product score for every (user, item) row in df.

    score = user_vector[u] · item_vector[i]

    Users/items not seen during training get score = 0 (cold start fallback).
    """
    uid_map = dict(zip(user_index["id"], user_index["index"]))
    iid_map = dict(zip(item_index["id"], item_index["index"]))

    user_idx = df["user_id"].map(uid_map).fillna(-1).astype(int).values
    item_idx = df["item_id"].map(iid_map).fillna(-1).astype(int).values

    scores = np.zeros(len(df))

    # Vectorized dot product for all known (user, item) pairs
    known_mask = (user_idx >= 0) & (item_idx >= 0)
    if known_mask.any():
        u_vecs = user_vectors[user_idx[known_mask]]   # (n_known, K)
        i_vecs = item_vectors[item_idx[known_mask]]   # (n_known, K)
        scores[known_mask] = np.sum(u_vecs * i_vecs, axis=1)

    return pd.Series(scores, index=df.index)


def add_baseline_scores(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    """
    Add two baseline ranking scores to compare against ALS.

    Baseline A — original position score:
      Items are scored by their INVERSE shown position.
      Position 1 → score 10, Position 10 → score 1.
      This represents "no reranking" — just use the original order.
      If ALS beats this, it's correcting for position bias.

    Baseline B — random score:
      Uniformly random scores per session.
      This is the floor — any reasonable model should beat random.
    """
    # Baseline A: inverse position (higher position rank = higher score)
    df["score_position_baseline"] = (11 - df["shown_position"]).astype(float)

    # Baseline B: random (shuffled within each session for fairness)
    random_scores = np.zeros(len(df))
    for _, group in df.groupby("session_id"):
        random_scores[group.index] = rng.random(len(group))
    df["score_random_baseline"] = random_scores

    return df


def main():
    print("=" * 60)
    print("Step 3c: Reranking + NDCG@K Evaluation")
    print("=" * 60)

    rng = np.random.RandomState(RANDOM_SEED)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    df           = pd.read_parquet(os.path.join(FEATURE_DIR, "training_data.parquet"))
    user_vectors = np.load(os.path.join(MODEL_DIR, "user_vectors.npy"))
    item_vectors = np.load(os.path.join(MODEL_DIR, "item_vectors.npy"))
    user_index   = pd.read_parquet(os.path.join(MATRIX_DIR, "user_index.parquet"))
    item_index   = pd.read_parquet(os.path.join(MATRIX_DIR, "item_index.parquet"))

    print(f"      Events       : {len(df):,}")
    print(f"      User vectors : {user_vectors.shape}")
    print(f"      Item vectors : {item_vectors.shape}")

    # ── Compute ALS scores ────────────────────────────────────────────────
    print("\n[2/5] Computing ALS reranking scores...")
    df["score_als"] = compute_als_scores(
        df, user_vectors, item_vectors, user_index, item_index
    )
    print(f"      Score range  : [{df['score_als'].min():.4f}, {df['score_als'].max():.4f}]")
    print(f"      Score mean   : {df['score_als'].mean():.4f}")
    print(f"      Score std    : {df['score_als'].std():.4f}")

    # ── Add baselines ─────────────────────────────────────────────────────
    print("\n[3/5] Adding baseline scores...")
    df = add_baseline_scores(df, rng)

    # ── Evaluate NDCG@K ───────────────────────────────────────────────────
    print("\n[4/5] Evaluating NDCG@K...")

    # Split search and homepage sessions — evaluate separately
    search_df   = df[df["session_type"] == "search"].copy()
    homepage_df = df[df["session_type"] == "homepage"].copy()

    rankers = {
        "als":               "score_als",
        "position_baseline": "score_position_baseline",
        "random_baseline":   "score_random_baseline",
    }

    evaluation = {}
    for surface, surface_df in [("search", search_df), ("homepage", homepage_df)]:
        evaluation[surface] = {}
        for name, col in rankers.items():
            ndcg_scores = compute_ndcg_for_sessions(surface_df, col, K_VALUES)
            evaluation[surface][name] = ndcg_scores

    # ── Save results ──────────────────────────────────────────────────────
    print("\n[5/5] Saving results...")
    output_cols = [
        "session_id", "user_id", "item_id", "session_type",
        "query_string", "shown_position", "relevance_label",
        "score_als", "score_position_baseline", "score_random_baseline"
    ]
    df[output_cols].to_parquet(os.path.join(EVAL_DIR, "rerank_results.parquet"), index=False)

    with open(os.path.join(EVAL_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(evaluation, f, indent=2)

    # ── Print results table ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for surface in ["search", "homepage"]:
        n_sessions = df[df["session_type"] == surface]["session_id"].nunique()
        print(f"\n{'─'*60}")
        print(f"  {surface.upper()} SESSIONS  ({n_sessions:,} sessions)")
        print(f"{'─'*60}")
        print(f"  {'Ranker':<25} " + "  ".join(f"NDCG@{k:<4}" for k in K_VALUES))
        print(f"  {'─'*25} " + "  ".join("─"*8 for _ in K_VALUES))

        for name in ["als", "position_baseline", "random_baseline"]:
            scores = evaluation[surface][name]
            label  = {"als": "ALS (ours)",
                      "position_baseline": "Original position",
                      "random_baseline":   "Random"}[name]
            row = f"  {label:<25} " + "  ".join(f"{scores[k]:.4f}  " for k in K_VALUES)
            print(row)

    # ── Improvement summary ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  IMPROVEMENT OVER BASELINES")
    print(f"{'─'*60}")

    for surface in ["search", "homepage"]:
        print(f"\n  {surface.capitalize()}:")
        als_10   = evaluation[surface]["als"][10]
        pos_10   = evaluation[surface]["position_baseline"][10]
        rand_10  = evaluation[surface]["random_baseline"][10]

        vs_pos  = (als_10 - pos_10)  / pos_10  * 100
        vs_rand = (als_10 - rand_10) / rand_10 * 100

        print(f"    ALS vs original position : {vs_pos:+.1f}%  (NDCG@10: {als_10:.4f} vs {pos_10:.4f})")
        print(f"    ALS vs random            : {vs_rand:+.1f}%  (NDCG@10: {als_10:.4f} vs {rand_10:.4f})")

    # ── Position bias correction check ───────────────────────────────────
    print(f"\n{'─'*60}")
    print("  POSITION BIAS CORRECTION CHECK")
    print(f"{'─'*60}")
    print("  Average ALS score by original shown position (search):")
    pos_scores = (search_df.groupby("shown_position")["score_als"]
                            .mean().round(4))
    for pos, score in pos_scores.items():
        bar = "█" * int(score * 20) if score > 0 else ""
        print(f"    Position {pos:2d}: {score:.4f}  {bar}")

    print("\n  If ALS scores are NOT strictly decreasing by position,")
    print("  the model is successfully correcting for position bias. ✅")

if __name__ == "__main__":
    main()