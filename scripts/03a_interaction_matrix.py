"""
Step 3a: Build User-Item Interaction Matrix
============================================
Reads training_data.parquet and constructs the sparse interaction
matrix R that ALS will factorize.

Input  : data/features/training_data.parquet
Output :
  data/matrix/interaction_matrix.npz   (sparse R matrix)
  data/matrix/user_index.parquet        (user_id  → row index mapping)
  data/matrix/item_index.parquet        (item_id  → col index mapping)
  data/matrix/matrix_stats.json         (shape, density, label breakdown)

Run:
    python scripts/03a_interaction_matrix.py
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp

FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "features")
MATRIX_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "matrix")
os.makedirs(MATRIX_DIR, exist_ok=True)

TRAINING_DATA_PATH = os.path.join(FEATURE_DIR, "training_data.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE WEIGHTING
#
# ALS with implicit feedback doesn't use raw ratings directly.
# Instead it uses two things:
#
#   p_ui (preference) = 1 if user interacted with item, else 0
#   c_ui (confidence) = 1 + alpha * interaction_strength
#
# The interaction_strength is our relevance_label (0-3), mapped to a
# meaningful confidence value. Higher alpha = model trusts strong signals more.
#
# Why not just use relevance_label as the rating directly?
# Because ALS implicit feedback treats ALL observations as either
# "user showed preference" (p=1) or "unknown" (p=0, not negative).
# The confidence weight controls HOW STRONGLY to enforce each preference.
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 40   # standard starting value from Hu et al. 2008

LABEL_TO_CONFIDENCE = {
    0: 1.0,          # shown but ignored  → confidence=1  (weak, almost noise)
    1: 1 + ALPHA,    # clicked            → confidence=41
    2: 1 + ALPHA*2,  # carted             → confidence=81
    3: 1 + ALPHA*4,  # purchased          → confidence=161 (strongest signal)
}

# Preference: any interaction beyond just being shown = 1
# Being shown but ignored = 0 (we don't know if user disliked it or missed it)
LABEL_TO_PREFERENCE = {
    0: 0,   # no preference signal
    1: 1,   # clicked → preference
    2: 1,   # carted  → preference
    3: 1,   # bought  → preference
}


def build_index(series: pd.Series) -> pd.DataFrame:
    """Map unique IDs to consecutive integer indices."""
    unique_ids = sorted(series.unique())
    return pd.DataFrame({
        "id":    unique_ids,
        "index": range(len(unique_ids))
    })


def build_interaction_matrix(df: pd.DataFrame,
                              user_index: pd.DataFrame,
                              item_index: pd.DataFrame) -> sp.csr_matrix:
    """
    Build sparse confidence-weighted interaction matrix R.

    Shape: (n_users, n_items)
    Value: confidence weight c_ui for each (user, item) pair
           where preference p_ui = 1 (i.e. relevance_label > 0)

    For (user, item) pairs with multiple sessions (user saw item multiple times),
    we take the MAX label — a purchase beats a click.

    Zero entries = user never interacted with item (preference=0, confidence=1)
    We only store non-zero confidence values in the sparse matrix.
    """
    uid_map  = dict(zip(user_index["id"],    user_index["index"]))
    iid_map  = dict(zip(item_index["id"],    item_index["index"]))

    # Aggregate: one row per (user, item) — take max label across sessions
    agg = (
        df.groupby(["user_id", "item_id"])["relevance_label"]
          .max()
          .reset_index()
    )

    # Only keep rows where user showed preference (label > 0)
    # Label=0 rows (ignored impressions) are implicit zeros in the sparse matrix
    interactions = agg[agg["relevance_label"] > 0].copy()

    # Map to integer indices
    interactions["user_idx"] = interactions["user_id"].map(uid_map)
    interactions["item_idx"] = interactions["item_id"].map(iid_map)

    # Drop any unmapped (shouldn't happen but defensive)
    interactions = interactions.dropna(subset=["user_idx", "item_idx"])
    interactions["user_idx"] = interactions["user_idx"].astype(int)
    interactions["item_idx"] = interactions["item_idx"].astype(int)

    # Map labels to confidence values
    interactions["confidence"] = interactions["relevance_label"].map(LABEL_TO_CONFIDENCE)

    # Build COO sparse matrix then convert to CSR (efficient row slicing for ALS)
    n_users = len(user_index)
    n_items = len(item_index)

    R = sp.coo_matrix(
        (
            interactions["confidence"].values,
            (interactions["user_idx"].values, interactions["item_idx"].values)
        ),
        shape=(n_users, n_items)
    ).tocsr()

    return R, interactions


def main():
    print("=" * 60)
    print("Step 3a: Building Interaction Matrix")
    print("=" * 60)

    # ── Load training data ────────────────────────────────────────────────
    print(f"\n[1/4] Loading training_data.parquet...")
    df = pd.read_parquet(TRAINING_DATA_PATH)
    print(f"      Shape     : {df.shape}")
    print(f"      Users     : {df['user_id'].nunique():,}")
    print(f"      Items     : {df['item_id'].nunique():,}")
    print(f"      Sessions  : {df['session_id'].nunique():,}")

    # ── Build ID → index mappings ─────────────────────────────────────────
    print(f"\n[2/4] Building user/item index mappings...")
    user_index = build_index(df["user_id"])
    item_index = build_index(df["item_id"])

    user_index.to_parquet(os.path.join(MATRIX_DIR, "user_index.parquet"), index=False)
    item_index.to_parquet(os.path.join(MATRIX_DIR, "item_index.parquet"), index=False)
    print(f"      User index: {len(user_index):,} users  (u_00000 → 0, u_00001 → 1, ...)")
    print(f"      Item index: {len(item_index):,} items  (i_00000 → 0, i_00001 → 1, ...)")

    # ── Build interaction matrix ──────────────────────────────────────────
    print(f"\n[3/4] Building sparse interaction matrix R...")
    R, interactions = build_interaction_matrix(df, user_index, item_index)

    sp.save_npz(os.path.join(MATRIX_DIR, "interaction_matrix.npz"), R)

    n_users, n_items = R.shape
    nnz      = R.nnz
    density  = nnz / (n_users * n_items)

    print(f"      Shape     : ({n_users:,} users  ×  {n_items:,} items)")
    print(f"      Non-zeros : {nnz:,}  (user-item pairs with preference=1)")
    print(f"      Density   : {density:.4%}  (how sparse — lower = harder problem)")
    print(f"      Zeros     : {n_users*n_items - nnz:,}  (unknown preferences)")

    # ── Matrix stats ──────────────────────────────────────────────────────
    print(f"\n[4/4] Computing matrix statistics...")

    # Interactions per user distribution
    interactions_per_user = np.diff(R.indptr)   # CSR row lengths = items per user
    interactions_per_item = np.diff(R.tocsc().indptr)

    stats = {
        "n_users":               int(n_users),
        "n_items":               int(n_items),
        "n_nonzero":             int(nnz),
        "density":               round(float(density), 6),
        "alpha":                 ALPHA,
        "label_to_confidence":   {str(k): v for k, v in LABEL_TO_CONFIDENCE.items()},
        "interactions_per_user": {
            "min":    int(interactions_per_user.min()),
            "max":    int(interactions_per_user.max()),
            "mean":   round(float(interactions_per_user.mean()), 2),
            "median": round(float(np.median(interactions_per_user)), 2),
        },
        "interactions_per_item": {
            "min":    int(interactions_per_item.min()),
            "max":    int(interactions_per_item.max()),
            "mean":   round(float(interactions_per_item.mean()), 2),
            "median": round(float(np.median(interactions_per_item)), 2),
        },
        "confidence_distribution": {
            str(label): int((interactions["relevance_label"] == label).sum())
            for label in [1, 2, 3]
        }
    }

    with open(os.path.join(MATRIX_DIR, "matrix_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MATRIX SUMMARY")
    print("=" * 60)

    print(f"""
Matrix shape   : {n_users:,} × {n_items:,}
Non-zero cells : {nnz:,}
Density        : {density:.4%}

This means {density:.2%} of all possible (user, item) pairs
have a known preference. The remaining {1-density:.2%} are unknown
— ALS will learn to predict scores for all of them.

Interactions per user:
  min    = {stats['interactions_per_user']['min']}
  max    = {stats['interactions_per_user']['max']}
  mean   = {stats['interactions_per_user']['mean']}
  median = {stats['interactions_per_user']['median']}

Interactions per item:
  min    = {stats['interactions_per_item']['min']}
  max    = {stats['interactions_per_item']['max']}
  mean   = {stats['interactions_per_item']['mean']}
  median = {stats['interactions_per_item']['median']}

Confidence weight distribution:
  Label 1 (click)     → confidence={LABEL_TO_CONFIDENCE[1]}:   {stats['confidence_distribution']['1']:,} pairs
  Label 2 (cart)      → confidence={LABEL_TO_CONFIDENCE[2]}:   {stats['confidence_distribution']['2']:,} pairs
  Label 3 (purchase)  → confidence={LABEL_TO_CONFIDENCE[3]}: {stats['confidence_distribution']['3']:,} pairs
""")

    print("Files saved:")
    for fname in ["interaction_matrix.npz", "user_index.parquet",
                  "item_index.parquet", "matrix_stats.json"]:
        path = os.path.join(MATRIX_DIR, fname)
        print(f"  → {path}")

if __name__ == "__main__":
    main()