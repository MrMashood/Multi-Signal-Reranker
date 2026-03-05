"""
Step 3b: ALS Model Training (from scratch)
==========================================
Implements Alternating Least Squares for implicit feedback
as described in Hu, Koren & Volinsky (2008).

Math recap:
  R ≈ U × Iᵀ   where U = user embeddings, I = item embeddings

  Fix I, solve for each user vector u:
      u = (IᵀC_uI + λI)⁻¹ Iᵀ C_u p_u

  Fix U, solve for each item vector i:
      i = (UᵀC_iU + λI)⁻¹ Uᵀ C_i p_i

  Where:
      p   = preference (1 if interacted, 0 otherwise)
      C_u = diagonal confidence matrix for user u
      λ   = regularization (prevents overfitting)

Input  : data/matrix/interaction_matrix.npz
         data/matrix/user_index.parquet
         data/matrix/item_index.parquet
Output : data/model/user_vectors.npy    (N × K)
         data/model/item_vectors.npy    (M × K)
         data/model/training_log.json

Run:
    python scripts/03b_als_model.py
"""

import os
import json
import time
import numpy as np
import scipy.sparse as sp

MATRIX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "matrix")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
K          = 32     # number of latent factors (embedding dimension)
N_ITERS    = 15     # ALS alternating iterations
LAMBDA     = 1.00   # L2 regularization strength
RANDOM_SEED = 42


class ALSModel:
    """
    Implicit feedback Matrix Factorization via Alternating Least Squares.

    The core idea:
      We want to find U (users × K) and I (items × K) such that
      U @ I.T approximates the true preference matrix P.

      We can't solve for U and I simultaneously (non-convex).
      But if we FIX one, solving for the other becomes a
      standard least squares problem with a closed-form solution.

      ALS exploits this by alternating:
        → Fix I, solve for all user vectors (one linear system per user)
        → Fix U, solve for all item vectors (one linear system per item)
        → Repeat until convergence
    """

    def __init__(self, K: int = 32, n_iters: int = 15,
                 lambda_reg: float = 0.01, random_seed: int = 42):
        self.K          = K
        self.n_iters    = n_iters
        self.lambda_reg = lambda_reg
        self.random_seed = random_seed

        self.user_vectors = None   # shape: (n_users, K)
        self.item_vectors = None   # shape: (n_items, K)
        self.training_log = []

    def _init_embeddings(self, n_users: int, n_items: int):
        """
        Initialize embeddings with small random values.
        Small values (scale=0.01) prevent any single factor from
        dominating before training begins.
        """
        rng = np.random.RandomState(self.random_seed)
        self.user_vectors = rng.normal(scale=0.01, size=(n_users, self.K))
        self.item_vectors = rng.normal(scale=0.01, size=(n_items, self.K))

    def _solve_vectors(self, fixed: np.ndarray, R_slice: sp.csr_matrix,
                       lambda_reg: float) -> np.ndarray:
        """
        Core ALS solver. Given the fixed embedding matrix, solve for
        all vectors in the other matrix.

        For each entity u (user or item):

          A = fixedᵀ @ C_u @ fixed  +  λI
          b = fixedᵀ @ C_u @ p_u
          solved_vector = A⁻¹ b

        Where:
          C_u = diag(confidence weights for entity u)
          p_u = preference vector for entity u (binary)

        The key optimization: instead of building the full C_u matrix
        (which is n_items × n_items), we exploit sparsity:

          fixedᵀ C_u fixed = fixedᵀ fixed  +  fixedᵀ (C_u - I) fixed

        The second term only involves non-zero confidence entries,
        which are very few (sparse). This makes each solve O(K²·nnz_u)
        instead of O(K²·n_items).
        """
        n_entities = R_slice.shape[0]
        K          = fixed.shape[1]
        solved     = np.zeros((n_entities, K))

        # Precompute fixedᵀ @ fixed — shared across all entities
        # This is the "base" gram matrix before confidence weighting
        fixed_T_fixed = fixed.T @ fixed                         # (K × K)
        reg_matrix    = lambda_reg * np.eye(K)                  # (K × K)

        for u in range(n_entities):
            # Get non-zero entries for this user/item
            # R_slice[u] is a sparse row: non-zero cols = interacted items
            row        = R_slice[u]                             # sparse (1 × n_items)
            indices    = row.indices                            # which items
            confidences = row.data                              # confidence values c_ui

            if len(indices) == 0:
                # No interactions — can't solve, leave as zeros
                # (This user/item has no signal; embedding stays near init)
                solved[u] = np.zeros(K)
                continue

            # Preference vector p_u: all interacted items have preference=1
            # (we already filtered to only non-zero confidence entries)
            # p_u values are implicitly 1 for all entries in `indices`

            # Embeddings of interacted items/users
            fixed_u = fixed[indices]                            # (nnz_u × K)

            # Confidence - 1 for non-zero entries (the "extra" weight beyond baseline)
            c_minus_1 = confidences - 1.0                      # (nnz_u,)

            # Efficient computation of fixedᵀ C_u fixed:
            #   = fixedᵀ fixed  +  fixedᵀ (C_u - I) fixed
            # The second term is computed only over non-zero entries
            A = fixed_T_fixed + (fixed_u.T * c_minus_1) @ fixed_u + reg_matrix

            # Right-hand side: fixedᵀ C_u p_u
            # Since p_u = 1 for all interacted items:
            #   fixedᵀ C_u p_u = fixedᵀ c_u  (just sum weighted embeddings)
            b = fixed_u.T @ confidences                         # (K,)

            # Solve the linear system A @ x = b
            # np.linalg.solve is O(K³) — fast since K is small (32)
            solved[u] = np.linalg.solve(A, b)

        return solved

    def _compute_loss(self, R: sp.csr_matrix) -> float:
        """
        Compute training loss for monitoring convergence.

        Loss = Σ c_ui (p_ui - u_u · i_iᵀ)²  +  λ(||U||² + ||I||²)

        We approximate this over non-zero entries only (for speed).
        A decreasing loss across iterations = model is converging.
        """
        # Predicted scores for observed (non-zero) entries only
        R_coo    = R.tocoo()
        u_idx    = R_coo.row
        i_idx    = R_coo.col
        conf     = R_coo.data

        # Dot product for each observed pair
        preds    = np.sum(self.user_vectors[u_idx] * self.item_vectors[i_idx], axis=1)
        prefs    = np.ones(len(conf))   # p_ui = 1 for all non-zero entries

        # Weighted squared error over observed entries
        obs_loss = np.sum(conf * (prefs - preds) ** 2)

        # Regularization
        reg_loss = self.lambda_reg * (
            np.sum(self.user_vectors ** 2) +
            np.sum(self.item_vectors ** 2)
        )

        return float(obs_loss + reg_loss)

    def fit(self, R: sp.csr_matrix):
        """
        Train ALS model on sparse interaction matrix R.

        R shape: (n_users × n_items)
        R values: confidence weights c_ui (non-zero = interacted)
        """
        n_users, n_items = R.shape
        R_csr = R.tocsr()   # efficient row access (for user solve)
        R_csc = R.tocsc().T.tocsr()  # transposed, efficient row access (for item solve)

        print(f"\n  Hyperparameters:")
        print(f"    K (latent factors) = {self.K}")
        print(f"    Iterations         = {self.n_iters}")
        print(f"    Lambda (reg)       = {self.lambda_reg}")
        print(f"    Matrix shape       = {n_users} × {n_items}")

        # Initialize embeddings
        self._init_embeddings(n_users, n_items)
        print(f"\n  Initialized embeddings — U:{self.user_vectors.shape}, I:{self.item_vectors.shape}")
        print(f"\n  {'Iter':<6} {'Loss':>12} {'Δ Loss':>12} {'Time':>8}")
        print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*8}")

        prev_loss = None
        t_total   = time.time()

        for iteration in range(1, self.n_iters + 1):
            t_iter = time.time()

            # ── Step A: Fix item vectors, solve for user vectors ──────────
            self.user_vectors = self._solve_vectors(
                fixed    = self.item_vectors,
                R_slice  = R_csr,
                lambda_reg = self.lambda_reg
            )

            # ── Step B: Fix user vectors, solve for item vectors ──────────
            self.item_vectors = self._solve_vectors(
                fixed    = self.user_vectors,
                R_slice  = R_csc,
                lambda_reg = self.lambda_reg
            )

            # ── Compute loss ──────────────────────────────────────────────
            loss      = self._compute_loss(R_csr)
            delta     = (loss - prev_loss) if prev_loss is not None else 0.0
            iter_time = time.time() - t_iter

            print(f"  {iteration:<6} {loss:>12.2f} {delta:>+12.2f} {iter_time:>7.1f}s")

            self.training_log.append({
                "iteration": iteration,
                "loss":      round(loss, 4),
                "delta":     round(delta, 4),
                "time_s":    round(iter_time, 2),
            })
            prev_loss = loss

        total_time = time.time() - t_total
        print(f"\n  Total training time: {total_time:.1f}s")
        return self

    def score(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Score a set of items for a given user.
        Returns dot product of user vector with each item vector.
        Higher score = model thinks user will prefer this item.
        """
        u_vec = self.user_vectors[user_idx]                     # (K,)
        i_vecs = self.item_vectors[item_indices]                 # (n_items, K)
        return i_vecs @ u_vec                                    # (n_items,)

    def recommend(self, user_idx: int, n: int = 10,
                  exclude_known: np.ndarray = None) -> np.ndarray:
        """
        Return top-n item indices for a given user.
        Optionally exclude items the user has already interacted with.
        """
        all_scores = self.item_vectors @ self.user_vectors[user_idx]  # (n_items,)

        if exclude_known is not None:
            all_scores[exclude_known] = -np.inf

        return np.argsort(all_scores)[::-1][:n]

    def save(self, model_dir: str):
        np.save(os.path.join(model_dir, "user_vectors.npy"), self.user_vectors)
        np.save(os.path.join(model_dir, "item_vectors.npy"), self.item_vectors)
        with open(os.path.join(model_dir, "training_log.json"), "w") as f:
            json.dump({
                "hyperparameters": {
                    "K": self.K, "n_iters": self.n_iters,
                    "lambda_reg": self.lambda_reg
                },
                "training_log": self.training_log
            }, f, indent=2)
        print(f"\n  Saved user_vectors : {self.user_vectors.shape}")
        print(f"  Saved item_vectors : {self.item_vectors.shape}")

    @classmethod
    def load(cls, model_dir: str) -> "ALSModel":
        model = cls()
        model.user_vectors = np.load(os.path.join(model_dir, "user_vectors.npy"))
        model.item_vectors = np.load(os.path.join(model_dir, "item_vectors.npy"))
        with open(os.path.join(model_dir, "training_log.json")) as f:
            data = json.load(f)
        model.K          = data["hyperparameters"]["K"]
        model.n_iters    = data["hyperparameters"]["n_iters"]
        model.lambda_reg = data["hyperparameters"]["lambda_reg"]
        return model


def main():
    print("=" * 60)
    print("Step 3b: ALS Model Training")
    print("=" * 60)

    # ── Load interaction matrix ───────────────────────────────────────────
    print("\n[1/3] Loading interaction matrix...")
    R = sp.load_npz(os.path.join(MATRIX_DIR, "interaction_matrix.npz"))
    print(f"      Shape   : {R.shape}")
    print(f"      Non-zeros: {R.nnz:,}")

    # ── Train ALS ─────────────────────────────────────────────────────────
    print("\n[2/3] Training ALS model...")
    model = ALSModel(K=K, n_iters=N_ITERS, lambda_reg=LAMBDA,
                     random_seed=RANDOM_SEED)
    model.fit(R)

    # ── Save model ────────────────────────────────────────────────────────
    print("\n[3/3] Saving model...")
    model.save(MODEL_DIR)

    # ── Sanity checks ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # Check 1: embedding norms (should be moderate, not zero or exploding)
    u_norms = np.linalg.norm(model.user_vectors, axis=1)
    i_norms = np.linalg.norm(model.item_vectors, axis=1)
    print(f"\nUser vector norms   — mean: {u_norms.mean():.4f}  std: {u_norms.std():.4f}  max: {u_norms.max():.4f}")
    print(f"Item vector norms   — mean: {i_norms.mean():.4f}  std: {i_norms.std():.4f}  max: {i_norms.max():.4f}")

    # Check 2: score distribution for one user
    test_user = 0
    scores = model.score(test_user, np.arange(R.shape[1]))
    print(f"\nScore distribution for user 0:")
    print(f"  min={scores.min():.4f}  max={scores.max():.4f}  "
          f"mean={scores.mean():.4f}  std={scores.std():.4f}")

    # Check 3: top recommendations for user 0
    known_items = R[test_user].indices
    top_items   = model.recommend(test_user, n=5, exclude_known=known_items)
    print(f"\nTop 5 recommended items for user 0 (excluding {len(known_items)} known):")
    for rank, item_idx in enumerate(top_items, 1):
        score = scores[item_idx]
        print(f"  Rank {rank}: item_idx={item_idx:4d}  score={score:.4f}")

    # Check 4: loss should be decreasing
    losses = [entry["loss"] for entry in model.training_log]
    is_decreasing = all(losses[i] >= losses[i+1] for i in range(len(losses)-1))
    print(f"\nLoss monotonically decreasing: {'✅ Yes' if is_decreasing else '⚠️ No — check hyperparameters'}")
    print(f"  First iteration loss : {losses[0]:,.2f}")
    print(f"  Final iteration loss : {losses[-1]:,.2f}")
    print(f"  Total reduction      : {((losses[0]-losses[-1])/losses[0])*100:.1f}%")


if __name__ == "__main__":
    main()