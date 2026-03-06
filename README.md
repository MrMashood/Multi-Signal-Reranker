# Step 03a — Build User-Item Interaction Matrix

**Branch:** `step/03-model-training`  
**Input:** `data/features/training_data.parquet`  
**Output:** `data/matrix/interaction_matrix.npz`, `user_index.parquet`, `item_index.parquet`, `matrix_stats.json`  
**Next:** Step 03b — ALS training

---

## What This Step Does

Transforms the feature-engineered training table into a **sparse matrix R** that
ALS can factorize. This is the data structure at the heart of collaborative filtering.

```
training_data.parquet          sparse matrix R
(500k rows)          →         (n_users × n_items)
user_id | item_id | label      value = confidence weight c_ui
```

---

## The Core Concept: From Events to a Matrix

Raw events are a long list of (user, item, interaction) records. ALS needs a matrix
where rows are users, columns are items, and each cell holds a number representing
how strongly a user prefers an item.

```
         i_00001  i_00002  i_00003  i_00004  ...
u_00001  [  161  ] [   0  ] [  41  ] [   0  ]
u_00002  [   0  ] [  81  ] [   0  ] [   0  ]
u_00003  [  41  ] [   0  ] [   0  ] [ 161  ]
   ...
```

Most cells are zero — users only interact with a tiny fraction of all items.
This is why we use a **sparse matrix** (only stores non-zero values).

---

## Implicit Feedback: Preference vs Confidence

This is the most important conceptual shift from traditional recommendation systems.

In explicit feedback systems (like Netflix star ratings), users directly say "I rate
this 4/5." We don't have that. We have implicit signals — clicks, cart adds, purchases
— which tell us a user *probably* likes something but never say they *don't* like it.

The Hu, Koren & Volinsky (2008) paper formalizes this with two variables per (user, item) pair:

### Preference `p_ui`
Binary. Did the user show any preference for this item?

```
relevance_label > 0  →  p_ui = 1  (user interacted — we think they like it)
relevance_label = 0  →  p_ui = 0  (user was shown it but ignored it — unknown)
```

Note: `p_ui = 0` does NOT mean the user dislikes the item. They may simply not have
noticed it, or it was shown at a bad position. This ambiguity is why we don't store
negative signals — we call them "unknown" instead.

### Confidence `c_ui`
How strongly do we trust this preference signal?

```
c_ui = 1 + alpha × interaction_strength
```

| Event | Label | Confidence (alpha=40) | Interpretation |
|---|---|---|---|
| Shown, ignored | 0 | 1.0 | Almost no signal |
| Clicked | 1 | 41 | Mild preference |
| Added to cart | 2 | 81 | Strong preference |
| Purchased | 3 | 161 | Strongest signal |

A purchase is treated as 4× more trustworthy than a click. The matrix stores
these confidence values — not the raw labels.

### Why alpha=40?
Alpha controls the gap between "interacted" and "not interacted." Too low and the
model can't distinguish signal from noise. Too high and it overfits to observed
interactions. 40 is the value from the original Hu et al. paper and a standard
starting point — it can be tuned in later iterations.

---

## Handling Multiple Sessions

A user might see the same item in multiple sessions. We resolve this by taking
the **maximum label** across all sessions:

```
user u_00042 saw item i_00301:
  Session 1: was_clicked=True,  label=1
  Session 2: was_purchased=True, label=3
  → final label = 3 (purchase wins)
```

This prevents a single purchase from being diluted by earlier browse sessions.

---

## The Sparsity Problem

With 5,000 users and 2,000 items, the full matrix would have 10,000,000 cells.
Only a small fraction will be non-zero — this is called **sparsity**, and it's
the core challenge ALS is designed to solve.

```
Density ≈ 0.5–2%   →   ~50,000–200,000 known preferences
          98–99.5% unknown
```

ALS learns to fill in the unknowns by finding patterns:
*"Users who bought A and B also tend to buy C — so predict high score for C
for any user who bought A and B but hasn't seen C yet."*

---

## Output Files

| File | Description |
|---|---|
| `interaction_matrix.npz` | Sparse CSR matrix R, shape (n_users × n_items), scipy format |
| `user_index.parquet` | Maps `user_id` string → integer row index |
| `item_index.parquet` | Maps `item_id` string → integer column index |
| `matrix_stats.json` | Shape, density, confidence distribution stats |

The index files are critical — they allow us to go back from matrix row/column
numbers to real user_id and item_id strings at serving time.

---

## How to Run

```bash
python scripts/03a_interaction_matrix.py
```

Expected runtime: ~10 seconds.

---

## Project Structure at This Stage

```
reranker/
├── data/
│   ├── raw/            (Step 1 outputs)
│   ├── features/       (Step 2 outputs)
│   └── matrix/
│       ├── interaction_matrix.npz    ← generated here
│       ├── user_index.parquet        ← generated here
│       ├── item_index.parquet        ← generated here
│       └── matrix_stats.json         ← generated here
└── scripts/
    ├── 01_generate_synthetic_data.py
    ├── 02_feature_engineering.py
    └── 03a_interaction_matrix.py
```
---

# Step 03b — ALS Model Training (From Scratch)

---

## What This Step Does

Trains an Alternating Least Squares (ALS) model on the sparse interaction
matrix from Step 3a. The output is two embedding matrices:

- `user_vectors.npy` — one 32-dimensional vector per user
- `item_vectors.npy` — one 32-dimensional vector per item

The dot product of any user vector and item vector produces a relevance
score. These scores are what drive the reranker in Step 3c.

---

## The Math

### The Problem We're Solving

We have a sparse matrix **R** (5,000 × 2,000) where most entries are zero
(unknown) and non-zero entries hold confidence weights from Step 3a.

We want to find two smaller matrices:

```
R  ≈  U  ×  Iᵀ
(N×M)  (N×K)  (K×M)

N = 5,000 users
M = 2,000 items
K = 32    latent factors
```

Each row of **U** is a user's taste encoded as 32 numbers.
Each row of **I** is an item's attributes encoded as 32 numbers.
Their dot product approximates how much a user would like an item.

### Why Not Gradient Descent?

We could minimize the reconstruction error with gradient descent, but
implicit feedback matrices have a problem: we have **10 million cells**
(most unknown) and computing the full loss requires visiting all of them.

ALS sidesteps this. If you fix **I**, the optimal **U** has a closed-form
solution — no gradient descent needed. And vice versa. So we alternate:

```
Iteration 1:  Fix I (random) → solve for U analytically
Iteration 2:  Fix U          → solve for I analytically
Iteration 3:  Fix I          → solve for U analytically
...repeat 15 times
```

Each solve is an independent linear system per entity — fast and parallelizable.

### The Closed-Form Solution

For each user `u`, fixing all item vectors **I**:

```
A  =  Iᵀ C_u I  +  λ·Identity
b  =  Iᵀ C_u p_u

user_vector_u  =  A⁻¹ b
```

Where:
- `C_u` = diagonal matrix of confidence weights for user u's interactions
- `p_u` = preference vector (1 for interacted items, 0 otherwise)
- `λ`   = regularization strength (prevents overfitting)

The same formula applies symmetrically for item vectors.

### The Sparsity Optimization

Naively, `Iᵀ C_u I` would require multiplying all item vectors by the
full confidence matrix — O(M²K) per user. Expensive.

We use the identity:

```
Iᵀ C_u I  =  Iᵀ I  +  Iᵀ (C_u - Identity) I
```

The first term `Iᵀ I` is precomputed once and shared across all users.
The second term only involves non-zero entries of `C_u` (the items user u
actually interacted with) — typically 10-50 items, not 2,000.

This reduces the per-user cost to O(K² × nnz_u) — very fast.

---

## Hyperparameters

| Parameter | Value | What it controls |
|---|---|---|
| `K` | 32 | Embedding dimension. Higher K = more expressive but slower and more prone to overfitting |
| `n_iters` | 15 | How many alternating steps. Usually converges in 10–20 |
| `lambda_reg` | 0.01 | L2 regularization. Higher = stronger penalty on large embeddings |
| `alpha` | 40 | Set in Step 3a. Controls confidence gap between interactions and non-interactions |

### How to tune K
- Too low (K=4): model underfits, can't capture nuanced preferences
- Too high (K=128): overfits to training data, slow to train
- K=32 is a reliable starting point for datasets of this size

---

## What the Training Log Tells You

Each iteration prints:

```
  Iter   Loss         Δ Loss       Time
  ────── ──────────── ──────────── ────────
  1      4,823,102.00    +0.00      3.2s
  2      3,910,445.00  -912,657.00  3.1s
  3      3,654,221.00  -256,224.00  3.0s
  ...
  15     3,401,882.00   -12,043.00  3.0s
```

**What healthy training looks like:**
- Loss drops sharply in early iterations then flattens — normal convergence
- Δ Loss should always be negative (loss decreasing)
- By iteration 10–15, Δ Loss should be small — model has converged

**Warning signs:**
- Loss increasing → lambda_reg too small, try 0.1
- Loss not decreasing at all → check matrix has non-zero entries
- Norms exploding (>100) → regularization too weak

---

## The Output Embeddings

After training, each user and item is represented as a 32-dimensional vector:

```python
user_vectors[42]  =  [0.12, -0.34, 0.07, 0.89, ..., -0.23]   # user u_00042
item_vectors[301] =  [0.45,  0.11, -0.67, 0.33, ...,  0.18]  # item i_00301

# Relevance score:
score = np.dot(user_vectors[42], item_vectors[301])
```

Users with similar tastes will have similar vectors. Items in the same category
bought by similar users will have similar vectors. The dot product captures
how well a user's taste aligns with an item's attributes.

---

## How to Run

```bash
python scripts/03b_als_model.py
```

Expected runtime: ~45–90 seconds for 15 iterations on this dataset size.

---

## Project Structure at This Stage

```
reranker/
├── data/
│   ├── raw/
│   ├── features/
│   ├── matrix/
│   └── model/
│       ├── user_vectors.npy      ← (5000 × 32) float64
│       ├── item_vectors.npy      ← (2000 × 32) float64
│       └── training_log.json     ← loss per iteration
└── scripts/
    ├── 01_generate_synthetic_data.py
    ├── 02_feature_engineering.py
    ├── 03a_interaction_matrix.py
    └── 03b_als_model.py
```
---

# Step 03c — Reranking with ALS Embeddings + NDCG@K Evaluation

---

## What This Step Does

Uses the trained ALS embeddings to rerank items within each session,
then evaluates ranking quality with NDCG@K against two baselines.

This is the first time we get a number that answers the question:
**"Is the model actually useful?"**

---

## How Reranking Works

For every session in the dataset, 10 items were shown to a user.
Reranking means re-ordering those 10 items using the model's scores
instead of their original shown position.

```
Original order (position bias):       ALS reranked order:
  Rank 1: item_idx=42  label=0          Rank 1: item_idx=7   label=3  ← purchase
  Rank 2: item_idx=7   label=3          Rank 2: item_idx=15  label=2  ← cart
  Rank 3: item_idx=15  label=2          Rank 3: item_idx=42  label=0
  ...                                   ...
```

A good reranker moves high-label items (purchases, carts) to the top.
NDCG@K measures how well it does this.

---

## Scoring: The Dot Product

For each (user, item) pair in a session:

```python
score = user_vectors[user_idx] · item_vectors[item_idx]
      = Σ user_vectors[u][k] * item_vectors[i][k]   for k in 0..31
```

This is a single matrix multiply — extremely fast even for large catalogs.
Higher score = model predicts stronger user preference for this item.

---

## NDCG@K — How It's Computed

NDCG (Normalized Discounted Cumulative Gain) measures ranking quality
on a scale of 0 to 1.

### Step 1: DCG@K
```
DCG@K = Σ (2^label - 1) / log2(rank + 1)   for rank 1..K
```

Rewards putting high-label items early. The log2 discount means rank 1
is worth much more than rank 5:

```
Rank 1 discount: 1/log2(2) = 1.000
Rank 2 discount: 1/log2(3) = 0.631
Rank 3 discount: 1/log2(4) = 0.500
Rank 5 discount: 1/log2(6) = 0.387
Rank 10 discount:1/log2(11)= 0.289
```

### Step 2: IDCG@K (Ideal DCG)
Sort labels in descending order → compute DCG. This is the best
possible score for this session.

### Step 3: NDCG@K
```
NDCG@K = DCG@K / IDCG@K   → value in [0, 1]
```

A score of 1.0 means the ranker produced the perfect ordering.
A score of 0.5 means it captured about half the possible gain.

### Why K=10?
In ecommerce, users typically see 10 items on the first page.
NDCG@10 measures quality of exactly what the user experiences.
We also report @5 (above the fold) and @20 (second page).

---

## The Two Baselines

### Baseline A — Original Position
Items scored by inverse shown position (position 1 = score 10).
This is what the user actually saw — no reranking applied.

**If ALS beats this**, the model is correcting for position bias.
It's finding items that were buried at rank 8–10 but deserved rank 1–3.

### Baseline B — Random
Items scored randomly within each session.
This is the absolute floor — any model should beat this.

---

## The Position Bias Correction Check

The most important diagnostic in this step. We group all ALS scores
by the item's original shown position:

```
Position  1: avg ALS score = ?
Position  2: avg ALS score = ?
...
Position 10: avg ALS score = ?
```

**If scores are NOT decreasing by position** — the model learned
something real. It's assigning high scores to items based on their
quality signals (CTR, category match) rather than just echoing
the original position order back.

**If scores ARE strictly decreasing by position** — the model
overfit to position bias and learned nothing useful.

---

## Expected Results

```
────────────────────────────────────────────────────────────
  SEARCH SESSIONS  (~32,500 sessions)
────────────────────────────────────────────────────────────
  Ranker                    NDCG@5    NDCG@10   NDCG@20
  ───────────────────────── ──────── ──────── ────────
  ALS (ours)                0.65+    0.67+    0.70+
  Original position         ~0.50    ~0.52    ~0.55
  Random                    ~0.45    ~0.47    ~0.50
```

The exact numbers depend on random seed and data. What matters is the ordering:
**ALS > Original Position > Random**

---

## How to Run

```bash
python scripts/03c_rerank.py
```

Expected runtime: ~20 seconds.

---

## Project Structure at This Stage

```
reranker/
├── data/
│   ├── raw/
│   ├── features/
│   ├── matrix/
│   ├── model/
│   └── eval/
│       ├── rerank_results.parquet     ← per-row scores + labels
│       └── evaluation_summary.json   ← NDCG@K table
└── scripts/
    ├── 01_generate_synthetic_data.py
    ├── 02_feature_engineering.py
    ├── 03a_interaction_matrix.py
    ├── 03b_als_model.py
    └── 03c_rerank.py
```

---