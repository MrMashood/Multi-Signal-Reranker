# Step 01 — Synthetic Data Generation

**Branch:** `step/01-data-generation`  
**Input:** None  
**Output:** `data/raw/users.parquet`, `data/raw/items.parquet`, `data/raw/events.parquet`  
**Next step:** `step/02-feature-engineering`

---

## What This Step Does

Before we can train a reranking model, we need behavioral data — the kind an ecommerce platform
collects over months of real user traffic. Since we don't have a live platform, we simulate it.

This script generates three datasets that together represent a realistic ecommerce behavioral
log: users with preferences, items with attributes, and a record of every impression, click,
cart add, and purchase across 50,000 simulated sessions.

The data is intentionally designed to have **learnable patterns** — not random noise. The model
trained in Step 3 will need to discover these patterns from behavioral signals alone.

---

## Why Synthetic Data (and Not a Public Dataset)?

Public ecommerce datasets (like Coveo or Instacart) don't expose the ground-truth relevance
signals we need to properly evaluate a reranker. By generating data ourselves we get:

- Full control over the signal-to-noise ratio
- A known ground truth (`quality_score`) to validate against
- Position bias baked in at a known, measurable level
- Both search and homepage sessions in one dataset

---

## The Three Output Files

### `users.parquet`
5,000 users. Each user has hidden preferences that drive their behavior throughout the simulation.

| Column | Type | Description |
|---|---|---|
| `user_id` | string | Unique user identifier (`u_00001`) |
| `primary_category` | string | Category the user shops most (electronics, clothing, etc.) |
| `secondary_category` | string | Secondary interest category |
| `price_bucket` | string | Budget sensitivity: `low`, `mid`, or `high` |
| `preferred_brand` | string | Brand this user slightly favors |
| `brand_loyalty` | float | 0–1 score. Higher = stronger brand preference |
| `activity_level` | string | How frequently this user shops: `low`, `mid`, `high` |
| `signup_date` | date | Account creation date |

### `items.parquet`
2,000 products across 7 categories.

| Column | Type | Description |
|---|---|---|
| `item_id` | string | Unique item identifier (`i_00001`) |
| `category` | string | electronics / clothing / home / sports / beauty / books / toys |
| `brand` | string | One of 10 brands |
| `price` | float | Price in USD |
| `price_bucket` | string | `low` ($5–50), `mid` ($50–200), `high` ($200–1500) |
| `quality_score` | float | **Ground truth only.** 0–1. Drives true relevance. Not used as a training feature. |
| `date_listed` | date | When the item was listed |
| `title` | string | Product title |

> ⚠️ `quality_score` is the hidden ground truth. It is intentionally excluded from all
> feature engineering steps. The model must infer item quality from behavioral signals
> (CTR, conversion rate) — not from this column directly.

### `events.parquet`
~500,000 rows. One row per item shown per session (50,000 sessions × 10 items each).
This is the core behavioral log — the raw material for all downstream steps.

| Column | Type | Description |
|---|---|---|
| `event_id` | string | Unique event identifier |
| `session_id` | string | Groups all events in one user session |
| `user_id` | string | Which user |
| `item_id` | string | Which item was shown |
| `session_type` | string | `search` or `homepage` |
| `query_string` | string | Search query (null for homepage sessions) |
| `query_category` | string | Category the query maps to |
| `position` | int | Rank position the item was displayed at (1–10) |
| `was_clicked` | bool | Did the user click this item? |
| `was_carted` | bool | Did the user add to cart? |
| `was_purchased` | bool | Did the user purchase? |
| `relevance_label` | int | **Training target.** 0=ignored, 1=clicked, 2=carted, 3=purchased |
| `timestamp` | datetime | When the event occurred |

---

## The Core Design: Position Bias

This is the most important concept in the dataset. Items are shown in **random order** —
position does not reflect quality. A high-quality item can appear at rank 8; a mediocre
item can appear at rank 1.

Users are much more likely to click rank 1 than rank 10, regardless of quality:

```
Rank  1: ~25% CTR   ████████████████████████
Rank  2: ~21% CTR   █████████████████████
Rank  3: ~18% CTR   ██████████████████
Rank  4: ~15% CTR   ███████████████
Rank  5: ~12% CTR   ████████████
Rank  6: ~10% CTR   ██████████
Rank  7: ~8%  CTR   ████████
Rank  8: ~6%  CTR   ██████
Rank  9: ~5%  CTR   █████
Rank 10: ~3%  CTR   ███
```

This means raw click data is **biased** — items at the top look better than they are
simply because they were shown first. The reranking model's job is to learn true item
quality from cross-session patterns and correct for this distortion.

---

## Behavioral Signal Hierarchy

The simulation encodes four signals into the data, in descending strength:

```
1. Item quality       → items with higher quality_score get more clicks across all sessions
2. Category match     → users click items that match their query or primary_category more
3. Price match        → users click items within their price_bucket more
4. Brand affinity     → users with high brand_loyalty slightly favor their preferred_brand
5. Position bias      → rank 1 always gets more clicks regardless of signals 1–4
```

The goal of the model (Step 3) is to learn signals 1–4 and suppress signal 5.

---

## Expected Output When You Run the Script

```
============================================================
Generating synthetic ecommerce data
============================================================

[1/3] 5,000 users...       Shape: (5000, 8)
[2/3] 2,000 items...       Shape: (2000, 8)
[3/3] 50,000 sessions...
  5,000 / 50,000 sessions done...
  10,000 / 50,000 sessions done...
  ...
  Shape: (500000, 13)

============================================================
SUMMARY
============================================================

Impressions : 500,000
Clicks      : ~62,000    CTR=~12.4%
Cart adds   : ~8,500     Cart/Click=~13.7%
Purchases   : ~1,800     CVR=~2.9%

Label distribution:
  Label 0:  438,000  ████████████████████████████████████████████████████
  Label 1:   52,000  ██████
  Label 2:    8,500  █
  Label 3:    1,800  

CTR by rank position (search only):
  Rank  1: ~25.00%  ███████████████████████████████████████████████████████████████████████████
  Rank  2: ~21.00%  ███████████████████████████████████████████████████████████████
  ...
  Rank 10:  ~3.00%  █████████
```

If you see CTR clearly declining from rank 1 to rank 10, position bias is working correctly.

---

## How to Run

```bash
# 1. Install dependencies (one time)
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn

# 2. Run from project root
python scripts/01_generate_synthetic_data.py
```

Files will be saved to `data/raw/`. Total runtime: ~60–90 seconds.

---

## Project Structure at This Stage

```
reranker/
├── data/
│   └── raw/
│       ├── users.parquet       ← generated here
│       ├── items.parquet       ← generated here
│       └── events.parquet      ← generated here
└── scripts/
    └── 01_generate_synthetic_data.py
```

---

## What Changes Between Sessions

Each monthly training run (simulating the Airflow DAG from the architecture) would regenerate
this data from the production event log. In our project, re-running this script with a
different `SEED` value simulates a fresh month of data — useful for testing whether
the model generalizes across time periods.

---

*Continue to → [`step/02-feature-engineering`](../step/02-feature-engineering/README.md)*