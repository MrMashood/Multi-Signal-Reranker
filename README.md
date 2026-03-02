<<<<<<< Updated upstream
# Step 01 — Synthetic Data Generation

**Branch:** `step/01-data-generation`  
**Input:** None  
**Output:** `data/raw/users.parquet`, `data/raw/items.parquet`, `data/raw/events.parquet`  
**Next step:** `step/02-feature-engineering`
=======
# Step 02 — Feature Engineering with DuckDB

**Branch:** `step/02-feature-engineering`  
**Input:** `data/raw/users.parquet`, `data/raw/items.parquet`, `data/raw/events.parquet`  
**Output:** `data/features/training_data.parquet`  
**Next step:** `step/03-model-training`
>>>>>>> Stashed changes

---

## What This Step Does

<<<<<<< Updated upstream
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
=======
Takes the raw behavioral event log from Step 1 and transforms it into a structured
training table where every row has:

- A **group key** (`session_id` or `query_string`) — tells LambdaRank which rows compete against each other
- A **relevance label** (0–3) — the target the model learns to predict
- A set of **numerical features** — signals the model uses to rank items

This is the most SQL-heavy step in the project and the closest equivalent to what BigQuery
does in a production GCP pipeline.

---

## Why DuckDB?

DuckDB runs entirely in-process with Python — no server, no Docker, no credentials.
It reads Parquet files natively, supports full analytical SQL (window functions,
CTEs, lateral joins), and produces results fast enough for our dataset size.

The SQL written here is **intentionally identical** to what you would write in BigQuery,
with minor dialect differences. Migrating to BigQuery later requires almost no rewriting.

---

## The Feature Engineering Problem

Raw events look like this:

```
session_id | user_id | item_id | position | was_clicked | was_purchased
s_0000001  | u_00042 | i_00301 |    3     |    True     |    False
s_0000001  | u_00042 | i_00892 |    7     |    False    |    False
s_0000001  | u_00042 | i_00017 |    1     |    True     |    True
```

A model can't learn from this directly. It needs features that **aggregate behavior
across many sessions** to reveal signal. For example:

- `item_global_ctr = 0.18` tells the model this item gets clicked often in general
- `query_item_ctr = 0.31` tells the model this item is particularly popular for *this* query
- `user_top_category_match = 1` tells the model this item matches the user's taste

These aggregates are computed in SQL and joined back onto every impression row.

---

## The Five Feature Buckets

### Bucket 1 — Item Popularity
*How good is this item globally, across all users?*

Aggregated at the **item level**. One row per item.

| Feature | Description | Why it matters |
|---|---|---|
| `item_global_ctr` | clicks / impressions across all sessions | Items with high CTR are genuinely attractive |
| `item_global_cvr` | purchases / clicks globally | Converts clicks into money — strongest quality proxy |
| `item_avg_position_when_clicked` | avg rank at click time | If users click it even from rank 8, it's a strong item |
| `item_impression_count` | total times shown | Low count = unreliable stats, high count = trustworthy signal |
| `item_total_purchases` | total purchases | Absolute purchase volume |

### Bucket 2 — Query-Item Features
*How well does this item perform for this specific search query?*

Aggregated at the **(query, item) pair level**. Only populated for search sessions.
Homepage rows get `0` for all of these.

| Feature | Description | Why it matters |
|---|---|---|
| `query_item_ctr` | CTR for this item under this query | Captures query-specific relevance |
| `query_item_cvr` | CVR for this item under this query | Revenue signal per query |
| `query_item_impressions` | how often shown for this query | Reliability of the above stats |
| `query_item_avg_position` | avg rank for this query | Baseline from retrieval system |

> This is the most powerful bucket for search reranking. An item that converts well
> specifically for "running shoes" should rank high for that query even if it's mediocre
> globally.

### Bucket 3 — User-Item History
*Has this user interacted with this exact item before?*

Aggregated at the **(user, item) pair level**.

| Feature | Description | Why it matters |
|---|---|---|
| `user_item_click_count` | how many times user clicked this item | Repeat interest signal |
| `user_item_purchased_before` | binary: has user bought this? | If yes, probably don't resurface it |

### Bucket 4 — User Affinity
*What does this user tend to buy in general?*

Aggregated at the **(user, category) level**, then joined with item category.

| Feature | Description | Why it matters |
|---|---|---|
| `user_category_click_share` | % of user's clicks in this item's category | Preference signal |
| `user_category_purchase_share` | % of user's purchases in this category | Stronger preference signal |
| `user_top_category_match` | binary: is item in user's #1 category? | Direct match flag |
| `user_price_match` | binary: does item price bucket = user's preferred bucket? | Budget alignment |

### Bucket 5 — User Activity
*How engaged is this user overall?*

Aggregated at the **user level**.

| Feature | Description | Why it matters |
|---|---|---|
| `user_total_sessions` | lifetime session count | Power user vs. casual visitor |
| `user_total_clicks` | lifetime clicks | Engagement level |
| `user_total_purchases` | lifetime purchases | High-value user signal |
| `user_overall_ctr` | user's average CTR across all sessions | Clickiness baseline |
| `user_overall_cvr` | user's purchase rate per click | Buying intent baseline |

---

## The Final Training Table Schema

Every row = one item shown to one user in one session.

```
training_data.parquet
│
├── Identifiers (not fed to model)
│   ├── session_id          group key for homepage ranking
│   ├── user_id
│   ├── item_id
│   ├── session_type        'search' or 'homepage'
│   ├── query_string        group key for search ranking (null for homepage)
│   └── shown_position      original rank — used to verify position bias correction
│
├── Label
│   └── relevance_label     0=ignored, 1=clicked, 2=carted, 3=purchased
│
├── Item metadata (used for filtering/analysis, not direct features)
│   ├── item_category
│   ├── item_price_bucket
│   └── item_price
│
└── Features (30 numerical columns fed to LightGBM)
    ├── item_global_ctr
    ├── item_global_cvr
    ├── item_avg_position_when_clicked
    ├── item_impression_count
    ├── item_total_purchases
    ├── query_item_ctr
    ├── query_item_cvr
    ├── query_item_impressions
    ├── query_item_avg_position
    ├── user_item_click_count
    ├── user_item_purchased_before
    ├── user_category_click_share
    ├── user_category_purchase_share
    ├── user_price_match
    ├── user_top_category_match
    ├── user_total_sessions
    ├── user_total_clicks
    ├── user_total_purchases
    ├── user_overall_ctr
    └── user_overall_cvr
```

---

## An Important Note on Data Leakage

All features here are **aggregated over the entire event log** — including the session
being used as a training example. In a production system you would use a **training cutoff date**:

```sql
-- Production-safe version
WHERE timestamp < '2024-09-01'   -- only use past data to compute features
```

For this project we skip the cutoff to keep the code simple, but Step 3 will do a
**time-based train/validation split** to partially compensate for this.
>>>>>>> Stashed changes

---

## How to Run

```bash
<<<<<<< Updated upstream
# 1. Install dependencies (one time)
pip install pandas numpy pyarrow duckdb lightgbm scikit-learn

# 2. Run from project root
python scripts/01_generate_synthetic_data.py
```

Files will be saved to `data/raw/`. Total runtime: ~60–90 seconds.
=======
python scripts/02_feature_engineering.py
```

Expected runtime: ~30 seconds for 500k rows on a laptop.

---

## Expected Output

```
============================================================
Step 2: Feature Engineering (DuckDB)
============================================================

Building feature tables...
  ✓ Loaded raw parquet views
  ✓ feat_item_popularity
  ✓ feat_query_item
  ✓ feat_user_item
  ✓ feat_user_affinity
  ✓ feat_user_activity
  ✓ training_data (final join)

Exporting training_data.parquet...
  ✓ Saved → data/features/training_data.parquet
  ✓ Shape : (500000, 28)

TRAINING TABLE SUMMARY
============================================================

Total rows       : 500,000
Search rows      : 325,000
Homepage rows    : 175,000

Label distribution:
  Label 0:  438,000  (87.60%)  ███████████████████████████████████████████████████████████████████████████████████████
  Label 1:   52,000  (10.40%)  ██████████
  Label 2:    8,500  ( 1.70%)  █
  Label 3:    1,800  ( 0.36%)

Feature null check (should all be 0 after COALESCE):
  item_global_ctr                          ✓
  item_global_cvr                          ✓
  query_item_ctr                           ✓
  user_item_click_count                    ✓
  user_category_click_share                ✓
  user_overall_ctr                         ✓
  user_price_match                         ✓
  user_top_category_match                  ✓
```
>>>>>>> Stashed changes

---

## Project Structure at This Stage

```
reranker/
├── data/
<<<<<<< Updated upstream
│   └── raw/
│       ├── users.parquet       ← generated here
│       ├── items.parquet       ← generated here
│       └── events.parquet      ← generated here
└── scripts/
    └── 01_generate_synthetic_data.py
=======
│   ├── raw/
│   │   ├── users.parquet
│   │   ├── items.parquet
│   │   └── events.parquet
│   └── features/
│       └── training_data.parquet    ← generated here
└── scripts/
    ├── 01_generate_synthetic_data.py
    └── 02_feature_engineering.py
>>>>>>> Stashed changes
```

---

<<<<<<< Updated upstream
## What Changes Between Sessions

Each monthly training run (simulating the Airflow DAG from the architecture) would regenerate
this data from the production event log. In our project, re-running this script with a
different `SEED` value simulates a fresh month of data — useful for testing whether
the model generalizes across time periods.

---

*Continue to → [`step/02-feature-engineering`](../step/02-feature-engineering/README.md)*
=======
*Continue to → [`step/03-model-training`](../step/03-model-training/README.md)*
>>>>>>> Stashed changes
