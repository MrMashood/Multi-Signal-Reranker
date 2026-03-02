# Step 02 — Feature Engineering with DuckDB

**Branch:** `step/02-feature-engineering`  
**Input:** `data/raw/users.parquet`, `data/raw/items.parquet`, `data/raw/events.parquet`  
**Output:** `data/features/training_data.parquet`  
**Next step:** `step/03-model-training`

---

## What This Step Does

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

---

## How to Run

```bash
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

---

## Project Structure at This Stage

```
reranker/
├── data/
│   ├── raw/
│   │   ├── users.parquet
│   │   ├── items.parquet
│   │   └── events.parquet
│   └── features/
│       └── training_data.parquet    ← generated here
└── scripts/
    ├── 01_generate_synthetic_data.py
    └── 02_feature_engineering.py
```

---

*Continue to → [`step/03-model-training`](../step/03-model-training/README.md)*