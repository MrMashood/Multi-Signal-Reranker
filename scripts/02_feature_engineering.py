"""
Step 2: Feature Engineering with DuckDB
========================================
Reads raw parquet files, runs SQL transformations,
and produces a training_data.parquet ready for LightGBM LambdaRank.

Input  : data/raw/users.parquet, items.parquet, events.parquet
Output : data/features/training_data.parquet

Run:
    python scripts/02_feature_engineering.py
"""

import os
import duckdb
import pandas as pd

RAW_DIR      = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
FEATURE_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

EVENTS_PATH  = os.path.join(RAW_DIR, "events.parquet")
ITEMS_PATH   = os.path.join(RAW_DIR, "items.parquet")
USERS_PATH   = os.path.join(RAW_DIR, "users.parquet")
OUTPUT_PATH  = os.path.join(FEATURE_DIR, "training_data.parquet")


def run(con: duckdb.DuckDBPyConnection):

    # ── Load raw files into DuckDB views ─────────────────────────────────────
    con.execute(f"CREATE OR REPLACE VIEW events AS SELECT * FROM read_parquet('{EVENTS_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW items  AS SELECT * FROM read_parquet('{ITEMS_PATH}')")
    con.execute(f"CREATE OR REPLACE VIEW users  AS SELECT * FROM read_parquet('{USERS_PATH}')")
    print("  ✓ Loaded raw parquet views")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE BUCKET 1: Item Popularity Features
    # How good is this item globally, across all users and queries?
    # These are item-level aggregates — one row per item.
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE feat_item_popularity AS
        SELECT
            item_id,

            -- How often is this item clicked when shown?
            ROUND(SUM(was_clicked::INT)::FLOAT / COUNT(*), 6)
                AS item_global_ctr,

            -- Of clicks, how many convert to purchase?
            ROUND(
                SUM(was_purchased::INT)::FLOAT /
                NULLIF(SUM(was_clicked::INT), 0)
            , 6) AS item_global_cvr,

            -- Average position this item is shown at when clicked
            -- Low value = users click it even from bad positions = strong item
            ROUND(AVG(CASE WHEN was_clicked THEN position END), 4)
                AS item_avg_position_when_clicked,

            -- Total impressions (helps filter low-exposure items)
            COUNT(*) AS item_impression_count,

            -- Total purchases
            SUM(was_purchased::INT) AS item_total_purchases

        FROM events
        GROUP BY item_id
    """)
    print("  ✓ feat_item_popularity")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE BUCKET 2: Query-Item Features
    # How does this item perform specifically for this query?
    # These are (query, item) pair aggregates.
    # Only applies to search sessions — null for homepage.
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE feat_query_item AS
        SELECT
            query_string,
            item_id,

            -- CTR for this item when this specific query was searched
            ROUND(SUM(was_clicked::INT)::FLOAT / COUNT(*), 6)
                AS query_item_ctr,

            -- CVR for this item under this query
            ROUND(
                SUM(was_purchased::INT)::FLOAT /
                NULLIF(SUM(was_clicked::INT), 0)
            , 6) AS query_item_cvr,

            -- How often does this item appear for this query?
            COUNT(*) AS query_item_impressions,

            -- Average rank this item appears at for this query
            ROUND(AVG(position), 4) AS query_item_avg_position

        FROM events
        WHERE session_type = 'search'
          AND query_string IS NOT NULL
        GROUP BY query_string, item_id
    """)
    print("  ✓ feat_query_item")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE BUCKET 3: User-Item Interaction Features
    # Has this specific user interacted with this specific item before?
    # These capture personal history signals.
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE feat_user_item AS
        SELECT
            user_id,
            item_id,

            -- Total times this user clicked this item
            SUM(was_clicked::INT)   AS user_item_click_count,

            -- Did this user ever purchase this item?
            MAX(was_purchased::INT) AS user_item_purchased_before,

            -- Last time user interacted with this item
            MAX(timestamp)          AS user_item_last_interaction

        FROM events
        WHERE was_clicked = TRUE
        GROUP BY user_id, item_id
    """)
    print("  ✓ feat_user_item")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE BUCKET 4: User Affinity Features
    # What categories and price buckets does this user gravitate towards?
    # Computed from purchase history — stronger signal than clicks.
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE feat_user_affinity AS
        WITH user_category_counts AS (
            SELECT
                e.user_id,
                i.category,
                SUM(e.was_clicked::INT)    AS clicks_in_category,
                SUM(e.was_purchased::INT)  AS purchases_in_category
            FROM events e
            JOIN items i USING (item_id)
            GROUP BY e.user_id, i.category
        ),
        user_totals AS (
            SELECT
                user_id,
                SUM(clicks_in_category)    AS total_clicks,
                SUM(purchases_in_category) AS total_purchases
            FROM user_category_counts
            GROUP BY user_id
        ),
        user_top_category AS (
            -- The category with the most purchases per user
            SELECT DISTINCT ON (user_id)
                user_id,
                category AS user_top_purchase_category
            FROM user_category_counts
            ORDER BY user_id, purchases_in_category DESC
        ),
        user_price_affinity AS (
            -- The price bucket this user most often buys from
            SELECT DISTINCT ON (e.user_id)
                e.user_id,
                i.price_bucket AS user_preferred_price_bucket
            FROM events e
            JOIN items i USING (item_id)
            WHERE e.was_purchased = TRUE
            GROUP BY e.user_id, i.price_bucket
            ORDER BY e.user_id, COUNT(*) DESC
        )
        SELECT
            uc.user_id,
            uc.category,
            ROUND(uc.clicks_in_category::FLOAT    / NULLIF(ut.total_clicks, 0),    4) AS user_category_click_share,
            ROUND(uc.purchases_in_category::FLOAT / NULLIF(ut.total_purchases, 0), 4) AS user_category_purchase_share,
            utc.user_top_purchase_category,
            upa.user_preferred_price_bucket
        FROM user_category_counts uc
        JOIN user_totals ut           USING (user_id)
        LEFT JOIN user_top_category utc  USING (user_id)
        LEFT JOIN user_price_affinity upa USING (user_id)
    """)
    print("  ✓ feat_user_affinity")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE BUCKET 5: User Activity Features
    # General engagement profile of this user.
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE feat_user_activity AS
        SELECT
            user_id,
            COUNT(DISTINCT session_id)                          AS user_total_sessions,
            SUM(was_clicked::INT)                               AS user_total_clicks,
            SUM(was_purchased::INT)                             AS user_total_purchases,
            ROUND(SUM(was_clicked::INT)::FLOAT   / COUNT(*), 6) AS user_overall_ctr,
            ROUND(SUM(was_purchased::INT)::FLOAT /
                  NULLIF(SUM(was_clicked::INT),0), 6)           AS user_overall_cvr,
            COUNT(DISTINCT item_id)                             AS user_unique_items_seen
        FROM events
        GROUP BY user_id
    """)
    print("  ✓ feat_user_activity")

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL JOIN: Build the training table
    #
    # One row = one (session, item) impression
    # All feature buckets joined together
    # query_string serves as the LTR "group key" for search sessions
    # session_id serves as the group key for homepage sessions
    # ─────────────────────────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE training_data AS
        SELECT
            -- ── Identifiers (not features, used for grouping) ──────────────
            e.session_id,
            e.user_id,
            e.item_id,
            e.session_type,
            e.query_string,
            e.position          AS shown_position,

            -- ── Label ──────────────────────────────────────────────────────
            e.relevance_label,

            -- ── Item metadata (categorical, not raw features) ──────────────
            i.category          AS item_category,
            i.price_bucket      AS item_price_bucket,
            i.price             AS item_price,

            -- ── FEATURE BUCKET 1: Item Popularity ─────────────────────────
            COALESCE(ip.item_global_ctr,                  0) AS item_global_ctr,
            COALESCE(ip.item_global_cvr,                  0) AS item_global_cvr,
            COALESCE(ip.item_avg_position_when_clicked,   5) AS item_avg_position_when_clicked,
            COALESCE(ip.item_impression_count,            0) AS item_impression_count,
            COALESCE(ip.item_total_purchases,             0) AS item_total_purchases,

            -- ── FEATURE BUCKET 2: Query-Item (search only) ────────────────
            COALESCE(qi.query_item_ctr,          0) AS query_item_ctr,
            COALESCE(qi.query_item_cvr,          0) AS query_item_cvr,
            COALESCE(qi.query_item_impressions,  0) AS query_item_impressions,
            COALESCE(qi.query_item_avg_position, 5) AS query_item_avg_position,

            -- ── FEATURE BUCKET 3: User-Item History ───────────────────────
            COALESCE(ui.user_item_click_count,       0) AS user_item_click_count,
            COALESCE(ui.user_item_purchased_before,  0) AS user_item_purchased_before,

            -- ── FEATURE BUCKET 4: User Category Affinity ──────────────────
            COALESCE(ua.user_category_click_share,    0) AS user_category_click_share,
            COALESCE(ua.user_category_purchase_share, 0) AS user_category_purchase_share,

            -- ── FEATURE BUCKET 4b: Price affinity match ───────────────────
            -- Binary: does this item's price bucket match user's preferred bucket?
            CASE
                WHEN ua.user_preferred_price_bucket = i.price_bucket THEN 1
                ELSE 0
            END AS user_price_match,

            -- ── FEATURE BUCKET 4c: Category affinity match ────────────────
            -- Binary: is this item in the user's top purchase category?
            CASE
                WHEN ua.user_top_purchase_category = i.category THEN 1
                ELSE 0
            END AS user_top_category_match,

            -- ── FEATURE BUCKET 5: User Activity ───────────────────────────
            COALESCE(uact.user_total_sessions,    0) AS user_total_sessions,
            COALESCE(uact.user_total_clicks,      0) AS user_total_clicks,
            COALESCE(uact.user_total_purchases,   0) AS user_total_purchases,
            COALESCE(uact.user_overall_ctr,       0) AS user_overall_ctr,
            COALESCE(uact.user_overall_cvr,       0) AS user_overall_cvr

        FROM events e

        -- Item metadata
        JOIN items i
            ON e.item_id = i.item_id

        -- Feature bucket 1: item popularity
        LEFT JOIN feat_item_popularity ip
            ON e.item_id = ip.item_id

        -- Feature bucket 2: query-item (null for homepage rows)
        LEFT JOIN feat_query_item qi
            ON e.item_id      = qi.item_id
            AND e.query_string = qi.query_string

        -- Feature bucket 3: user-item history
        LEFT JOIN feat_user_item ui
            ON e.user_id = ui.user_id
            AND e.item_id = ui.item_id

        -- Feature bucket 4: user affinity (join on user + item category)
        LEFT JOIN feat_user_affinity ua
            ON e.user_id     = ua.user_id
            AND i.category   = ua.category

        -- Feature bucket 5: user activity
        LEFT JOIN feat_user_activity uact
            ON e.user_id = uact.user_id
    """)
    print("  ✓ training_data (final join)")


def summarize(con: duckdb.DuckDBPyConnection):
    print("\n" + "="*60)
    print("TRAINING TABLE SUMMARY")
    print("="*60)

    total = con.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
    print(f"\nTotal rows       : {total:,}")

    search = con.execute("SELECT COUNT(*) FROM training_data WHERE session_type='search'").fetchone()[0]
    home   = con.execute("SELECT COUNT(*) FROM training_data WHERE session_type='homepage'").fetchone()[0]
    print(f"Search rows      : {search:,}")
    print(f"Homepage rows    : {home:,}")

    print("\nLabel distribution:")
    rows = con.execute("""
        SELECT relevance_label, COUNT(*) as cnt,
               ROUND(COUNT(*)*100.0 / SUM(COUNT(*)) OVER(), 2) as pct
        FROM training_data
        GROUP BY relevance_label
        ORDER BY relevance_label
    """).fetchall()
    for label, cnt, pct in rows:
        bar = "█" * int(pct)
        print(f"  Label {label}: {cnt:>8,}  ({pct:5.2f}%)  {bar}")

    print("\nFeature null check (should all be 0 after COALESCE):")
    feature_cols = [
        "item_global_ctr", "item_global_cvr", "query_item_ctr",
        "user_item_click_count", "user_category_click_share",
        "user_overall_ctr", "user_price_match", "user_top_category_match"
    ]
    for col in feature_cols:
        nulls = con.execute(f"SELECT COUNT(*) FROM training_data WHERE {col} IS NULL").fetchone()[0]
        status = "✓" if nulls == 0 else f"⚠️  {nulls:,} nulls"
        print(f"  {col:<40} {status}")

    print("\nSample rows (search session):")
    sample = con.execute("""
        SELECT session_id, item_id, relevance_label,
               item_global_ctr, query_item_ctr,
               user_price_match, user_top_category_match
        FROM training_data
        WHERE session_type = 'search'
        LIMIT 5
    """).df()
    print(sample.to_string(index=False))

    print("\nSearch query groups (top 10 by session count):")
    qg = con.execute("""
        SELECT query_string,
               COUNT(DISTINCT session_id) AS sessions,
               COUNT(*) AS total_rows,
               ROUND(AVG(relevance_label), 4) AS avg_label
        FROM training_data
        WHERE session_type = 'search'
        GROUP BY query_string
        ORDER BY sessions DESC
        LIMIT 10
    """).df()
    print(qg.to_string(index=False))


def main():
    print("="*60)
    print("Step 2: Feature Engineering (DuckDB)")
    print("="*60)

    con = duckdb.connect()

    print("\nBuilding feature tables...")
    run(con)

    print("\nExporting training_data.parquet...")
    con.execute(f"COPY training_data TO '{OUTPUT_PATH}' (FORMAT PARQUET)")
    print(f"  ✓ Saved → {OUTPUT_PATH}")

    df = pd.read_parquet(OUTPUT_PATH)
    print(f"  ✓ Shape : {df.shape}")
    print(f"  ✓ Columns: {list(df.columns)}")

    summarize(con)
    con.close()

if __name__ == "__main__":
    main()