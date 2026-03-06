"""
Step 1: Synthetic Ecommerce Data Generator
===========================================
Generates realistic ecommerce behavioral data with:
- User category preferences and price sensitivity
- Position bias in clicks (users click rank 1 more even if rank 5 is better)
- Category affinity (users prefer items matching their taste)
- Purchase behavior correlated with genuine item quality
- Both search sessions and homepage sessions

Output (saved to data/raw/):
  - users.parquet
  - items.parquet
  - events.parquet

Install requirements:
    pip install pandas numpy pyarrow duckdb lightgbm scikit-learn
"""

import numpy as np
import pandas as pd
import os
import random
from datetime import datetime, timedelta

SEED = int(datetime.now().strftime("%Y%m"))  # changes every month — different data each run
np.random.seed(SEED)
random.seed(SEED)

N_USERS          = 5_000
N_ITEMS          = 2_000
N_SESSIONS       = 50_000
SEARCH_RATIO     = 0.65
RESULTS_PER_PAGE = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CATEGORIES = ["electronics", "clothing", "home", "sports", "beauty", "books", "toys"]

SEARCH_QUERIES = {
    "laptop":              "electronics",
    "wireless headphones": "electronics",
    "smartphone":          "electronics",
    "running shoes":       "sports",
    "yoga mat":            "sports",
    "gym gloves":          "sports",
    "winter jacket":       "clothing",
    "jeans":               "clothing",
    "dress":               "clothing",
    "coffee maker":        "home",
    "desk lamp":           "home",
    "bed sheets":          "home",
    "face moisturizer":    "beauty",
    "shampoo":             "beauty",
    "novel":               "books",
    "kids puzzle":         "toys",
}

BRANDS = ["BrandA","BrandB","BrandC","BrandD","BrandE",
          "BrandF","BrandG","BrandH","BrandI","BrandJ"]

PRICE_BUCKETS = {
    "low":  (5,    50),
    "mid":  (50,   200),
    "high": (200,  1500),
}

POSITION_BIAS = {
    1:1.00, 2:0.85, 3:0.70, 4:0.58, 5:0.47,
    6:0.38, 7:0.30, 8:0.23, 9:0.17, 10:0.12
}


def generate_users(n, id_offset=0):
    """
    Generate n users. id_offset ensures new users get unique IDs
    that don't clash with users from previous runs.
    e.g. Run 1: u_00000–u_04999, Run 2: u_05000–u_09999
    """
    users = []
    for i in range(n):
        primary_cat   = random.choice(CATEGORIES)
        secondary_cat = random.choice([c for c in CATEGORIES if c != primary_cat])
        users.append({
            "user_id":            f"u_{(i + id_offset):05d}",
            "primary_category":   primary_cat,
            "secondary_category": secondary_cat,
            "price_bucket":       np.random.choice(["low","mid","high"], p=[0.40,0.45,0.15]),
            "preferred_brand":    random.choice(BRANDS),
            "brand_loyalty":      round(np.random.beta(2, 5), 3),
            "activity_level":     np.random.choice(["low","mid","high"], p=[0.5,0.35,0.15]),
            "signup_date":        (datetime(2024,1,1) + timedelta(days=random.randint(0,300))).date(),
        })
    return pd.DataFrame(users)


def generate_items(n, id_offset=0):
    """
    Generate n items. id_offset ensures new items get unique IDs
    that don't clash with items from previous runs.
    e.g. Run 1: i_00000–i_01999, Run 2: i_02000–i_03999
    """
    items = []
    for i in range(n):
        category     = random.choice(CATEGORIES)
        price_bucket = np.random.choice(["low","mid","high"], p=[0.45,0.40,0.15])
        price        = round(random.uniform(*PRICE_BUCKETS[price_bucket]), 2)
        items.append({
            "item_id":       f"i_{(i + id_offset):05d}",
            "category":      category,
            "brand":         random.choice(BRANDS),
            "price":         price,
            "price_bucket":  price_bucket,
            "quality_score": round(np.random.beta(2, 3), 4),
            "date_listed":   (datetime(2023,6,1) + timedelta(days=random.randint(0,500))).date(),
            "title":         f"{category.capitalize()} Product {(i + id_offset):05d}",
        })
    return pd.DataFrame(items)


def click_prob(user, item, query_category, position, base=0.08):
    p = base
    p += item["quality_score"] * 0.30
    if item["category"] == query_category:               p += 0.25
    elif item["category"] == user["primary_category"]:   p += 0.15
    elif item["category"] == user["secondary_category"]: p += 0.07
    if item["price_bucket"] == user["price_bucket"]:     p += 0.10
    if item["brand"] == user["preferred_brand"]:
        p += user["brand_loyalty"] * 0.15
    p *= POSITION_BIAS[position]
    return min(p, 0.95)


def purchase_prob(cp, user, item):
    p = cp * 0.12
    p += item["quality_score"] * 0.08
    if item["price_bucket"] == user["price_bucket"]: p += 0.05
    return min(p, 0.40)


def generate_events(users_df, items_df, n_sessions, session_id_offset=0):
    """
    Generate n_sessions of events. session_id_offset ensures session IDs
    are unique across runs so appended events never have duplicate session IDs.
    e.g. Run 1: s_0000000–s_0049999, Run 2: s_0050000–s_0099999
    """
    users        = users_df.to_dict("records")
    items        = items_df.to_dict("records")
    items_by_cat = {c:[it for it in items if it["category"]==c] for c in CATEGORIES}
    queries      = list(SEARCH_QUERIES.keys())
    events       = []
    t0           = datetime(2024, 6, 1)

    for s in range(n_sessions):
        user       = random.choice(users)
        is_search  = random.random() < SEARCH_RATIO
        session_id = f"s_{(s + session_id_offset):07d}"
        ts = t0 + timedelta(days=random.randint(0,180),
                            hours=random.randint(0,23),
                            minutes=random.randint(0,59))
        if is_search:
            query     = random.choice(queries)
            query_cat = SEARCH_QUERIES[query]
            pool      = items_by_cat.get(query_cat, items)
            n_cat     = min(7, len(pool))
            shown     = random.sample(pool, n_cat) + random.sample(items, RESULTS_PER_PAGE-n_cat)
        else:
            query     = None
            query_cat = user["primary_category"]
            pool      = items_by_cat.get(query_cat, items)
            n_cat     = min(5, len(pool))
            shown     = random.sample(pool, n_cat) + random.sample(items, RESULTS_PER_PAGE-n_cat)

        random.shuffle(shown)

        for pos, item in enumerate(shown, 1):
            cp  = click_prob(user, item, query_cat, pos)
            clk = random.random() < cp
            crt = random.random() < (cp*0.25) if clk else False
            pur = random.random() < purchase_prob(cp, user, item) if crt else False
            label = 3 if pur else 2 if crt else 1 if clk else 0

            events.append({
                "event_id":        f"e_{(s + session_id_offset):07d}_{pos:02d}",
                "session_id":      session_id,
                "user_id":         user["user_id"],
                "item_id":         item["item_id"],
                "session_type":    "search" if is_search else "homepage",
                "query_string":    query,
                "query_category":  query_cat,
                "position":        pos,
                "was_clicked":     clk,
                "was_carted":      crt,
                "was_purchased":   pur,
                "relevance_label": label,
                "timestamp":       ts + timedelta(seconds=pos*3),
            })

        if (s+1) % 5_000 == 0:
            print(f"  {s+1:,} / {n_sessions:,} sessions done...")

    return pd.DataFrame(events)


def main():
    print("="*60)
    print("Generating synthetic ecommerce data")
    print(f"Run seed: {SEED}")
    print("="*60)

    # ── Load existing data to compute offsets ─────────────────────────────
    users_path   = os.path.join(OUTPUT_DIR, "users.parquet")
    items_path   = os.path.join(OUTPUT_DIR, "items.parquet")
    events_path  = os.path.join(OUTPUT_DIR, "events.parquet")

    existing_users  = pd.DataFrame()
    existing_items  = pd.DataFrame()
    existing_events = pd.DataFrame()

    is_first_run = not os.path.exists(events_path)

    if is_first_run:
        print("\n  First run — creating fresh data files.")
        user_id_offset    = 0
        item_id_offset    = 0
        session_id_offset = 0
    else:
        existing_users  = pd.read_parquet(users_path)
        existing_items  = pd.read_parquet(items_path)
        existing_events = pd.read_parquet(events_path)

        # Offsets ensure new IDs never clash with existing ones
        user_id_offset    = len(existing_users)
        item_id_offset    = len(existing_items)
        session_id_offset = existing_events["session_id"].nunique()

        print(f"\n  Existing data detected:")
        print(f"    Users    : {len(existing_users):,}  → new users start at u_{user_id_offset:05d}")
        print(f"    Items    : {len(existing_items):,}  → new items start at i_{item_id_offset:05d}")
        print(f"    Sessions : {session_id_offset:,}  → new sessions start at s_{session_id_offset:07d}")

    # ── Generate new users ────────────────────────────────────────────────
    print(f"\n[1/3] Generating {N_USERS:,} new users...")
    users_df = generate_users(N_USERS, id_offset=user_id_offset)

    if not is_first_run:
        users_df = pd.concat([existing_users, users_df], ignore_index=True)
        users_df = users_df.drop_duplicates(subset=["user_id"])

    users_df.to_parquet(users_path, index=False)
    print(f"      Total users now : {len(users_df):,}")

    # ── Generate new items ────────────────────────────────────────────────
    print(f"\n[2/3] Generating {N_ITEMS:,} new items...")
    items_df = generate_items(N_ITEMS, id_offset=item_id_offset)

    if not is_first_run:
        items_df = pd.concat([existing_items, items_df], ignore_index=True)
        items_df = items_df.drop_duplicates(subset=["item_id"])

    items_df.to_parquet(items_path, index=False)
    print(f"      Total items now : {len(items_df):,}")

    # ── Generate new events and append ───────────────────────────────────
    # Events use ALL users and items (old + new) so new items can appear
    # in sessions alongside existing users — simulating real catalog growth
    print(f"\n[3/3] Generating {N_SESSIONS:,} new sessions...")
    new_events_df = generate_events(users_df, items_df, N_SESSIONS,
                                    session_id_offset=session_id_offset)

    if not is_first_run:
        events_df = pd.concat([existing_events, new_events_df], ignore_index=True)
        print(f"      New sessions    : {N_SESSIONS:,}")
        print(f"      Total sessions  : {events_df['session_id'].nunique():,}")
        print(f"      Total events    : {len(events_df):,}")
    else:
        events_df = new_events_df
        print(f"      Total sessions  : {events_df['session_id'].nunique():,}")
        print(f"      Total events    : {len(events_df):,}")

    events_df.to_parquet(events_path, index=False)

    # ── Summary stats (on new events only — not full history) ─────────────
    print("\n" + "="*60)
    print("THIS RUN SUMMARY (new events only)")
    print("="*60)
    n      = len(new_events_df)
    clicks = new_events_df["was_clicked"].sum()
    carts  = new_events_df["was_carted"].sum()
    purch  = new_events_df["was_purchased"].sum()
    print(f"\nNew impressions : {n:,}")
    print(f"Clicks          : {clicks:,}  CTR={clicks/n:.2%}")
    print(f"Cart adds       : {carts:,}   Cart/Click={carts/clicks:.2%}")
    print(f"Purchases       : {purch:,}   CVR={purch/clicks:.2%}")

    print("\nLabel distribution (new events):")
    for lbl, cnt in new_events_df["relevance_label"].value_counts().sort_index().items():
        print(f"  Label {lbl}: {cnt:>8,}  {'█'*int(cnt/n*150)}")

    print("\nCTR by rank position — new events (search only):")
    se = new_events_df[new_events_df["session_type"]=="search"]
    for pos, ctr in (se.groupby("position")["was_clicked"].mean()*100).round(2).items():
        print(f"  Rank {pos:2d}: {ctr:5.2f}%  {'█'*int(ctr*3)}")

    print("\n" + "="*60)
    print("CUMULATIVE DATA STATE")
    print("="*60)
    print(f"  Total users    : {len(users_df):,}")
    print(f"  Total items    : {len(items_df):,}")
    print(f"  Total sessions : {events_df['session_id'].nunique():,}")
    print(f"  Total events   : {len(events_df):,}")
    print("\n✅ Done! Next → scripts/02_feature_engineering.py")


if __name__ == "__main__":
    main()