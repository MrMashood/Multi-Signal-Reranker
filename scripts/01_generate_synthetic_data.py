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

SEED = 42
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

# Position bias: rank 1 gets 8x more attention than rank 10
# This is the KEY distortion LambdaRank must learn to correct
POSITION_BIAS = {
    1:1.00, 2:0.85, 3:0.70, 4:0.58, 5:0.47,
    6:0.38, 7:0.30, 8:0.23, 9:0.17, 10:0.12
}


def generate_users(n):
    users = []
    for i in range(n):
        primary_cat    = random.choice(CATEGORIES)
        secondary_cat  = random.choice([c for c in CATEGORIES if c != primary_cat])
        users.append({
            "user_id":            f"u_{i:05d}",
            "primary_category":   primary_cat,
            "secondary_category": secondary_cat,
            "price_bucket":       np.random.choice(["low","mid","high"], p=[0.40,0.45,0.15]),
            "preferred_brand":    random.choice(BRANDS),
            "brand_loyalty":      round(np.random.beta(2, 5), 3),
            "activity_level":     np.random.choice(["low","mid","high"], p=[0.5,0.35,0.15]),
            "signup_date":        (datetime(2024,1,1) + timedelta(days=random.randint(0,300))).date(),
        })
    return pd.DataFrame(users)


def generate_items(n):
    items = []
    for i in range(n):
        category     = random.choice(CATEGORIES)
        price_bucket = np.random.choice(["low","mid","high"], p=[0.45,0.40,0.15])
        price        = round(random.uniform(*PRICE_BUCKETS[price_bucket]), 2)
        items.append({
            "item_id":       f"i_{i:05d}",
            "category":      category,
            "brand":         random.choice(BRANDS),
            "price":         price,
            "price_bucket":  price_bucket,
            "quality_score": round(np.random.beta(2, 3), 4),  # ground truth only
            "date_listed":   (datetime(2023,6,1) + timedelta(days=random.randint(0,500))).date(),
            "title":         f"{category.capitalize()} Product {i:05d}",
        })
    return pd.DataFrame(items)


def click_prob(user, item, query_category, position, base=0.08):
    p = base
    p += item["quality_score"] * 0.30
    if item["category"] == query_category:       p += 0.25
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


def generate_events(users_df, items_df, n_sessions):
    users  = users_df.to_dict("records")
    items  = items_df.to_dict("records")
    items_by_cat = {c:[it for it in items if it["category"]==c] for c in CATEGORIES}
    queries      = list(SEARCH_QUERIES.keys())
    events       = []
    t0           = datetime(2024, 6, 1)

    for s in range(n_sessions):
        user       = random.choice(users)
        is_search  = random.random() < SEARCH_RATIO
        session_id = f"s_{s:07d}"
        ts = t0 + timedelta(days=random.randint(0,180),
                            hours=random.randint(0,23),
                            minutes=random.randint(0,59))
        if is_search:
            query      = random.choice(queries)
            query_cat  = SEARCH_QUERIES[query]
            pool       = items_by_cat.get(query_cat, items)
            n_cat      = min(7, len(pool))
            shown      = random.sample(pool, n_cat) + random.sample(items, RESULTS_PER_PAGE-n_cat)
        else:
            query      = None
            query_cat  = user["primary_category"]
            pool       = items_by_cat.get(query_cat, items)
            n_cat      = min(5, len(pool))
            shown      = random.sample(pool, n_cat) + random.sample(items, RESULTS_PER_PAGE-n_cat)

        random.shuffle(shown)  # random positions — position != quality

        for pos, item in enumerate(shown, 1):
            cp  = click_prob(user, item, query_cat, pos)
            clk = random.random() < cp
            crt = random.random() < (cp*0.25) if clk else False
            pur = random.random() < purchase_prob(cp, user, item) if crt else False
            label = 3 if pur else 2 if crt else 1 if clk else 0

            events.append({
                "event_id":        f"e_{s:07d}_{pos:02d}",
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
    print("="*60)

    print(f"\n[1/3] {N_USERS:,} users...")
    users_df = generate_users(N_USERS)
    users_df.to_parquet(os.path.join(OUTPUT_DIR, "users.parquet"), index=False)
    print(f"      Shape: {users_df.shape}")

    print(f"\n[2/3] {N_ITEMS:,} items...")
    items_df = generate_items(N_ITEMS)
    items_df.to_parquet(os.path.join(OUTPUT_DIR, "items.parquet"), index=False)
    print(f"      Shape: {items_df.shape}")

    print(f"\n[3/3] {N_SESSIONS:,} sessions...")
    events_df = generate_events(users_df, items_df, N_SESSIONS)
    events_df.to_parquet(os.path.join(OUTPUT_DIR, "events.parquet"), index=False)
    print(f"      Shape: {events_df.shape}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    n      = len(events_df)
    clicks = events_df["was_clicked"].sum()
    carts  = events_df["was_carted"].sum()
    purch  = events_df["was_purchased"].sum()
    print(f"\nImpressions : {n:,}")
    print(f"Clicks      : {clicks:,}  CTR={clicks/n:.2%}")
    print(f"Cart adds   : {carts:,}   Cart/Click={carts/clicks:.2%}")
    print(f"Purchases   : {purch:,}   CVR={purch/clicks:.2%}")

    print("\nLabel distribution:")
    for lbl, cnt in events_df["relevance_label"].value_counts().sort_index().items():
        print(f"  Label {lbl}: {cnt:>8,}  {'█'*int(cnt/n*150)}")

    print("\nCTR by rank position (search only):")
    se = events_df[events_df["session_type"]=="search"]
    for pos, ctr in (se.groupby("position")["was_clicked"].mean()*100).round(2).items():
        print(f"  Rank {pos:2d}: {ctr:5.2f}%  {'█'*int(ctr*3)}")


if __name__ == "__main__":
    main()