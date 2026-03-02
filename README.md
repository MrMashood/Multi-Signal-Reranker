reranker/
├── data/
│   ├── raw/
│   │   ├── events.parquet
│   │   ├── items.parquet
│   │   └── users.parquet
│   └── features/
│       └── training_data.parquet
├── sql/
│   ├── 01_user_item_features.sql
│   ├── 02_query_item_features.sql
│   ├── 03_item_features.sql
│   └── 04_build_training_table.sql
├── scripts/
│   ├── 01_generate_synthetic_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_reranker.py
│   └── 04_evaluate.py
└── notebooks/
    └── exploration.ipynb