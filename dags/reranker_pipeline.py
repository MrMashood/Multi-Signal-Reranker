"""
Reranker Automation Pipeline DAG
=================================
Orchestrates the full reranker pipeline using DockerOperator.
Each task spins up a fresh python:3.11 container, mounts the
reranker project folder, and runs one script.

Schedule: Monthly (1st of every month at midnight)
Manual:   Can be triggered anytime from Airflow UI

Task flow:
  generate_data → feature_engineering → build_matrix → train_als → rerank_evaluate

Place this file in:
  C:/Users/Volmatica/Desktop/airflow-docker/dags/reranker_pipeline.py
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# ── DAG default arguments ─────────────────────────────────────────────────────
default_args = {
    "owner":            "reranker",
    "retries":          1,
    "retry_delay":      timedelta(minutes=2),
    "email_on_failure": False,
    "email_on_retry":   False,
}

# ── Shared config ─────────────────────────────────────────────────────────────
# Path to reranker project ON THE HOST MACHINE (not inside container)
HOST_RERANKER_PATH = "C:/Users/Volmatica/Desktop/reranker"

# Where it gets mounted inside each task container
CONTAINER_RERANKER_PATH = "/opt/reranker"

# Docker image to run scripts in
PYTHON_IMAGE = "python:3.11-slim"

# Dependencies to install before each script runs
PIP_INSTALL = (
    "pip install --quiet "
    "pandas numpy scipy pyarrow duckdb scikit-learn"
)

# Shared mount — reranker project folder → container
reranker_mount = Mount(
    target=CONTAINER_RERANKER_PATH,
    source=HOST_RERANKER_PATH,
    type="bind",
)


def make_task(dag: DAG, task_id: str, script: str) -> DockerOperator:
    """
    Factory function — creates a DockerOperator task that:
      1. Spins up a python:3.11-slim container
      2. Mounts the reranker project at /opt/reranker
      3. pip installs dependencies
      4. Runs the specified script
      5. Container exits and is auto-removed

    Args:
        dag     : the parent DAG
        task_id : unique task name shown in Airflow UI
        script  : path to script relative to /opt/reranker
    """
    return DockerOperator(
        task_id=task_id,
        image=PYTHON_IMAGE,
        command=f'bash -c "{PIP_INSTALL} && python {CONTAINER_RERANKER_PATH}/{script}"',
        mounts=[reranker_mount],
        mount_tmp_dir=False,
        auto_remove="force",        # clean up container after task completes
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        dag=dag,
    )


# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="reranker_pipeline",
    description="Monthly reranker retraining pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule="0 0 1 * *",          # 1st of every month at midnight
    catchup=False,                  # don't backfill missed runs
    max_active_runs=1,              # only one run at a time
    tags=["reranker", "ml", "recommendation"],
) as dag:

    # ── Task 1: Generate synthetic data ───────────────────────────────────
    # Simulates a new month of user traffic
    # Output: data/raw/users.parquet, items.parquet, events.parquet
    generate_data = make_task(
        dag=dag,
        task_id="generate_data",
        script="scripts/01_generate_synthetic_data.py",
    )

    # ── Task 2: Feature engineering ───────────────────────────────────────
    # Runs DuckDB SQL transformations on raw parquet files
    # Output: data/features/training_data.parquet
    feature_engineering = make_task(
        dag=dag,
        task_id="feature_engineering",
        script="scripts/02_feature_engineering.py",
    )

    # ── Task 3: Build interaction matrix ─────────────────────────────────
    # Converts training_data into sparse confidence-weighted matrix
    # Output: data/matrix/interaction_matrix.npz + index files
    build_matrix = make_task(
        dag=dag,
        task_id="build_matrix",
        script="scripts/03a_interaction_matrix.py",
    )

    # ── Task 4: Train ALS model ───────────────────────────────────────────
    # Runs ALS from scratch on the interaction matrix
    # Output: data/model/user_vectors.npy, item_vectors.npy
    train_als = make_task(
        dag=dag,
        task_id="train_als",
        script="scripts/03b_als_model.py",
    )

    # ── Task 5: Rerank + evaluate ─────────────────────────────────────────
    # Scores all sessions with ALS embeddings, computes NDCG@K
    # Output: data/eval/evaluation_summary.json
    rerank_evaluate = make_task(
        dag=dag,
        task_id="rerank_evaluate",
        script="scripts/03c_rerank.py",
    )

    # ── Task dependencies (linear pipeline) ──────────────────────────────
    generate_data >> feature_engineering >> build_matrix >> train_als >> rerank_evaluate