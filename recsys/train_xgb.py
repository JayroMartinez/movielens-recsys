"""
Train a simple XGBoost model on MovieLens-100K and log everything to MLflow.
Target: rating >= 4  (implicit positive feedback)
Features:
    - user_id (as int)
    - item_id (as int)
    - user_mean_rating
    - item_mean_rating
"""

from __future__ import annotations

import pathlib
import pickle
from datetime import datetime

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from pathlib import Path

DATA_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "ratings.parquet"
)
MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True, parents=True)

mlflow.set_tracking_uri(f"sqlite:///{Path(__file__).resolve().parents[1] / 'mlruns.db'}")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple statistical features."""
    user_means = df.groupby("user_id")["rating"].mean()
    item_means = df.groupby("item_id")["rating"].mean()
    df = df.copy()
    df["user_mean_rating"] = df["user_id"].map(user_means)
    df["item_mean_rating"] = df["item_id"].map(item_means)
    return df


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run preprocess.py first.")

    df = pd.read_parquet(DATA_PATH)
    df = build_features(df)

    X = df[["user_id", "item_id", "user_mean_rating", "item_mean_rating"]]
    y = (df["rating"] >= 4).astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        eval_metric="auc",
    )

    with mlflow.start_run(run_name=f"xgb_{datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        mlflow.log_metric("val_auc", float(auc))

        # serialise model locally
        local_path = MODEL_DIR / "xgb_default.pkl"
        with open(local_path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(local_path, artifact_path="model")

        print(f"Validation AUC: {auc:.4f}")
        print(f"Model saved → {local_path}")

        # optional: register in MLflow Model Registry
        mlflow.xgboost.log_model(
            model,
            name="xgb_model",
            registered_model_name="MovieLensXGB",
        )


if __name__ == "__main__":
    mlflow.set_experiment("MovieLens-XGB")
    train()
