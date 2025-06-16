"""
FastAPI inference service
=========================

* Loads model `MovieLensXGB` via alias `prod`.
* Endpoints:
    • GET /health                – liveness check
    • GET /recommend/{user_id}   – top-k recommendations
"""

from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from fastapi import FastAPI, HTTPException

# --------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------- #
MODEL_URI = "models:/MovieLensXGB@prod"
K_DEFAULT = 10

# --------------------------------------------------------------------- #
# load artefacts
# --------------------------------------------------------------------- #
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as exc:
    raise RuntimeError(f"Cannot load {MODEL_URI}: {exc}") from exc

DATA_DIR   = Path("data/processed")
user_means = pd.read_parquet(DATA_DIR / "user_means.parquet")["mean"]
item_means = pd.read_parquet(DATA_DIR / "item_means.parquet")["mean"]
all_items  = item_means.index.to_numpy()

# --------------------------------------------------------------------- #
# FastAPI setup
# --------------------------------------------------------------------- #
app = FastAPI(title="MovieLens-XGB Recommender", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = K_DEFAULT) -> dict:
    if k <= 0:
        raise HTTPException(status_code=400, detail="k must be positive")
    if k > len(all_items):
        raise HTTPException(status_code=400, detail="k exceeds catalogue size")

    u_mean    = user_means.get(user_id, 3.5)
    num_items = len(all_items)

    feats = pd.DataFrame({
        "user_id":          np.full(num_items, user_id, dtype=np.int32),
        "item_id":          all_items,
        "user_mean_rating": np.full(num_items, u_mean,  dtype=np.float32),
        "item_mean_rating": item_means.reindex(all_items).fillna(3.5).to_numpy(),
    })

    # predict returns probability for class 1 with binary:logistic
    scores  = model.predict(feats)
    top_idx = scores.argsort()[-k:][::-1]

    return {
        "user_id": user_id,
        "items":   all_items[top_idx].tolist(),
        "scores":  scores[top_idx].round(4).tolist(),
    }
