"""
Bootstrap container

Guarantees that MLflow has an alias `prod` for model `MovieLensXGB`
whose artifact is readable from inside Docker. If the alias is missing
(or its artifact cannot be loaded) a minimal baseline is trained.

Steps
-----
1. Check if alias 'prod' already works → exit quickly.
2. Otherwise:
   • download MovieLens-100K
   • preprocess ratings
   • train a tiny XGBoost model (4 features, 10 trees)
   • log & register as new version
   • set alias 'prod' to that version
   • touch /app/mlruns/.alias_ready so Docker can consider bootstrap done
"""

from pathlib import Path
import os
import urllib.request, zipfile
import mlflow
from mlflow import MlflowClient
from xgboost import XGBClassifier
import pandas as pd

# ------------------------------------------------------------------ #
# configuration
# ------------------------------------------------------------------ #
MODEL_NAME  = os.getenv("MODEL_NAME",  "MovieLensXGB")
ALIAS_NAME  = os.getenv("ALIAS_NAME",  "prod")
RAW_DIR     = Path("data/raw")
PROC_DIR    = Path("data/processed")
ML_100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
READY_FLAG  = Path("/app/mlruns/.alias_ready")
EXPERIMENT  = "bootstrap"

client = MlflowClient()

# ------------------------------------------------------------------ #
# helper
# ------------------------------------------------------------------ #
def alias_points_to_valid_artifact() -> bool:
    """Return True if alias exists and its artifact can be loaded."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS_NAME)
        mlflow.pyfunc.load_model(mv.source)      # raises if unreadable
        print(f"[bootstrap] alias '{ALIAS_NAME}' already usable (v{mv.version})")
        return True
    except Exception:
        return False

# fast exit
if alias_points_to_valid_artifact():
    READY_FLAG.touch()
    raise SystemExit(0)

print("[bootstrap] alias missing – creating baseline model")

# ------------------------------------------------------------------ #
# download MovieLens-100K (once)
# ------------------------------------------------------------------ #
RAW_DIR.mkdir(parents=True, exist_ok=True)
if not (RAW_DIR / "ml-100k/u.data").exists():
    zip_path = RAW_DIR / "ml100k.zip"
    print("[bootstrap] downloading MovieLens-100K …")
    urllib.request.urlretrieve(ML_100K_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

# ------------------------------------------------------------------ #
# preprocess
# ------------------------------------------------------------------ #
ratings = pd.read_csv(
    RAW_DIR / "ml-100k/u.data",
    sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
).drop(columns="timestamp")

PROC_DIR.mkdir(parents=True, exist_ok=True)
ratings.to_parquet(PROC_DIR / "ratings.parquet")

ratings.groupby("user_id")["rating"].mean() \
       .to_frame("mean").to_parquet(PROC_DIR / "user_means.parquet")
ratings.groupby("item_id")["rating"].mean() \
       .to_frame("mean").to_parquet(PROC_DIR / "item_means.parquet")

# ------------------------------------------------------------------ #
# tiny XGBoost baseline – four features (matches the API)
# ------------------------------------------------------------------ #
ratings["user_mean_rating"] = ratings.groupby("user_id")["rating"].transform("mean")
ratings["item_mean_rating"] = ratings.groupby("item_id")["rating"].transform("mean")

FEATURES = ["user_id", "item_id", "user_mean_rating", "item_mean_rating"]
X = ratings[FEATURES]
y = (ratings["rating"] >= 4).astype(int)

# ensure experiment exists
exp = client.get_experiment_by_name(EXPERIMENT)
exp_id = exp.experiment_id if exp else client.create_experiment(EXPERIMENT)
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="bootstrap-minimal"):
    model = XGBClassifier(
        n_estimators=10,
        max_depth=3,
        objective="binary:logistic",
        n_jobs=-1,
    )
    model.fit(X, y)

    # log and register
    mlflow.xgboost.log_model(
        model,
        artifact_path="xgb_model",
        registered_model_name=MODEL_NAME,
    )

    new_version = client.get_latest_versions(MODEL_NAME)[0]
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=new_version.version,
    )
    print(f"[bootstrap] alias '{ALIAS_NAME}' → v{new_version.version}")

# mark ready for Docker health-check (if used)
READY_FLAG.touch()
print("[bootstrap] done – readiness flag created")
