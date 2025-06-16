"""
MovieLens 100K preprocessing script.

Reads u.data (tab-separated), renames columns, casts dtypes,
and writes a clean Parquet file to data/processed/ratings.parquet.
"""

from __future__ import annotations
import pathlib
import pandas as pd

RAW_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "ml-100k"
PROC_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = RAW_DIR / "u.data"
OUTPUT_PATH = PROC_DIR / "ratings.parquet"


def preprocess() -> None:
    """Load raw ratings, clean them and store as Parquet."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run recsys.download_data first."
        )

    df = (
        pd.read_csv(
            INPUT_PATH,
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"],
            dtype={
                "user_id": "int32",
                "item_id": "int32",
                "rating": "int8",
                "timestamp": "int32",
            },
        )
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )

    # quick sanity check
    assert df["rating"].between(1, 5).all(), "Ratings outside 1-5 range!"
    print(
        f"Users: {df.user_id.nunique():>4} | "
        f"Items: {df.item_id.nunique():>4} | "
        f"Ratings: {len(df):>6}"
    )

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()
