"""Extract AmazonHelp customer→brand reply pairs from raw Kaggle data."""
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import ROOT, load_config


def get_data_path() -> Path:
    """Resolve Kaggle dataset path (kagglehub cache or local copy)."""
    candidates = [
        ROOT / "data" / "raw" / "twcs.csv",
        Path.home() / ".cache" / "kagglehub" / "datasets" / "thoughtvector" /
        "customer-support-on-twitter" / "versions" / "10" / "twcs" / "twcs.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Dataset not found. Run: python scripts/download_data.py"
    )


def extract_brand_pairs(
    brand: str = "amazonhelp",
    max_pairs: int = 50000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Build (customer_message, brand_reply) pairs for a brand."""
    data_path = get_data_path()
    brand_lower = brand.lower()

    print(f"Loading tweets for @{brand} from {data_path}...")
    brand_tweets: list[pd.DataFrame] = []
    customer_tweets: list[pd.DataFrame] = []

    for chunk in tqdm(
        pd.read_csv(data_path, chunksize=200_000, low_memory=False),
        desc="Scanning dataset",
    ):
        brand_mask = chunk["author_id"].str.lower() == brand_lower
        mention = f"@{brand}"
        customer_mask = (
            chunk["text"].str.contains(mention, case=False, na=False)
            & ~brand_mask
        )
        if brand_mask.any():
            brand_tweets.append(chunk[brand_mask])
        if customer_mask.any():
            customer_tweets.append(chunk[customer_mask])

    brand_df = pd.concat(brand_tweets, ignore_index=True)
    customer_df = pd.concat(customer_tweets, ignore_index=True)

    id_index: dict[int, pd.Series] = {}
    for _, row in customer_df.iterrows():
        tid = int(row["tweet_id"])
        if tid not in id_index:
            id_index[tid] = row

    pairs = []

    for _, brand_row in brand_df.iterrows():
        resp_to = brand_row.get("in_response_to_tweet_id")
        if pd.isna(resp_to):
            continue
        resp_to = int(resp_to)
        if resp_to not in id_index:
            continue
        cust_row = id_index[resp_to]
        cust_text = str(cust_row["text"]).strip()
        brand_text = str(brand_row["text"]).strip()

        # Skip very short or non-English-heavy threads for training clarity
        if len(cust_text) < 10 or len(brand_text) < 10:
            continue

        pairs.append({
            "tweet_id": resp_to,
            "customer_message": cust_text,
            "brand_reply": brand_text,
            "created_at": cust_row.get("created_at", ""),
            "customer_author": str(cust_row.get("author_id", "")),
        })
        if len(pairs) >= max_pairs:
            break

    df = pd.DataFrame(pairs)
    print(f"Extracted {len(df)} customer→brand pairs for @{brand}")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")

    return df


def build_retrieval_index(pairs_df: pd.DataFrame, output_path: Path) -> None:
    """Save pairs as JSONL for retrieval."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in pairs_df.iterrows():
            f.write(json.dumps({
                "customer_message": row["customer_message"],
                "brand_reply": row["brand_reply"],
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cfg = load_config()
    brand = cfg["brand"]
    out = ROOT / "data" / "processed" / f"{brand}_pairs.csv"
    df = extract_brand_pairs(brand=brand, max_pairs=cfg["data"]["max_train_samples"], output_path=out)
    build_retrieval_index(df, ROOT / "data" / "processed" / f"{brand}_retrieval.jsonl")
