"""Explore conversation thread structure."""
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(
    r"C:\Users\Prash\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv"
)
BRAND = "amazonhelp"


def main():
    # Load tweets mentioning amazonhelp or from amazonhelp
    brand_tweets = []
    related = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200000, low_memory=False):
        brand_mask = chunk["author_id"].str.lower() == BRAND
        mention_mask = chunk["text"].str.contains("@AmazonHelp", case=False, na=False)
        brand_tweets.append(chunk[brand_mask])
        related.append(chunk[mention_mask & ~brand_mask])
        if sum(len(r) for r in brand_tweets) > 5000:
            break

    brand_df = pd.concat(brand_tweets)
    customer_df = pd.concat(related)
    print(f"Brand tweets: {len(brand_df)}")
    print(f"Customer tweets mentioning @AmazonHelp: {len(customer_df)}")
    print(f"Brand inbound field: {brand_df['inbound'].value_counts().to_dict()}")

    # Sample a thread: find brand reply with response
    sample = brand_df.dropna(subset=["in_response_to_tweet_id"]).head(5)
    all_ids = set(brand_df["tweet_id"]) | set(customer_df["tweet_id"])
    id_to_row = {}
    for _, row in pd.concat([brand_df, customer_df]).iterrows():
        id_to_row[row["tweet_id"]] = row

    for _, brand_row in sample.iterrows():
        cust_id = int(brand_row["in_response_to_tweet_id"])
        cust_row = id_to_row.get(cust_id)
        print("\n--- Thread ---")
        if cust_row is not None:
            print(f"CUSTOMER: {str(cust_row['text'])[:200]}")
        else:
            print(f"CUSTOMER (id={cust_id}): [not in sample]")
        print(f"BRAND: {str(brand_row['text'])[:200]}")


if __name__ == "__main__":
    main()
