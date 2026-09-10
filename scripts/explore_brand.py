"""Quick exploration script for brand data."""
import re
import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(
    r"C:\Users\Prash\.cache\kagglehub\datasets\thoughtvector\customer-support-on-twitter\versions\10\twcs\twcs.csv"
)
BRAND = sys.argv[1] if len(sys.argv) > 1 else "amazonhelp"


def main():
    rows = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=200000):
        mask = chunk["author_id"].str.lower() == BRAND
        rows.append(chunk[mask])
        if sum(len(r) for r in rows) > 30000:
            break

    df = pd.concat(rows)
    inbound = df[df["inbound"] == True]
    outbound = df[df["inbound"] == False]
    print(f"{BRAND} tweets: {len(df)}")
    print(f"Inbound: {len(inbound)}, Outbound: {len(outbound)}")
    print("\nSample inbound:")
    for t in inbound["text"].head(15):
        print(f"  - {str(t)[:140]}")
    print("\nSample outbound:")
    for t in outbound["text"].head(10):
        print(f"  - {str(t)[:140]}")


if __name__ == "__main__":
    main()
