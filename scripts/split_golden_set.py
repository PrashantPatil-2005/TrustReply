"""Split golden set into train (180) and holdout test (40)."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ROOT, load_config


def main():
    cfg = load_config()
    src = ROOT / cfg["evaluation"]["golden_set_path"]
    examples = [json.loads(l) for l in open(src, encoding="utf-8")]

    rng = random.Random(cfg["evaluation"]["random_seed"])
    by_intent = defaultdict(list)
    for ex in examples:
        by_intent[ex["intent"]].append(ex)

    test, train = [], []
    for intent, pool in by_intent.items():
        rng.shuffle(pool)
        test.extend(pool[:4])   # 4 per intent = 40
        train.extend(pool[4:])  # rest to train

    train_path = ROOT / "data" / "golden_train.jsonl"
    test_path = ROOT / "data" / "golden_test.jsonl"

    for path, data in [(train_path, train), (test_path, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Train: {len(train)} -> {train_path}")
    print(f"Test:  {len(test)} -> {test_path}")


if __name__ == "__main__":
    main()
