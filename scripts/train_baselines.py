"""Script to train baseline models."""

import sys
import os
from leredd.utils import load_pairs_json
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline

def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/annotated/adb_pairs.json"
    pairs = load_pairs_json(train_path)
    print(f"Loaded {len(pairs)} pairs for training baselines.")

    print("Training BERT baseline classifier...")
    bert_model = BERTBaseline()
    bert_model.fit(pairs)

    print("Training TF-IDF & LSA baseline...")
    tfidf_model = TFIDF_LSABaseline()
    tfidf_model.fit(pairs)

    print("Baseline training complete!")

if __name__ == "__main__":
    main()
