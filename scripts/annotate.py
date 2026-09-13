"""Script to inspect and convert pair annotations."""

import sys
import json
from leredd.utils import load_pairs_json, save_pairs_json

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/annotate.py <input_json> [output_json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "data/annotated/pairs.json"

    pairs = load_pairs_json(input_file)
    print(f"Loaded {len(pairs)} pairs from '{input_file}'.")
    save_pairs_json(pairs, output_file)
    print(f"Saved to '{output_file}'.")

if __name__ == "__main__":
    main()
