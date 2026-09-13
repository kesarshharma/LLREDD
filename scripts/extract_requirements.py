"""Script to extract requirements from SRS documents."""

import sys
import os
import json
from leredd.utils import extract_requirements_from_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_requirements.py <srs_path> [system_name] [output_path]")
        sys.exit(1)

    srs_path = sys.argv[1]
    system_name = sys.argv[2] if len(sys.argv) > 2 else "ADB"
    output_path = sys.argv[3] if len(sys.argv) > 3 else "data/processed/extracted_reqs.json"

    print(f"Extracting requirements from '{srs_path}'...")
    reqs = extract_requirements_from_file(srs_path, system_name=system_name)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = [r.model_dump(mode="json") for r in reqs]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully saved {len(reqs)} requirements to '{output_path}'")

if __name__ == "__main__":
    main()
