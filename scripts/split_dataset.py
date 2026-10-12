"""Script to generate leakage-resistant dataset splits across ADB, APA, and TJA systems."""

import os
import glob
from leredd.utils import load_pairs_json
from leredd.dataset_splitter import DatasetSplitter, SplitConfig, SplitStrategy
from loguru import logger


def main():
    annotated_dir = "data/annotated"
    output_base_dir = "data/splits"

    pattern = os.path.join(annotated_dir, "*_pairs.json")
    json_files = glob.glob(pattern)

    if not json_files:
        logger.error(f"No annotated JSON files found in {annotated_dir}")
        return

    all_pairs = []
    for fpath in sorted(json_files):
        pairs = load_pairs_json(fpath)
        logger.info(f"Loaded {len(pairs)} pairs from {fpath}")
        all_pairs.extend(pairs)

    logger.info(f"Total pairs collected across all systems: {len(all_pairs)}")

    # 1. System-level split (ADB -> Train, APA -> Dev, TJA -> Test)
    sys_config = SplitConfig(
        strategy=SplitStrategy.SYSTEM,
        explicit_system_splits={
            "train": ["ADB"],
            "dev": ["APA"],
            "test": ["TJA"],
        },
        random_seed=42
    )
    sys_split = DatasetSplitter.split(all_pairs, config=sys_config)
    sys_out = os.path.join(output_base_dir, "system_disjoint")
    DatasetSplitter.save_split(sys_split, sys_out)
    sys_leakage = DatasetSplitter.check_leakage(sys_split)
    logger.info(f"System-level split completed. Leakage detected: {sys_leakage.has_leakage}")

    # 2. Requirement-level split (Disjoint requirements across Train/Dev/Test)
    req_config = SplitConfig(
        strategy=SplitStrategy.REQUIREMENT,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    req_split = DatasetSplitter.split(all_pairs, config=req_config)
    req_out = os.path.join(output_base_dir, "requirement_disjoint")
    DatasetSplitter.save_split(req_split, req_out)
    req_leakage = DatasetSplitter.check_leakage(req_split)
    logger.info(f"Requirement-level split completed. Leakage detected: {req_leakage.has_leakage}")

    print("\n=== DATASET SPLITTING COMPLETED ===")
    print(f"1. System Disjoint Split saved to: {sys_out}")
    print(f"   Train pairs: {len(sys_split.train_pairs)}, Dev pairs: {len(sys_split.dev_pairs)}, Test pairs: {len(sys_split.test_pairs)}")
    print(f"   Requirement leakage: {sys_leakage.has_leakage} (rate: {sys_leakage.leakage_rate:.2%})")
    print(f"2. Requirement Disjoint Split saved to: {req_out}")
    print(f"   Train pairs: {len(req_split.train_pairs)}, Dev pairs: {len(req_split.dev_pairs)}, Test pairs: {len(req_split.test_pairs)}")
    print(f"   Requirement leakage: {req_leakage.has_leakage} (rate: {req_leakage.leakage_rate:.2%})")


if __name__ == "__main__":
    main()
