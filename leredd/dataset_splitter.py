"""Leakage-resistant dataset splitter for Requirement Dependency Detection.

Provides deterministic, reproducible dataset splitting strategies:
1. SYSTEM: Disjoint by system name (e.g., ADB vs APA vs TJA).
2. DOCUMENT: Disjoint by SRS document/module metadata.
3. REQUIREMENT: Disjoint by unique requirement IDs (prevents requirement overlap across train/dev/test/retrieval).
4. PAIR: Standard pair-level random split (with explicit leakage warning).
"""

import json
import math
import os
import random
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from loguru import logger
from pydantic import BaseModel, Field

from leredd.data_models import RequirementPair


class SplitStrategy(str, Enum):
    """Supported dataset splitting strategies."""
    SYSTEM = "system"
    DOCUMENT = "document"
    REQUIREMENT = "requirement"
    PAIR = "pair"


class SplitConfig(BaseModel):
    """Configuration options for dataset splitting."""
    strategy: SplitStrategy = Field(
        default=SplitStrategy.REQUIREMENT,
        description="Dataset splitting strategy to enforce."
    )
    train_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset assigned to training set."
    )
    dev_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset assigned to development/validation set."
    )
    test_ratio: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of dataset assigned to test set."
    )
    retrieval_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Optional fraction assigned to example retrieval pool."
    )
    random_seed: int = Field(
        default=42,
        description="Random seed for deterministic partitioning."
    )
    explicit_system_splits: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Optional explicit mapping of system names to partitions, e.g. {'train': ['ADB'], 'dev': ['APA'], 'test': ['TJA']}."
    )
    drop_cross_partition_pairs: bool = Field(
        default=True,
        description="In REQUIREMENT split strategy, whether to drop pairs spanning across requirement partitions to prevent leakage."
    )

    def validate_ratios(self) -> None:
        """Validate that non-zero ratios sum approximately to 1.0."""
        ratios = (self.train_ratio, self.dev_ratio, self.test_ratio, self.retrieval_ratio)
        if any(not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0 for ratio in ratios):
            raise ValueError("Split ratios must each be finite values between 0.0 and 1.0.")
        total = self.train_ratio + self.dev_ratio + self.test_ratio + self.retrieval_ratio
        if not math.isclose(total, 1.0, abs_tol=1e-3):
            raise ValueError(f"Split ratios must sum to 1.0 (got {total:.4f}: train={self.train_ratio}, dev={self.dev_ratio}, test={self.test_ratio}, retrieval={self.retrieval_ratio})")


class LeakageReport(BaseModel):
    """Report detailing requirement and system overlap across dataset partitions."""
    has_leakage: bool = Field(description="True if any requirement or system leakage detected where unexpected.")
    req_overlap_train_test: Set[str] = Field(default_factory=set, description="Requirement IDs present in both train and test partitions.")
    req_overlap_train_dev: Set[str] = Field(default_factory=set, description="Requirement IDs present in both train and dev partitions.")
    req_overlap_dev_test: Set[str] = Field(default_factory=set, description="Requirement IDs present in both dev and test partitions.")
    req_overlap_retrieval_test: Set[str] = Field(default_factory=set, description="Requirement IDs present in both retrieval pool and test partition.")
    req_overlap_retrieval_train: Set[str] = Field(default_factory=set, description="Requirement IDs present in both retrieval pool and train partition.")
    req_overlap_retrieval_dev: Set[str] = Field(default_factory=set, description="Requirement IDs present in both retrieval pool and dev partition.")
    system_overlap_train_test: Set[str] = Field(default_factory=set, description="System names present in both train and test partitions.")
    system_overlap_train_dev: Set[str] = Field(default_factory=set, description="System names present in both train and dev partitions.")
    system_overlap_dev_test: Set[str] = Field(default_factory=set, description="System names present in both dev and test partitions.")
    system_overlap_retrieval_train: Set[str] = Field(default_factory=set, description="System names present in both retrieval and train partitions.")
    system_overlap_retrieval_dev: Set[str] = Field(default_factory=set, description="System names present in both retrieval and dev partitions.")
    system_overlap_retrieval_test: Set[str] = Field(default_factory=set, description="System names present in both retrieval and test partitions.")
    total_unique_requirements: int = Field(default=0, description="Total unique requirements across all partitions.")
    total_pairs: int = Field(default=0, description="Total pairs across all partitions.")
    leakage_rate: float = Field(default=0.0, description="Fraction of test requirements found in training set.")


class DatasetSplit(BaseModel):
    """Container holding partitioned requirement pairs and metadata."""
    train_pairs: List[RequirementPair] = Field(default_factory=list)
    dev_pairs: List[RequirementPair] = Field(default_factory=list)
    test_pairs: List[RequirementPair] = Field(default_factory=list)
    retrieval_pairs: List[RequirementPair] = Field(default_factory=list)
    cross_partition_pairs: List[RequirementPair] = Field(
        default_factory=list,
        description="Pairs that cross requirement partitions (kept if drop_cross_partition_pairs=False)."
    )
    config: SplitConfig = Field(default_factory=SplitConfig)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of the dataset split."""
        def req_ids(pairs: List[RequirementPair]) -> Set[str]:
            res = set()
            for p in pairs:
                res.add(p.req_a.id)
                res.add(p.req_b.id)
            return res

        train_reqs = req_ids(self.train_pairs)
        dev_reqs = req_ids(self.dev_pairs)
        test_reqs = req_ids(self.test_pairs)
        retrieval_reqs = req_ids(self.retrieval_pairs)

        return {
            "strategy": self.config.strategy.value,
            "seed": self.config.random_seed,
            "counts": {
                "train": {"pairs": len(self.train_pairs), "requirements": len(train_reqs)},
                "dev": {"pairs": len(self.dev_pairs), "requirements": len(dev_reqs)},
                "test": {"pairs": len(self.test_pairs), "requirements": len(test_reqs)},
                "retrieval": {"pairs": len(self.retrieval_pairs), "requirements": len(retrieval_reqs)},
                "cross_partition": {"pairs": len(self.cross_partition_pairs)},
            },
            "total_pairs": len(self.train_pairs) + len(self.dev_pairs) + len(self.test_pairs) + len(self.retrieval_pairs) + len(self.cross_partition_pairs),
        }


class DatasetSplitter:
    """Splits requirement pair datasets while preventing requirement and system data leakage."""

    @staticmethod
    def split(
        pairs: List[RequirementPair],
        config: Optional[SplitConfig] = None
    ) -> DatasetSplit:
        """Split a list of RequirementPair objects according to the specified configuration strategy."""
        if config is None:
            config = SplitConfig()

        config.validate_ratios()

        if not pairs:
            logger.warning("Empty pairs list provided to DatasetSplitter.")
            return DatasetSplit(config=config, metadata={"note": "empty_input"})

        rng = random.Random(config.random_seed)

        if config.strategy == SplitStrategy.SYSTEM:
            return DatasetSplitter._split_by_system(pairs, config, rng)
        elif config.strategy == SplitStrategy.DOCUMENT:
            return DatasetSplitter._split_by_document(pairs, config, rng)
        elif config.strategy == SplitStrategy.REQUIREMENT:
            return DatasetSplitter._split_by_requirement(pairs, config, rng)
        elif config.strategy == SplitStrategy.PAIR:
            return DatasetSplitter._split_by_pair(pairs, config, rng)
        else:
            raise ValueError(f"Unsupported split strategy: {config.strategy}")

    @staticmethod
    def _split_by_system(
        pairs: List[RequirementPair],
        config: SplitConfig,
        rng: random.Random
    ) -> DatasetSplit:
        """Partition dataset by system identifier (e.g. ADB, APA, TJA)."""
        # Group pairs by system (using req_a.system or req_b.system)
        system_pairs: Dict[str, List[RequirementPair]] = {}
        for p in pairs:
            sys_a = p.req_a.system or "Generic"
            sys_b = p.req_b.system or "Generic"
            if sys_a != sys_b:
                raise ValueError(
                    f"SYSTEM split requires each pair to belong to one system; "
                    f"pair {p.id} spans {sys_a!r} and {sys_b!r}."
                )
            sys_name = sys_a
            system_pairs.setdefault(sys_name, []).append(p)

        systems = sorted(list(system_pairs.keys()))

        train_pairs: List[RequirementPair] = []
        dev_pairs: List[RequirementPair] = []
        test_pairs: List[RequirementPair] = []
        retrieval_pairs: List[RequirementPair] = []

        system_assignments: Dict[str, str] = {}

        if config.explicit_system_splits:
            for sys_name in systems:
                assigned = False
                for partition, sys_list in config.explicit_system_splits.items():
                    if sys_name in sys_list:
                        system_assignments[sys_name] = partition
                        assigned = True
                        break
                if not assigned:
                    system_assignments[sys_name] = "train"
        else:
            rng.shuffle(systems)
            n_sys = len(systems)
            if n_sys == 1:
                logger.warning(f"Only 1 system ('{systems[0]}') present. Cannot perform disjoint system-level split. Assigning all to train.")
                system_assignments[systems[0]] = "train"
            else:
                # Reserve one system for each requested partition that can fit,
                # then apportion the remainder by ratio. Rounded ratios otherwise
                # turn the default 60/20/20 split of three systems into 2/0/1.
                ratios = {
                    "train": config.train_ratio,
                    "dev": config.dev_ratio,
                    "test": config.test_ratio,
                    "retrieval": config.retrieval_ratio,
                }
                active = [
                    name for name, ratio in ratios.items()
                    if ratio > 0 and (name != "retrieval" or n_sys > 3)
                ]
                if not active:
                    active = ["train"]
                    ratios["train"] = 1.0
                while len(active) > n_sys:
                    # Keep train and test as the core supervised evaluation
                    # partitions; dev and retrieval are optional at this scale.
                    for optional in ("retrieval", "dev"):
                        if optional in active:
                            active.remove(optional)
                            break

                remaining = n_sys - len(active)
                total_weight = sum(ratios[name] for name in active)
                additions = {
                    name: int(remaining * ratios[name] / total_weight)
                    for name in active
                }
                seats_left = remaining - sum(additions.values())
                tie_order = {name: index for index, name in enumerate(active)}
                by_remainder = sorted(
                    active,
                    key=lambda name: (
                        -(remaining * ratios[name] / total_weight - additions[name]),
                        tie_order[name],
                    ),
                )
                for name in by_remainder[:seats_left]:
                    additions[name] += 1
                counts = {name: 1 + additions[name] for name in active}

                offset = 0
                for name in active:
                    for system in systems[offset:offset + counts[name]]:
                        system_assignments[system] = name
                    offset += counts[name]
                # Preserve the historical fallback for a zero-ratio or
                # otherwise unassigned system.
                for system in systems[offset:]:
                    system_assignments[system] = "train"

        for sys_name, sys_pair_list in system_pairs.items():
            target = system_assignments.get(sys_name, "train")
            if target == "train":
                train_pairs.extend(sys_pair_list)
            elif target == "dev":
                dev_pairs.extend(sys_pair_list)
            elif target == "test":
                test_pairs.extend(sys_pair_list)
            elif target == "retrieval":
                retrieval_pairs.extend(sys_pair_list)

        return DatasetSplit(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            test_pairs=test_pairs,
            retrieval_pairs=retrieval_pairs,
            config=config,
            metadata={"system_assignments": system_assignments}
        )

    @staticmethod
    def _split_by_document(
        pairs: List[RequirementPair],
        config: SplitConfig,
        rng: random.Random
    ) -> DatasetSplit:
        """Partition dataset by document or module metadata on requirements."""
        doc_pairs: Dict[str, List[RequirementPair]] = {}
        for p in pairs:
            # Check metadata or source document
            meta_a = getattr(p.req_a, "metadata", None) or {}
            meta_b = getattr(p.req_b, "metadata", None) or {}
            doc_a = meta_a.get("document") or meta_a.get("module") or p.req_a.source or p.req_a.system
            doc_b = meta_b.get("document") or meta_b.get("module") or p.req_b.source or p.req_b.system
            doc_name = str(doc_a if doc_a == doc_b else f"{doc_a}_{doc_b}")
            doc_pairs.setdefault(doc_name, []).append(p)

        docs = sorted(list(doc_pairs.keys()))
        rng.shuffle(docs)

        n_docs = len(docs)
        n_train = max(1, round(n_docs * config.train_ratio)) if config.train_ratio > 0 else 0
        n_retrieval = max(1, round(n_docs * config.retrieval_ratio)) if config.retrieval_ratio > 0 and n_docs > 3 else 0
        n_dev = max(1, round(n_docs * config.dev_ratio)) if config.dev_ratio > 0 and (n_docs - n_train - n_retrieval) >= 2 else 0

        train_docs = set(docs[:n_train])
        retrieval_docs = set(docs[n_train:n_train + n_retrieval])
        dev_docs = set(docs[n_train + n_retrieval:n_train + n_retrieval + n_dev])
        test_docs = set(docs[n_train + n_retrieval + n_dev:])

        if not test_docs and config.test_ratio > 0 and len(train_docs) > 1:
            moved = docs[n_train - 1]
            train_docs.remove(moved)
            test_docs.add(moved)

        train_pairs, dev_pairs, test_pairs, retrieval_pairs = [], [], [], []
        doc_assignments = {}

        for doc in docs:
            if doc in train_docs:
                doc_assignments[doc] = "train"
                train_pairs.extend(doc_pairs[doc])
            elif doc in retrieval_docs:
                doc_assignments[doc] = "retrieval"
                retrieval_pairs.extend(doc_pairs[doc])
            elif doc in dev_docs:
                doc_assignments[doc] = "dev"
                dev_pairs.extend(doc_pairs[doc])
            else:
                doc_assignments[doc] = "test"
                test_pairs.extend(doc_pairs[doc])

        return DatasetSplit(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            test_pairs=test_pairs,
            retrieval_pairs=retrieval_pairs,
            config=config,
            metadata={"document_assignments": doc_assignments}
        )

    @staticmethod
    def _split_by_requirement(
        pairs: List[RequirementPair],
        config: SplitConfig,
        rng: random.Random
    ) -> DatasetSplit:
        """Partition unique requirement IDs into disjoint sets to prevent requirement leakage across partitions."""
        # Collect all unique requirement IDs
        req_map = {}
        for p in pairs:
            req_map[p.req_a.id] = p.req_a
            req_map[p.req_b.id] = p.req_b

        req_ids = sorted(list(req_map.keys()))
        rng.shuffle(req_ids)

        n_reqs = len(req_ids)
        n_train = int(n_reqs * config.train_ratio)
        n_retrieval = int(n_reqs * config.retrieval_ratio)
        n_dev = int(n_reqs * config.dev_ratio)

        train_req_set = set(req_ids[:n_train])
        retrieval_req_set = set(req_ids[n_train:n_train + n_retrieval])
        dev_req_set = set(req_ids[n_train + n_retrieval:n_train + n_retrieval + n_dev])
        test_req_set = set(req_ids[n_train + n_retrieval + n_dev:])

        train_pairs: List[RequirementPair] = []
        dev_pairs: List[RequirementPair] = []
        test_pairs: List[RequirementPair] = []
        retrieval_pairs: List[RequirementPair] = []
        cross_pairs: List[RequirementPair] = []

        for p in pairs:
            id_a, id_b = p.req_a.id, p.req_b.id

            in_train = id_a in train_req_set and id_b in train_req_set
            in_dev = id_a in dev_req_set and id_b in dev_req_set
            in_test = id_a in test_req_set and id_b in test_req_set
            in_retrieval = id_a in retrieval_req_set and id_b in retrieval_req_set

            if in_train:
                train_pairs.append(p)
            elif in_dev:
                dev_pairs.append(p)
            elif in_test:
                test_pairs.append(p)
            elif in_retrieval:
                retrieval_pairs.append(p)
            else:
                cross_pairs.append(p)

        if not config.drop_cross_partition_pairs:
            # If keeping cross-partition pairs, route them to train set to keep test/dev strictly pure
            logger.info(f"Adding {len(cross_pairs)} cross-partition pairs to training set.")
            train_pairs.extend(cross_pairs)
            cross_pairs = []

        return DatasetSplit(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            test_pairs=test_pairs,
            retrieval_pairs=retrieval_pairs,
            cross_partition_pairs=cross_pairs,
            config=config,
            metadata={
                "total_unique_requirements": n_reqs,
                "partition_req_counts": {
                    "train": len(train_req_set),
                    "dev": len(dev_req_set),
                    "test": len(test_req_set),
                    "retrieval": len(retrieval_req_set),
                }
            }
        )

    @staticmethod
    def _split_by_pair(
        pairs: List[RequirementPair],
        config: SplitConfig,
        rng: random.Random
    ) -> DatasetSplit:
        """Standard pair-level random split (emits leakage warning)."""
        logger.warning(
            "WARNING: Using PAIR strategy (pair-level random split). "
            "This strategy DOES NOT prevent requirement leakage between train and test sets!"
        )
        shuffled = list(pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * config.train_ratio)
        n_retrieval = int(n * config.retrieval_ratio)
        n_dev = int(n * config.dev_ratio)

        train_pairs = shuffled[:n_train]
        retrieval_pairs = shuffled[n_train:n_train + n_retrieval]
        dev_pairs = shuffled[n_train + n_retrieval:n_train + n_retrieval + n_dev]
        test_pairs = shuffled[n_train + n_retrieval + n_dev:]

        return DatasetSplit(
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            test_pairs=test_pairs,
            retrieval_pairs=retrieval_pairs,
            config=config,
            metadata={"warning": "pair_level_leakage_risk"}
        )

    @staticmethod
    def check_leakage(split: DatasetSplit) -> LeakageReport:
        """Analyze a dataset split and report requirement and system level data leakage."""
        def get_req_ids(pairs: List[RequirementPair]) -> Set[str]:
            ids = set()
            for p in pairs:
                ids.add(p.req_a.id)
                ids.add(p.req_b.id)
            return ids

        def get_systems(pairs: List[RequirementPair]) -> Set[str]:
            sys_set = set()
            for p in pairs:
                if p.req_a.system:
                    sys_set.add(p.req_a.system)
                if p.req_b.system:
                    sys_set.add(p.req_b.system)
            return sys_set

        train_reqs = get_req_ids(split.train_pairs)
        dev_reqs = get_req_ids(split.dev_pairs)
        test_reqs = get_req_ids(split.test_pairs)
        retrieval_reqs = get_req_ids(split.retrieval_pairs)
        retrieval_train = retrieval_reqs.intersection(train_reqs)
        retrieval_dev = retrieval_reqs.intersection(dev_reqs)

        overlap_train_test = train_reqs.intersection(test_reqs)
        overlap_train_dev = train_reqs.intersection(dev_reqs)
        overlap_dev_test = dev_reqs.intersection(test_reqs)
        overlap_retrieval_test = retrieval_reqs.intersection(test_reqs)

        train_sys = get_systems(split.train_pairs)
        dev_sys = get_systems(split.dev_pairs)
        test_sys = get_systems(split.test_pairs)
        retrieval_sys = get_systems(split.retrieval_pairs)
        sys_overlap = train_sys.intersection(test_sys)
        sys_overlap_train_dev = train_sys.intersection(dev_sys)
        sys_overlap_dev_test = dev_sys.intersection(test_sys)
        sys_overlap_retrieval_train = retrieval_sys.intersection(train_sys)
        sys_overlap_retrieval_dev = retrieval_sys.intersection(dev_sys)
        sys_overlap_retrieval_test = retrieval_sys.intersection(test_sys)

        all_reqs = train_reqs | dev_reqs | test_reqs | retrieval_reqs
        total_pairs = (
            len(split.train_pairs)
            + len(split.dev_pairs)
            + len(split.test_pairs)
            + len(split.retrieval_pairs)
            + len(split.cross_partition_pairs)
        )

        has_leakage = False
        if split.config.strategy in (SplitStrategy.REQUIREMENT, SplitStrategy.SYSTEM, SplitStrategy.DOCUMENT):
            if (
                overlap_train_test
                or overlap_train_dev
                or overlap_dev_test
                or retrieval_train
                or retrieval_dev
                or overlap_retrieval_test
            ):
                has_leakage = True

        if split.config.strategy == SplitStrategy.SYSTEM and (
            sys_overlap
            or sys_overlap_train_dev
            or sys_overlap_dev_test
            or sys_overlap_retrieval_train
            or sys_overlap_retrieval_dev
            or sys_overlap_retrieval_test
        ):
            has_leakage = True

        leakage_rate = len(overlap_train_test) / len(test_reqs) if test_reqs else 0.0

        return LeakageReport(
            has_leakage=has_leakage,
            req_overlap_train_test=overlap_train_test,
            req_overlap_train_dev=overlap_train_dev,
            req_overlap_dev_test=overlap_dev_test,
            req_overlap_retrieval_test=overlap_retrieval_test,
            req_overlap_retrieval_train=retrieval_train,
            req_overlap_retrieval_dev=retrieval_dev,
            system_overlap_train_test=sys_overlap,
            system_overlap_train_dev=sys_overlap_train_dev,
            system_overlap_dev_test=sys_overlap_dev_test,
            system_overlap_retrieval_train=sys_overlap_retrieval_train,
            system_overlap_retrieval_dev=sys_overlap_retrieval_dev,
            system_overlap_retrieval_test=sys_overlap_retrieval_test,
            total_unique_requirements=len(all_reqs),
            total_pairs=total_pairs,
            leakage_rate=leakage_rate,
        )

    @staticmethod
    def save_split(split: DatasetSplit, output_dir: str) -> None:
        """Save dataset split partitions to JSON files in the specified directory."""
        os.makedirs(output_dir, exist_ok=True)

        def dump_file(pairs: List[RequirementPair], filename: str) -> None:
            path = os.path.join(output_dir, filename)
            data = [p.model_dump(mode="json") for p in pairs]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        dump_file(split.train_pairs, "train_pairs.json")
        dump_file(split.dev_pairs, "dev_pairs.json")
        dump_file(split.test_pairs, "test_pairs.json")
        if split.retrieval_pairs:
            dump_file(split.retrieval_pairs, "retrieval_pairs.json")
        if split.cross_partition_pairs:
            dump_file(split.cross_partition_pairs, "cross_partition_pairs.json")

        # Save metadata and leakage report
        leakage = DatasetSplitter.check_leakage(split)
        meta = {
            "summary": split.summary(),
            "leakage_report": leakage.model_dump(mode="json"),
        }
        meta_path = os.path.join(output_dir, "split_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Dataset split saved to '{output_dir}'. Summary: {split.summary()['counts']}")
