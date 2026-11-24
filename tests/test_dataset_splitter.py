"""Unit tests for leredd.dataset_splitter (TASK-P2-02).

Verifies zero requirement leakage across partitions, system-level splitting,
document splitting, reproducibility, edge cases, persistence, and CLI integration.
"""

import json
import os
import tempfile
import pytest
from typer.testing import CliRunner

from leredd.data_models import Requirement, RequirementPair, DependencyType
from leredd.dataset_splitter import (
    DatasetSplitter,
    SplitConfig,
    SplitStrategy,
    DatasetSplit,
)
from leredd.cli import app


@pytest.fixture
def sample_pairs():
    """Generate synthetic requirement pairs for 4 systems (SYS1..SYS4), 5 reqs each."""
    pairs = []
    pair_id = 1
    for sys_idx in range(1, 5):
        sys_name = f"SYS{sys_idx}"
        reqs = [
            Requirement(
                id=f"REQ-{sys_name}-{r_idx:02d}",
                text=f"Requirement {r_idx} for system {sys_name} shall operate properly.",
                system=sys_name,
                metadata={"document": f"DOC-{sys_name}"}
            )
            for r_idx in range(1, 6)
        ]
        # Create all unique pairs within system
        for i in range(len(reqs)):
            for j in range(i + 1, len(reqs)):
                pairs.append(
                    RequirementPair(
                        id=f"PAIR-{sys_name}-{pair_id:04d}",
                        req_a=reqs[i],
                        req_b=reqs[j],
                        dependency_type=DependencyType.REQUIRES if (i + j) % 2 == 0 else DependencyType.NO_DEPENDENCY
                    )
                )
                pair_id += 1
    return pairs


def test_requirement_split_zero_leakage(sample_pairs):
    """Verify REQUIREMENT strategy produces zero requirement ID overlap between train and test partitions."""
    config = SplitConfig(
        strategy=SplitStrategy.REQUIREMENT,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    split = DatasetSplitter.split(sample_pairs, config)
    leakage = DatasetSplitter.check_leakage(split)

    assert not leakage.has_leakage, f"Leakage detected: {leakage}"
    assert len(leakage.req_overlap_train_test) == 0
    assert len(leakage.req_overlap_retrieval_test) == 0

    train_reqs = {p.req_a.id for p in split.train_pairs} | {p.req_b.id for p in split.train_pairs}
    test_reqs = {p.req_a.id for p in split.test_pairs} | {p.req_b.id for p in split.test_pairs}
    assert train_reqs.isdisjoint(test_reqs)


def test_system_split_zero_system_leakage(sample_pairs):
    """Verify SYSTEM strategy with explicit mapping produces zero system overlap."""
    config = SplitConfig(
        strategy=SplitStrategy.SYSTEM,
        explicit_system_splits={
            "train": ["SYS1", "SYS2"],
            "dev": ["SYS3"],
            "test": ["SYS4"],
        },
        random_seed=42
    )
    split = DatasetSplitter.split(sample_pairs, config)
    leakage = DatasetSplitter.check_leakage(split)

    assert not leakage.has_leakage
    assert len(leakage.system_overlap_train_test) == 0

    train_systems = {p.req_a.system for p in split.train_pairs}
    test_systems = {p.req_a.system for p in split.test_pairs}
    assert train_systems == {"SYS1", "SYS2"}
    assert test_systems == {"SYS4"}
    assert train_systems.isdisjoint(test_systems)


def test_default_system_split_reserves_train_dev_test_for_three_systems():
    """Default ratios must not round the development partition down to empty."""
    pairs = [
        RequirementPair(
            id=f"PAIR-{system}",
            req_a=Requirement(id=f"{system}-A", text=f"{system} requirement A", system=system),
            req_b=Requirement(id=f"{system}-B", text=f"{system} requirement B", system=system),
            dependency_type=DependencyType.REQUIRES,
        )
        for system in ("ADB", "APA", "TJA")
    ]

    split = DatasetSplitter.split(
        pairs,
        SplitConfig(strategy=SplitStrategy.SYSTEM, random_seed=42),
    )

    assert len(split.train_pairs) == 1
    assert len(split.dev_pairs) == 1
    assert len(split.test_pairs) == 1
    assert not DatasetSplitter.check_leakage(split).has_leakage
    assert set(split.metadata["system_assignments"].values()) == {
        "train", "dev", "test",
    }


def test_document_split_disjoint(sample_pairs):
    """Verify DOCUMENT strategy partitions by document metadata."""
    config = SplitConfig(
        strategy=SplitStrategy.DOCUMENT,
        train_ratio=0.5,
        dev_ratio=0.25,
        test_ratio=0.25,
        random_seed=42
    )
    split = DatasetSplitter.split(sample_pairs, config)
    assert len(split.train_pairs) > 0
    assert len(split.test_pairs) > 0


def test_pair_split_leakage_warning(sample_pairs):
    """Verify PAIR strategy creates pair partitions and logs warning."""
    config = SplitConfig(
        strategy=SplitStrategy.PAIR,
        train_ratio=0.6,
        dev_ratio=0.2,
        test_ratio=0.2,
        random_seed=42
    )
    split = DatasetSplitter.split(sample_pairs, config)
    assert len(split.train_pairs) > 0
    assert len(split.test_pairs) > 0
    assert split.metadata.get("warning") == "pair_level_leakage_risk"


def test_reproducibility(sample_pairs):
    """Verify identical random seeds produce identical dataset splits."""
    config1 = SplitConfig(strategy=SplitStrategy.REQUIREMENT, random_seed=123)
    config2 = SplitConfig(strategy=SplitStrategy.REQUIREMENT, random_seed=123)
    config3 = SplitConfig(strategy=SplitStrategy.REQUIREMENT, random_seed=999)

    split1 = DatasetSplitter.split(sample_pairs, config1)
    split2 = DatasetSplitter.split(sample_pairs, config2)
    split3 = DatasetSplitter.split(sample_pairs, config3)

    assert [p.id for p in split1.train_pairs] == [p.id for p in split2.train_pairs]
    assert [p.id for p in split1.test_pairs] == [p.id for p in split2.test_pairs]

    # Different seed should yield different split ordering/composition
    assert [p.id for p in split1.train_pairs] != [p.id for p in split3.train_pairs]


def test_ratio_validation():
    """Verify invalid ratio sum raises ValueError."""
    config = SplitConfig(train_ratio=0.5, dev_ratio=0.5, test_ratio=0.5)
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        config.validate_ratios()


@pytest.mark.parametrize("name,value", [
    ("train_ratio", -0.1),
    ("dev_ratio", 1.1),
    ("test_ratio", float("nan")),
])
def test_split_config_rejects_out_of_range_or_nonfinite_ratios(name, value):
    with pytest.raises(ValueError):
        SplitConfig(**{name: value})


def test_system_split_rejects_cross_system_pairs(sample_pairs):
    cross_system = sample_pairs[0].model_copy(update={
        "id": "PAIR-CROSS-SYSTEM",
        "req_b": sample_pairs[10].req_b,
    })
    with pytest.raises(ValueError, match="spans"):
        DatasetSplitter.split([cross_system], SplitConfig(strategy=SplitStrategy.SYSTEM))


def test_system_leakage_detected_for_single_overlapping_system():
    train_pair = RequirementPair(
        id="TRAIN-SYS-1",
        req_a=Requirement(id="A1", text="Train A", system="SYS1"),
        req_b=Requirement(id="A2", text="Train B", system="SYS1"),
        dependency_type=DependencyType.REQUIRES,
    )
    test_pair = RequirementPair(
        id="TEST-SYS-1",
        req_a=Requirement(id="T1", text="Test A", system="SYS1"),
        req_b=Requirement(id="T2", text="Test B", system="SYS1"),
        dependency_type=DependencyType.REQUIRES,
    )
    split = DatasetSplit(
        train_pairs=[train_pair],
        test_pairs=[test_pair],
        config=SplitConfig(strategy=SplitStrategy.SYSTEM),
    )

    leakage = DatasetSplitter.check_leakage(split)

    assert leakage.has_leakage is True
    assert leakage.system_overlap_train_test == {"SYS1"}


def test_edge_case_empty_pairs():
    """Verify empty pairs list returns empty DatasetSplit without error."""
    split = DatasetSplitter.split([], SplitConfig())
    assert len(split.train_pairs) == 0
    assert len(split.test_pairs) == 0


def test_edge_case_single_system(sample_pairs):
    """Verify single system under SYSTEM strategy handles gracefully."""
    sys1_pairs = [p for p in sample_pairs if p.req_a.system == "SYS1"]
    config = SplitConfig(strategy=SplitStrategy.SYSTEM, random_seed=42)
    split = DatasetSplitter.split(sys1_pairs, config)
    assert len(split.train_pairs) == len(sys1_pairs)


def test_save_and_load_split(sample_pairs):
    """Verify DatasetSplitter.save_split writes all JSON partition files correctly."""
    config = SplitConfig(strategy=SplitStrategy.REQUIREMENT, train_ratio=0.6, dev_ratio=0.2, test_ratio=0.2)
    split = DatasetSplitter.split(sample_pairs, config)

    with tempfile.TemporaryDirectory() as tmpdir:
        DatasetSplitter.save_split(split, tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "train_pairs.json"))
        assert os.path.exists(os.path.join(tmpdir, "dev_pairs.json"))
        assert os.path.exists(os.path.join(tmpdir, "test_pairs.json"))
        assert os.path.exists(os.path.join(tmpdir, "split_metadata.json"))

        with open(os.path.join(tmpdir, "split_metadata.json"), "r") as f:
            meta = json.load(f)
            assert "summary" in meta
            assert "leakage_report" in meta


def test_cli_split_command(tmp_path, sample_pairs):
    """Test CLI `leredd-cli split` command execution."""
    runner = CliRunner()
    input_json = tmp_path / "pairs.json"
    input_json.write_text(
        json.dumps([pair.model_dump(mode="json") for pair in sample_pairs]),
        encoding="utf-8",
    )
    out_dir = str(tmp_path / "cli_splits")

    result = runner.invoke(app, ["split", "--input", str(input_json), "--output-dir", out_dir, "--strategy", "requirement", "--seed", "42"])
    assert result.exit_code == 0, f"CLI output: {result.output}"
    assert "DATASET SPLIT SUMMARY" in result.output
    assert os.path.exists(os.path.join(out_dir, "train_pairs.json"))
    assert os.path.exists(os.path.join(out_dir, "test_pairs.json"))
