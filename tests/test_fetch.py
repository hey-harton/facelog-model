"""Unit tests for Kaggle data fetching and dataset/raw/<name>/ grouping."""

from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest

from src.data.fetch import DataFetchConfig, DatasetSpec, fetch_dataset, load_fetch_config


def test_load_fetch_config_groups_by_dataset_name() -> None:
    """Config loads all 3 datasets and maps each to dataset/raw/<name>."""
    config = load_fetch_config(Path("configs/data.yaml"))

    assert config.source_type == "kaggle"
    assert set(config.datasets.keys()) == {"vggface2", "pins", "lfw"}
    assert config.datasets["vggface2"].handle == "hearfool/vggface2"
    assert config.datasets["vggface2"].target_dir == Path("dataset/raw/vggface2")
    assert config.datasets["pins"].handle == "hereisburak/pins-face-recognition"
    assert config.datasets["pins"].target_dir == Path("dataset/raw/pins")
    assert config.datasets["lfw"].handle == "jessicali9530/lfw-dataset"
    assert config.datasets["lfw"].target_dir == Path("dataset/raw/lfw")


def test_fetch_dataset_dry_run_creates_target_without_download(tmp_path: Path) -> None:
    """Dry-run creates target directory without calling kagglehub."""
    raw_dir = tmp_path / "dataset" / "raw"
    config = DataFetchConfig(
        source_type="kaggle",
        raw_dir=raw_dir,
        datasets={
            "lfw": DatasetSpec(
                name="lfw",
                handle="jessicali9530/lfw-dataset",
                role="threshold_calibration_1v1",
                target_dir=raw_dir / "lfw",
            )
        },
    )

    result = fetch_dataset(config, dataset_name="lfw", dry_run=True)

    assert result == {"lfw": raw_dir / "lfw"}
    assert (raw_dir / "lfw").is_dir()


def test_fetch_dataset_preserves_internal_structure_and_respects_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Downloaded files preserve original subdirectories and respect force flag."""
    fake_cache = tmp_path / "kaggle_cache"
    (fake_cache / "lfw-deepfunneled" / "Person_A").mkdir(parents=True)
    sample_img = fake_cache / "lfw-deepfunneled" / "Person_A" / "Person_A_0001.jpg"
    sample_img.write_bytes(b"img-v1")
    pairs_csv = fake_cache / "matchpairsDevTest.csv"
    pairs_csv.write_text("name,imagenum1,imagenum2\nPerson_A,1,2\n", encoding="utf-8")

    fake_module = types.SimpleNamespace(dataset_download=lambda _handle: str(fake_cache))
    monkeypatch.setitem(sys.modules, "kagglehub", fake_module)

    raw_dir = tmp_path / "dataset" / "raw"
    target_dir = raw_dir / "lfw"
    target_dir.mkdir(parents=True)
    (target_dir / ".gitkeep").write_text("", encoding="utf-8")

    config = DataFetchConfig(
        source_type="kaggle",
        raw_dir=raw_dir,
        datasets={
            "lfw": DatasetSpec(
                name="lfw",
                handle="jessicali9530/lfw-dataset",
                role="threshold_calibration_1v1",
                target_dir=target_dir,
            )
        },
    )

    fetch_dataset(config, dataset_name="lfw", dry_run=False, force=False)
    copied_img = target_dir / "lfw-deepfunneled" / "Person_A" / "Person_A_0001.jpg"
    assert copied_img.read_bytes() == b"img-v1"
    assert (target_dir / "matchpairsDevTest.csv").exists()

    # Modify source and verify force=False skips re-copy
    sample_img.write_bytes(b"img-v2")
    fetch_dataset(config, dataset_name="lfw", dry_run=False, force=False)
    assert copied_img.read_bytes() == b"img-v1"

    # Verify force=True updates files while keeping .gitkeep
    fetch_dataset(config, dataset_name="lfw", dry_run=False, force=True)
    assert copied_img.read_bytes() == b"img-v2"
    assert (target_dir / ".gitkeep").exists()
