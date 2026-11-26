"""Kaggle dataset fetching utilities for populating dataset/raw/<dataset_name>/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


@dataclass(frozen=True)
class DatasetSpec:
    """Specification for a single named dataset."""

    name: str
    handle: str
    role: str
    target_dir: Path


@dataclass(frozen=True)
class DataFetchConfig:
    """Configuration for pulling raw face datasets."""

    source_type: str
    raw_dir: Path
    datasets: dict[str, DatasetSpec]
    timeout_seconds: int = 60


def load_fetch_config(config_path: Path) -> DataFetchConfig:
    """Load data fetch configuration from YAML."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load configs/data.yaml. "
            "Install dependencies with: pip install -r requirements-dev.txt"
        ) from exc

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    paths_cfg = raw_config.get("paths", {})
    source_cfg = raw_config.get("source", {})
    datasets_cfg = raw_config.get("datasets", {})

    raw_dir = Path(paths_cfg.get("raw_dir", "dataset/raw"))
    source_type = str(source_cfg.get("type", "kaggle")).strip().lower()
    timeout_seconds = int(source_cfg.get("timeout_seconds", 60))

    datasets: dict[str, DatasetSpec] = {}
    for name, item in datasets_cfg.items():
        if not isinstance(item, dict):
            raise ValueError(f"Invalid dataset entry for '{name}' in {config_path}")
        ds_handle = str(item.get("handle", "")).strip()
        if not ds_handle:
            raise ValueError(f"Dataset '{name}' is missing 'handle' in {config_path}")
        subdir = str(item.get("subdir", name)).strip()
        role = str(item.get("role", "")).strip()
        datasets[name] = DatasetSpec(
            name=name,
            handle=ds_handle,
            role=role,
            target_dir=raw_dir / subdir,
        )

    if not datasets:
        raise ValueError(f"No datasets configured under 'datasets' in {config_path}")

    return DataFetchConfig(
        source_type=source_type,
        raw_dir=raw_dir,
        datasets=datasets,
        timeout_seconds=timeout_seconds,
    )


def _has_dataset_files(target_dir: Path) -> bool:
    """Return True if target_dir already contains files/dirs other than .gitkeep."""
    if not target_dir.exists():
        return False
    return any(entry.name != ".gitkeep" for entry in target_dir.iterdir())


def _clean_target_dir(target_dir: Path) -> None:
    """Remove existing dataset contents in target_dir while preserving .gitkeep."""
    for entry in target_dir.iterdir():
        if entry.name == ".gitkeep":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _copy_dataset_tree(src_dir: Path, dst_dir: Path) -> int:
    """Copy downloaded dataset files into dst_dir preserving internal structure."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Downloaded dataset path not found: {src_dir}")

    copied = 0
    for item in src_dir.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src_dir)
            dest_file = dst_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
            copied += 1
    return copied


def fetch_dataset(
    config: DataFetchConfig,
    dataset_name: str = "all",
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """Fetch selected Kaggle dataset(s) into dataset/raw/<dataset_name>/.

    Args:
        config: Parsed data fetch configuration.
        dataset_name: Either 'all' or a configured dataset key (vggface2, pins, lfw).
        dry_run: When True, validate configuration and ensure target directories
            exist without downloading.
        force: When True, re-copy and overwrite existing files in target_dir.

    Returns:
        Mapping of selected dataset names to their target raw directories.
    """
    if not config.source_type:
        raise ValueError("source.type in data config must not be empty.")
    if config.timeout_seconds <= 0:
        raise ValueError("source.timeout_seconds must be greater than 0.")

    if dataset_name == "all":
        selected = list(config.datasets.keys())
    elif dataset_name in config.datasets:
        selected = [dataset_name]
    else:
        available = ", ".join(config.datasets.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available datasets: {available}"
        )

    results: dict[str, Path] = {}
    for name in selected:
        spec = config.datasets[name]
        spec.target_dir.mkdir(parents=True, exist_ok=True)
        results[name] = spec.target_dir

        if dry_run:
            print(
                f"[fetch_dataset] DRY-RUN dataset={name} "
                f"handle={spec.handle} target_dir={spec.target_dir}"
            )
            continue

        if _has_dataset_files(spec.target_dir) and not force:
            print(
                f"[fetch_dataset] SKIP dataset={name} "
                f"(already populated at {spec.target_dir}; use --force to re-copy)"
            )
            continue

        if config.source_type != "kaggle":
            raise NotImplementedError(
                f"Data fetching for source_type='{config.source_type}' is not "
                "implemented yet."
            )

        try:
            import kagglehub
        except ImportError as exc:
            raise ImportError(
                "kagglehub is required for source.type='kaggle'. "
                "Install dependencies with: pip install -r requirements-dev.txt"
            ) from exc

        print(
            f"[fetch_dataset] Downloading dataset={name} handle={spec.handle} ..."
        )
        downloaded_path = Path(kagglehub.dataset_download(spec.handle))
        if force and _has_dataset_files(spec.target_dir):
            _clean_target_dir(spec.target_dir)

        copied = _copy_dataset_tree(downloaded_path, spec.target_dir)
        print(
            f"[fetch_dataset] OK dataset={name} files={copied} "
            f"target_dir={spec.target_dir}"
        )

    return results
