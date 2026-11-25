"""Data fetching utilities for populating dataset/raw."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DataFetchConfig:
    """Configuration for pulling raw face datasets."""

    source_type: str
    endpoint: str
    raw_dir: Path
    timeout_seconds: int = 30


def load_fetch_config(config_path: Path) -> DataFetchConfig:
    """Load data fetch configuration from YAML and apply environment overrides."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load configs/data.yaml. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    paths_cfg = raw_config.get("paths", {})
    source_cfg = raw_config.get("source", {})

    endpoint = os.environ.get("DATA_SOURCE_URL") or source_cfg.get("endpoint", "")
    raw_dir = Path(paths_cfg.get("raw_dir", "dataset/raw"))
    source_type = str(source_cfg.get("type", "local")).strip().lower()
    timeout_seconds = int(source_cfg.get("timeout_seconds", 30))

    return DataFetchConfig(
        source_type=source_type,
        endpoint=str(endpoint).strip(),
        raw_dir=raw_dir,
        timeout_seconds=timeout_seconds,
    )


def fetch_dataset(config: DataFetchConfig, dry_run: bool = False) -> Path:
    """Fetch raw dataset into config.raw_dir.

    Args:
        config: Parsed data fetch configuration.
        dry_run: When True, validate configuration and ensure target directory
            exists without downloading from remote sources.

    Returns:
        Path to the raw dataset directory.
    """
    if not config.source_type:
        raise ValueError("source.type in data config must not be empty.")
    if config.timeout_seconds <= 0:
        raise ValueError("source.timeout_seconds must be greater than 0.")

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[fetch_dataset] source_type={config.source_type} "
        f"raw_dir={config.raw_dir} dry_run={dry_run}"
    )

    if dry_run:
        return config.raw_dir

    if config.source_type == "local":
        print(f"[fetch_dataset] Using local files in {config.raw_dir}")
        return config.raw_dir

    raise NotImplementedError(
        f"Remote data fetching for source_type='{config.source_type}' is not "
        "implemented yet. Implement the connector in src/data/fetch.py."
    )
