"""CLI entrypoint to pull raw dataset files into dataset/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fetch import fetch_dataset, load_fetch_config


def _load_env_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pulling data."""
    parser = argparse.ArgumentParser(
        description="Pull raw face dataset into dataset/raw/."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "data.yaml",
        help="Path to data configuration YAML file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and target directory without fetching remote data.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the data pull CLI."""
    _load_env_if_available()
    args = parse_args()
    config = load_fetch_config(args.config)
    fetch_dataset(config, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
