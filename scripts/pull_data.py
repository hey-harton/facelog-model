"""CLI entrypoint to pull Kaggle face datasets into dataset/raw/<name>/."""

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
    """Parse command-line arguments for pulling datasets."""
    parser = argparse.ArgumentParser(
        description="Pull Kaggle face datasets into dataset/raw/<dataset_name>/."
    )
    parser.add_argument(
        "--dataset",
        choices=["all", "vggface2", "pins", "lfw"],
        default="all",
        help="Dataset to pull (default: all).",
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
        help="Validate configuration and target directories without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-copy and overwrite existing files in target directory.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the data pull CLI."""
    _load_env_if_available()
    args = parse_args()
    config = load_fetch_config(args.config)
    fetch_dataset(
        config,
        dataset_name=args.dataset,
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
