#!/usr/bin/env python
"""
Fetch and cache OWID UK renewable/low-carbon electricity share data (spec
section 8).

Usage:
    python scripts/fetch_owid_data.py [--config CONFIG_PATH] [--force]

Separate, network-bound backfill step: downloads both OWID grapher CSVs
(renewable + low-carbon share) plus their metadata JSON, filters to the
United Kingdom, and caches the result (idempotent - fetch_and_cache_owid
skips the download if both the CSV and metadata files already exist, so
re-running only re-fetches with --force, e.g. to pick up OWID's next
annual update).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Config, setup_logging
from src.owid import fetch_and_cache_owid


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and cache OWID UK electricity-share data')
    parser.add_argument('--config', type=str, default=None,
                         help='Path to configuration file (for owid.cache_dir)')
    parser.add_argument('--force', action='store_true',
                         help='Re-download even if cached copies already exist')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='owid_fetch')

    cache_dir = Path(config.owid.cache_dir)
    logger.info("=" * 60)
    logger.info("Fetching OWID UK renewable/low-carbon electricity share data")
    logger.info("=" * 60)
    logger.info(f"Cache dir: {cache_dir}")

    paths = fetch_and_cache_owid(cache_dir, force=args.force)

    for kind, path in paths.items():
        logger.info(f"  {kind}: {path}")
    logger.info("Done.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
