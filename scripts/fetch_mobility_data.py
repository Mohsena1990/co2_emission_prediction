#!/usr/bin/env python
"""
Fetch and cache Google COVID-19 mobility data (spec section 4).

Usage:
    python scripts/fetch_mobility_data.py [--config CONFIG_PATH] [--force]

This is a separate, network-bound backfill step from the main 00-06
pipeline sequence: it downloads Google's Global Mobility Report CSV once,
filters to UK-national daily records, and caches the result (idempotent -
fetch_and_cache_uk_mobility skips the download if the cache file already
exists, so re-running is a no-op unless --force is passed). The historical
Google archive is fixed/discontinued (spec 4.1: mid-October 2022), so
unlike the grid fetcher there is no "new data since last run" case to
handle - re-fetching is only ever needed with --force (e.g. to refresh the
checksum/metadata sidecar).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Config, setup_logging
from src.mobility import fetch_and_cache_uk_mobility


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and cache Google UK mobility data')
    parser.add_argument('--config', type=str, default=None,
                         help='Path to configuration file (for mobility.cache_dir)')
    parser.add_argument('--force', action='store_true',
                         help='Re-download even if a cached copy already exists')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='mobility_fetch')

    cache_dir = Path(config.mobility.cache_dir)
    logger.info("=" * 60)
    logger.info("Fetching Google COVID-19 mobility data (UK national)")
    logger.info("=" * 60)
    logger.info(f"Cache dir: {cache_dir}")

    cache_path = fetch_and_cache_uk_mobility(cache_dir, force=args.force)

    logger.info(f"Done: cached mobility data at {cache_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
