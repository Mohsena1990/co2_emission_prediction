#!/usr/bin/env python
"""
Fetch and cache GB Carbon Intensity API historical data (spec section 6).

Usage:
    python scripts/fetch_grid_data.py [--config CONFIG_PATH] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

This is a separate, network-bound backfill step from the main 00-06
pipeline sequence: it downloads and caches raw half-hourly grid data once
(idempotent - fetch_and_cache_range skips any chunk file already on disk,
so re-running only fetches genuinely new data, e.g. more recent quarters),
then scripts/00_make_dataset.py aggregates the cache into quarterly A3/A4
grid features on every run without hitting the network again.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import Config, setup_logging
from src.grid import fetch_and_cache_range, EARLIEST_AVAILABLE_DATE


def parse_args():
    parser = argparse.ArgumentParser(description='Fetch and cache GB grid data')
    parser.add_argument('--config', type=str, default=None,
                         help='Path to configuration file (for grid.cache_dir)')
    parser.add_argument('--start', type=str, default=None,
                         help='YYYY-MM-DD (default: API earliest available, 2018-01-01)')
    parser.add_argument('--end', type=str, default=None,
                         help='YYYY-MM-DD, exclusive (default: today, UTC)')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.config and Path(args.config).exists():
        config = Config.load(args.config)
    else:
        config = Config()

    logger = setup_logging(log_dir=Path('outputs/logs'), run_id='grid_fetch')

    start = datetime.strptime(args.start, '%Y-%m-%d') if args.start else EARLIEST_AVAILABLE_DATE
    end = (
        datetime.strptime(args.end, '%Y-%m-%d') if args.end
        else datetime.now(timezone.utc).replace(tzinfo=None)
    )

    cache_dir = Path(config.grid.cache_dir)
    logger.info("=" * 60)
    logger.info("Fetching GB Carbon Intensity API historical data")
    logger.info("=" * 60)
    logger.info(f"Range: [{start:%Y-%m-%d}, {end:%Y-%m-%d}) -> cache: {cache_dir}")

    paths = fetch_and_cache_range(start, end, cache_dir)

    logger.info(f"Done: {len(paths)} cache files ready in {cache_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
