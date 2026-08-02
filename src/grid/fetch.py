"""
Electricity-grid data ingestion (spec section 6).

Source: the GB National Grid ESO Carbon Intensity API
(https://api.carbonintensity.org.uk), a free, keyless, official, publicly
documented data source with half-hourly carbon-intensity and generation-mix
data from 2018-01-01 onward. This is an external electricity-SYSTEM
predictor, not a re-measurement of total UK CO2e - it must never be treated
as, or substituted for, the CO2e target (see the module docstring in
`aggregate.py` and the feature-registry entries for the grid family).

Every raw HTTP response is cached to disk as-is (`cache_dir`) before any
parsing happens, keyed by the exact date range requested, so re-running the
pipeline never re-requests data already on disk - this satisfies spec
section 6's "the package must cache raw responses and avoid repeatedly
requesting the same data" requirement without needing a separate cache
layer bolted on afterward.
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import requests

from ..core.logging_utils import get_logger

API_BASE = "https://api.carbonintensity.org.uk"
CHUNK_DAYS = 28  # comfortably under the API's documented 14-31 day range limits
EARLIEST_AVAILABLE_DATE = datetime(2018, 1, 1)  # verified by direct probe (see audit)
# The spec assumes NESO grid data begin in 2009Q1 (spec section 6/14) - a
# live probe of this API (2026-08-01) confirms it returns empty results for
# every year 2008-2017 and real half-hourly data from 2018 onward. The
# common comparison period across A1-A4/B1-B4 (spec section 14) is bounded
# by this ACTUAL 2018 start, not the spec's stated 2009 assumption -
# documented here and in the final report rather than silently reconciled.


def _date_chunks(start: datetime, end: datetime, chunk_days: int = CHUNK_DAYS):
    """Yield (chunk_start, chunk_end) datetime pairs covering [start, end)."""
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        yield cur, nxt
        cur = nxt


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _cache_path(cache_dir: Path, kind: str, chunk_start: datetime, chunk_end: datetime) -> Path:
    return cache_dir / f"{kind}_{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}.json"


def _fetch_json(url: str, max_retries: int = 3, timeout: int = 30) -> dict:
    logger = get_logger()
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - network calls, any failure should retry/raise
            last_exc = e
            logger.warning(f"Grid API request failed (attempt {attempt + 1}/{max_retries}): {url} - {e}")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Grid API request failed after {max_retries} attempts: {url}") from last_exc


def fetch_and_cache_range(
    start: datetime,
    end: datetime,
    cache_dir: Path,
    kinds: Optional[List[str]] = None,
) -> List[Path]:
    """
    Fetch (or reuse cached) raw half-hourly carbon-intensity and
    generation-mix data for [start, end), chunked into <= CHUNK_DAYS windows
    to stay within the API's per-request range limits.

    Args:
        start, end: UTC datetime bounds (end exclusive).
        cache_dir: directory to store/read raw JSON responses.
        kinds: subset of {'intensity', 'generation'} to fetch (default both).

    Returns:
        List of cache file paths written or reused (both kinds interleaved).
    """
    logger = get_logger()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kinds = kinds or ["intensity", "generation"]

    paths = []
    for chunk_start, chunk_end in _date_chunks(start, end):
        for kind in kinds:
            path = _cache_path(cache_dir, kind, chunk_start, chunk_end)
            if path.exists():
                logger.debug(f"Grid cache hit: {path.name}")
                paths.append(path)
                continue

            url = f"{API_BASE}/{kind}/{_fmt(chunk_start)}/{_fmt(chunk_end)}"
            logger.info(f"Fetching grid {kind} data: {chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}")
            payload = _fetch_json(url)

            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            paths.append(path)

    logger.info(f"Grid data cache ready: {len(paths)} files in {cache_dir}")
    return paths
