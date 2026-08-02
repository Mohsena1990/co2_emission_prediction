"""
Google COVID-19 mobility data ingestion (spec section 4).

Source: Google's official historical Global Mobility Report CSV
(https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv), a
large multi-country file covering 2020-02-15 through the report's
discontinuation in mid-October 2022. This module downloads it with chunked
reading (spec 4.1's exact recipe - the file is too large to load in one
operation), filters to UK NATIONAL rows only (country GB, no sub_region),
and caches the filtered result locally so the multi-hundred-MB source file
is never re-downloaded on subsequent runs.

The six substantive columns are daily PERCENTAGE DEVIATIONS from a
pre-COVID baseline, not absolute mobility levels - see aggregate.py for the
quarterly aggregation and the neutral-zero convention outside the official
reporting window (spec 4.2/4.4).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from ..core.logging_utils import get_logger

GOOGLE_MOBILITY_URL = (
    "https://www.gstatic.com/covid19/mobility/"
    "Global_Mobility_Report.csv"
)

MOBILITY_COLUMNS = [
    "country_region_code",
    "country_region",
    "sub_region_1",
    "sub_region_2",
    "date",
    "retail_and_recreation_percent_change_from_baseline",
    "grocery_and_pharmacy_percent_change_from_baseline",
    "parks_percent_change_from_baseline",
    "transit_stations_percent_change_from_baseline",
    "workplaces_percent_change_from_baseline",
    "residential_percent_change_from_baseline",
]

# spec 4.3: maps the Google column names to the pipeline's feature names.
CATEGORY_COLUMN_MAP = {
    "retail_and_recreation_percent_change_from_baseline": "Mobility_Retail_Recreation",
    "grocery_and_pharmacy_percent_change_from_baseline": "Mobility_Grocery_Pharmacy",
    "parks_percent_change_from_baseline": "Mobility_Parks",
    "transit_stations_percent_change_from_baseline": "Mobility_Transit_Stations",
    "workplaces_percent_change_from_baseline": "Mobility_Workplaces",
    "residential_percent_change_from_baseline": "Mobility_Residential",
}

CACHE_FILENAME = "google_uk_national_mobility.csv"
METADATA_FILENAME = "google_uk_national_mobility.metadata.json"
CHUNK_SIZE = 250_000


def _uk_national_mask(chunk: pd.DataFrame) -> pd.Series:
    return (
        (
            chunk["country_region_code"].eq("GB")
            | chunk["country_region"].eq("United Kingdom")
        )
        & chunk["sub_region_1"].isna()
        & chunk["sub_region_2"].isna()
    )


def load_uk_google_mobility(
    url: str = GOOGLE_MOBILITY_URL,
    chunksize: int = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Chunked download + UK-national filter (spec 4.1's exact recipe). Never
    loads the full multi-country file into memory at once.
    """
    logger = get_logger()
    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        url,
        usecols=MOBILITY_COLUMNS,
        chunksize=chunksize,
        low_memory=False,
    ):
        selected = chunk.loc[_uk_national_mask(chunk)].copy()
        if not selected.empty:
            frames.append(selected)

    if not frames:
        raise RuntimeError(
            "No national United Kingdom mobility records were found."
        )

    mobility = pd.concat(frames, ignore_index=True)
    mobility["date"] = pd.to_datetime(mobility["date"], errors="raise")
    logger.info(f"Filtered {len(mobility)} UK-national daily mobility records from {url}")

    return mobility.sort_values("date").reset_index(drop=True)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _build_retrieval_metadata(mobility: pd.DataFrame, url: str, cache_path: Path) -> dict:
    category_cols = list(CATEGORY_COLUMN_MAP.keys())
    return {
        "source_url": url,
        "retrieval_date": datetime.now(timezone.utc).isoformat(),
        "cache_file": str(cache_path),
        "file_checksum_sha256": _sha256_of_file(cache_path),
        "first_observation": str(mobility["date"].min().date()),
        "last_observation": str(mobility["date"].max().date()),
        "n_daily_records": int(len(mobility)),
        "missing_values_by_category": {
            col: int(mobility[col].isna().sum()) for col in category_cols if col in mobility.columns
        },
    }


def fetch_and_cache_uk_mobility(
    cache_dir: Path,
    url: str = GOOGLE_MOBILITY_URL,
    force: bool = False,
) -> Path:
    """
    Fetch (or reuse the cached copy of) the UK-national daily mobility
    series. Idempotent by cache-file existence, same convention as
    src/grid/fetch.py::fetch_and_cache_range - safe to re-run.

    Returns the path to the cached CSV.
    """
    logger = get_logger()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / CACHE_FILENAME
    metadata_path = cache_dir / METADATA_FILENAME

    if cache_path.exists() and not force:
        logger.debug(f"Mobility cache hit: {cache_path}")
        return cache_path

    mobility = load_uk_google_mobility(url=url)
    mobility.to_csv(cache_path, index=False)

    metadata = _build_retrieval_metadata(mobility, url, cache_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Cached UK mobility data: {cache_path} ({len(mobility)} daily records)")
    return cache_path


def load_cached_uk_mobility(cache_dir: Path) -> pd.DataFrame:
    """Load the cached UK-national daily mobility CSV written by
    fetch_and_cache_uk_mobility(). Raises FileNotFoundError if not yet fetched."""
    cache_path = Path(cache_dir) / CACHE_FILENAME
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached mobility data at {cache_path}. Run "
            f"fetch_and_cache_uk_mobility() first (e.g. scripts/fetch_mobility_data.py)."
        )
    return pd.read_csv(cache_path, parse_dates=["date"])


def load_cached_metadata(cache_dir: Path) -> Optional[dict]:
    metadata_path = Path(cache_dir) / METADATA_FILENAME
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)
