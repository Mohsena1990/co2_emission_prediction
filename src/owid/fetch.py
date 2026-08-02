"""
OWID annual renewable/low-carbon electricity share ingestion (spec section 8).

Source: Our World in Data's grapher CSV exports for
share-electricity-renewables and share-electricity-low-carbon (public,
keyless, annual, per-country). Used as an independently-sourced, lagged
structural predictor - NOT the primary quarterly green-share signal (that
comes from NESO's high-frequency generation mix, src/grid/) - and as an
annual external validation series / data-quality cross-check (spec 8.2).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import requests

from ..core.logging_utils import get_logger

OWID_RENEWABLE_URL = (
    "https://ourworldindata.org/grapher/"
    "share-electricity-renewables.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)

OWID_LOW_CARBON_URL = (
    "https://ourworldindata.org/grapher/"
    "share-electricity-low-carbon.csv"
    "?v=1&csvType=full&useColumnShortNames=false"
)

OWID_RENEWABLE_METADATA_URL = (
    "https://ourworldindata.org/grapher/"
    "share-electricity-renewables.metadata.json"
    "?v=1&csvType=full&useColumnShortNames=false"
)

OWID_LOW_CARBON_METADATA_URL = (
    "https://ourworldindata.org/grapher/"
    "share-electricity-low-carbon.metadata.json"
    "?v=1&csvType=full&useColumnShortNames=false"
)

USER_AGENT = "Q-DECEM data fetch/1.0"

# (kind, csv_url, metadata_url, csv_filename, metadata_filename)
_SOURCES = {
    "renewable": (
        OWID_RENEWABLE_URL, OWID_RENEWABLE_METADATA_URL,
        "uk_renewable_electricity_share.csv", "renewable_metadata.json",
    ),
    "low_carbon": (
        OWID_LOW_CARBON_URL, OWID_LOW_CARBON_METADATA_URL,
        "uk_low_carbon_electricity_share.csv", "low_carbon_metadata.json",
    ),
}


def load_owid_uk_series(
    csv_url: str,
    metadata_url: str,
) -> Tuple[pd.DataFrame, dict]:
    """spec 8.1's exact recipe: download, filter to United Kingdom, return
    (uk_dataframe, metadata_dict) - value-column name is NOT assumed here,
    see resolve_value_column()."""
    data = pd.read_csv(
        csv_url,
        storage_options={"User-Agent": USER_AGENT},
    )

    metadata_response = requests.get(
        metadata_url,
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    metadata_response.raise_for_status()

    uk = data.loc[data["Entity"].eq("United Kingdom")].copy()

    if uk.empty:
        raise RuntimeError(
            "United Kingdom was not found in the OWID dataset."
        )

    return uk, metadata_response.json()


def resolve_value_column(uk_df: pd.DataFrame, metadata: dict) -> str:
    """
    Identify the annual-share value column without hardcoding its name
    (spec 8.1: "do not hardcode the value-column name without validating it
    against the downloaded file and metadata" - the renewable and
    low-carbon CSVs use DIFFERENT column names, "Renewables" vs. "Share of
    electricity from low-carbon sources", confirmed by direct probe).

    OWID's grapher CSV format is standardized as Entity/Code/Year plus
    exactly one indicator column - identify that column structurally, then
    cross-check the metadata declares exactly one indicator too.
    """
    candidate_cols = [c for c in uk_df.columns if c not in ("Entity", "Code", "Year")]
    if len(candidate_cols) != 1:
        raise ValueError(
            f"Expected exactly one OWID value column beyond Entity/Code/Year, "
            f"found {candidate_cols}. Refusing to guess - validate against "
            f"the downloaded file before proceeding (spec section 8.1)."
        )
    metadata_columns = metadata.get("columns", {})
    if len(metadata_columns) != 1:
        raise ValueError(
            f"Expected exactly one indicator in OWID metadata['columns'], "
            f"found {len(metadata_columns)}: {list(metadata_columns)}."
        )
    return candidate_cols[0]


def fetch_and_cache_owid(cache_dir: Path, force: bool = False) -> Dict[str, Path]:
    """
    Fetch (or reuse cached copies of) both UK OWID series. Idempotent by
    cache-file existence, same convention as src/grid/fetch.py and
    src/mobility/fetch.py.

    Returns {'renewable': csv_path, 'low_carbon': csv_path}.
    """
    logger = get_logger()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result: Dict[str, Path] = {}
    for kind, (csv_url, metadata_url, csv_filename, metadata_filename) in _SOURCES.items():
        csv_path = cache_dir / csv_filename
        metadata_path = cache_dir / metadata_filename

        if csv_path.exists() and metadata_path.exists() and not force:
            logger.debug(f"OWID cache hit ({kind}): {csv_path}")
            result[kind] = csv_path
            continue

        uk_df, metadata = load_owid_uk_series(csv_url, metadata_url)
        resolve_value_column(uk_df, metadata)  # fail fast if the schema ever changes
        uk_df.to_csv(csv_path, index=False)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Cached OWID {kind} share data: {csv_path} ({len(uk_df)} annual records)")
        result[kind] = csv_path

    return result


def load_cached_owid_uk_series(cache_dir: Path, kind: str) -> Tuple[pd.DataFrame, dict]:
    """Load a cached UK OWID series + its metadata. kind: 'renewable' or 'low_carbon'."""
    if kind not in _SOURCES:
        raise ValueError(f"Unknown OWID kind '{kind}', expected one of {list(_SOURCES)}")
    _, _, csv_filename, metadata_filename = _SOURCES[kind]
    cache_dir = Path(cache_dir)
    csv_path = cache_dir / csv_filename
    metadata_path = cache_dir / metadata_filename

    if not csv_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"No cached OWID '{kind}' data in {cache_dir}. Run "
            f"fetch_and_cache_owid() first (e.g. scripts/fetch_owid_data.py)."
        )

    uk_df = pd.read_csv(csv_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return uk_df, metadata
