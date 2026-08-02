"""
OWID annual renewable/low-carbon electricity share module (spec section 8).

Ingests Our World in Data's UK renewable/low-carbon electricity-share
series, caches locally, and builds the two one-year-lagged quarterly
structural features (`OWID_RenewableShare_L1Y`, `OWID_LowCarbonShare_L1Y`)
used by data configurations A3 (raw + grid) and A4 (raw + engineered + grid).
"""
from .fetch import (
    OWID_RENEWABLE_URL, OWID_LOW_CARBON_URL,
    OWID_RENEWABLE_METADATA_URL, OWID_LOW_CARBON_METADATA_URL,
    load_owid_uk_series, resolve_value_column,
    fetch_and_cache_owid, load_cached_owid_uk_series,
)
from .lag import build_owid_lagged_feature

__all__ = [
    'OWID_RENEWABLE_URL',
    'OWID_LOW_CARBON_URL',
    'OWID_RENEWABLE_METADATA_URL',
    'OWID_LOW_CARBON_METADATA_URL',
    'load_owid_uk_series',
    'resolve_value_column',
    'fetch_and_cache_owid',
    'load_cached_owid_uk_series',
    'build_owid_lagged_feature',
]
