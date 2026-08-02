"""
Google COVID-19 mobility-shock data fusion module (spec section 4).

Ingests Google's Global Mobility Report CSV, filters to UK-national daily
records, caches locally, and aggregates to the six quarterly
`Mobility_*` shock indicators used by data configurations A1-A4.
"""
from .fetch import (
    GOOGLE_MOBILITY_URL, MOBILITY_COLUMNS, CATEGORY_COLUMN_MAP,
    load_uk_google_mobility, fetch_and_cache_uk_mobility,
    load_cached_uk_mobility, load_cached_metadata,
)
from .aggregate import aggregate_to_quarterly, apply_neutral_zero_convention

__all__ = [
    'GOOGLE_MOBILITY_URL',
    'MOBILITY_COLUMNS',
    'CATEGORY_COLUMN_MAP',
    'load_uk_google_mobility',
    'fetch_and_cache_uk_mobility',
    'load_cached_uk_mobility',
    'load_cached_metadata',
    'aggregate_to_quarterly',
    'apply_neutral_zero_convention',
]
