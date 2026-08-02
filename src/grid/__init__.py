"""
Electricity-grid data fusion module (spec section 6).

Ingests half-hourly GB Carbon Intensity API data, caches raw responses, and
aggregates to quarterly electricity-system predictors used by data
configurations A3 (raw + grid) and A4 (raw + engineered + grid).
"""
from .fetch import fetch_and_cache_range, EARLIEST_AVAILABLE_DATE
from .aggregate import load_raw_halfhourly, aggregate_to_quarterly

__all__ = [
    'fetch_and_cache_range',
    'EARLIEST_AVAILABLE_DATE',
    'load_raw_halfhourly',
    'aggregate_to_quarterly',
]
