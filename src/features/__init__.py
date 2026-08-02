"""
Feature engineering module for CO2 forecasting framework.
"""
from .engineering import (
    create_target_variable,
    create_lag_features,
    create_seasonality_features,
    create_shock_features,
    create_rolling_features,
    create_intensity_features,
    create_direct_horizon_targets,
    engineer_features,
    create_feature_dictionary
)
from .registry import (
    FeatureRegistry, GOVERNED_FAMILY_NAMES, CONFIGURATION_TAGS,
    STREAM_B_TAGS, STREAM_B_TO_A,
)
from .configurations import (
    build_configuration_matrices, configuration_manifest,
    exclude_target_derived_features, NAMED_TARGET_DERIVED_FEATURES
)

__all__ = [
    'create_target_variable',
    'create_lag_features',
    'create_seasonality_features',
    'create_shock_features',
    'create_rolling_features',
    'create_intensity_features',
    'create_direct_horizon_targets',
    'engineer_features',
    'create_feature_dictionary',
    'FeatureRegistry',
    'GOVERNED_FAMILY_NAMES',
    'CONFIGURATION_TAGS',
    'STREAM_B_TAGS',
    'STREAM_B_TO_A',
    'build_configuration_matrices',
    'configuration_manifest',
    'exclude_target_derived_features',
    'NAMED_TARGET_DERIVED_FEATURES',
]
