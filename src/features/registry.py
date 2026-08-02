"""
Predictor-governance layer (spec section 5).

Loads config/feature_registry.yaml - the single machine-readable source of
truth for every raw and engineered candidate predictor - and exposes it to
the rest of the pipeline so that model matrices are built only from features
that have been explicitly classified and passed the availability/leakage
audit for the requested forecast horizon.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from ..core.logging_utils import get_logger

REQUIRED_COLUMNS = [
    'name', 'family', 'kind', 'formula', 'source_variables', 'target_derived',
    'min_lag', 'availability_category', 'release_delay_quarters',
    'leakage_risk', 'safe_at_h1', 'safe_at_h2', 'safe_at_h4',
    'missing_value_rule', 'retained_after_audit', 'exclusion_reason', 'role',
    'configuration_membership'
]

# Governed feature families (spec section 6).
GOVERNED_FAMILY_NAMES = ['G1', 'G2', 'G3', 'G4', 'G5']

# The four primary data configurations (spec section 8).
CONFIGURATION_TAGS = ['A1', 'A2', 'A3', 'A4']

# Stream B (feature-selection) candidate pools mirror their Stream A
# counterpart 1:1 (spec section 12: B1<-A1, B2<-A2, B3<-A3, B4<-A4) - not a
# separate registry column, just a naming/mapping convenience so Stream B
# code never has to hardcode "B1 means A1's membership".
STREAM_B_TAGS = ['B1', 'B2', 'B3', 'B4']
STREAM_B_TO_A = {'B1': 'A1', 'B2': 'A2', 'B3': 'A3', 'B4': 'A4'}


@dataclass
class FeatureRegistry:
    """In-memory view of the predictor-governance registry."""
    entries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: Optional[Path] = None

    @classmethod
    def load(cls, path: str = "config/feature_registry.yaml") -> "FeatureRegistry":
        logger = get_logger()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Feature registry not found at {path}. The pipeline requires "
                f"config/feature_registry.yaml (spec section 5) before any "
                f"model matrix can be built."
            )
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)

        features = raw.get('features', [])
        entries = {}
        for row in features:
            missing = [c for c in REQUIRED_COLUMNS if c not in row]
            if missing:
                raise ValueError(
                    f"Feature registry entry '{row.get('name', '?')}' is missing "
                    f"required column(s): {missing}"
                )
            entries[row['name']] = row

        logger.info(f"Loaded feature registry: {len(entries)} candidate predictors from {path}")
        return cls(entries=entries, source_path=path)

    def __len__(self) -> int:
        return len(self.entries)

    def all_names(self) -> List[str]:
        return list(self.entries.keys())

    def retained_names(self) -> List[str]:
        """Features that passed the availability/leakage audit at all."""
        return [n for n, e in self.entries.items() if e['retained_after_audit']]

    def excluded_names(self) -> List[str]:
        return [n for n, e in self.entries.items() if not e['retained_after_audit']]

    def is_retained(self, name: str) -> bool:
        entry = self.entries.get(name)
        return bool(entry and entry['retained_after_audit'])

    def is_safe_for_horizon(self, name: str, horizon: int) -> bool:
        """
        Whether a (retained) feature may be used for a direct forecast at
        the given horizon. Only features with min_lag >= 1, or raw
        exogenous features explicitly classified safe, are usable - see the
        safe_at_h{1,2,4} columns in the registry for the documented
        per-horizon judgement.
        """
        entry = self.entries.get(name)
        if entry is None or not entry['retained_after_audit']:
            return False
        key = f'safe_at_h{horizon}'
        if key not in entry:
            raise ValueError(
                f"Feature registry has no safe_at_h{horizon} classification for "
                f"'{name}'. Add horizon={horizon} coverage to "
                f"config/feature_registry.yaml before using it in a direct "
                f"h={horizon} model."
            )
        return bool(entry[key])

    def filter_for_horizon(self, candidate_names: List[str], horizon: int) -> List[str]:
        """Filter a candidate feature list down to those safe for `horizon`."""
        logger = get_logger()
        safe = [n for n in candidate_names if n in self.entries and self.is_safe_for_horizon(n, horizon)]
        dropped = sorted(set(candidate_names) - set(safe))
        if dropped:
            logger.info(f"Registry governance dropped {len(dropped)} feature(s) for H{horizon}: {dropped}")
        return safe

    def enforce(self, columns: List[str], horizon: int) -> List[str]:
        """
        Strict governance gate: raise if any requested column is not in the
        registry at all (unknown feature), then filter to horizon-safe ones.
        Use this whenever `enforce_availability_registry=True`.
        """
        unknown = [c for c in columns if c not in self.entries]
        if unknown:
            raise ValueError(
                f"enforce_availability_registry=True but the following columns "
                f"are not present in the feature registry: {unknown}. Either "
                f"add them to config/feature_registry.yaml with an explicit "
                f"leakage/availability classification, or remove them from X."
            )
        return self.filter_for_horizon(columns, horizon)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Table 2 (complete candidate-feature registry) as a DataFrame - also
        the basis for the `metadata/feature_registry.csv` export (spec
        section 15). Adds A1-A4 and B1-B4 boolean membership columns
        computed from `configuration_membership` rather than stored
        redundantly in the YAML (B1-B4 always mirror A1-A4 exactly, spec
        section 12 - storing them separately would just be a second copy
        that could silently drift out of sync).
        """
        rows = []
        for name, e in self.entries.items():
            row = dict(e)
            membership = e.get('configuration_membership') or []
            row['source_variables'] = ', '.join(e.get('source_variables', []) or [])
            row['configuration_membership'] = ', '.join(membership)
            for a_tag in CONFIGURATION_TAGS:
                row[f'{a_tag}_membership'] = a_tag in membership
            for b_tag in STREAM_B_TAGS:
                row[f'{b_tag}_candidate_membership'] = STREAM_B_TO_A[b_tag] in membership
            rows.append(row)
        return pd.DataFrame(rows)

    def export_csv(self, path: str = "metadata/feature_registry.csv") -> Path:
        """
        Write the spec section 15-mandated `metadata/feature_registry.csv`
        as a derived export of this registry - config/feature_registry.yaml
        remains the single validated source of truth (FeatureRegistry.load/
        enforce), this is a read-only artifact for the final report/tables.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    # -- Data configurations A1-A4 (spec section 8) -------------------------

    def configuration_members(self, tag: str) -> List[str]:
        """
        Names of retained features tagged as members of data configuration
        `tag` (one of 'A1'/'A2'/'A3'/'A4'). Feature order follows the
        registry's own insertion order (matches config/feature_registry.yaml).
        """
        if tag not in CONFIGURATION_TAGS:
            raise ValueError(f"Unknown configuration tag '{tag}'. Expected one of {CONFIGURATION_TAGS}")
        return [
            n for n, e in self.entries.items()
            if e['retained_after_audit'] and tag in (e.get('configuration_membership') or [])
        ]

    def all_configuration_memberships(self) -> Dict[str, List[str]]:
        return {tag: self.configuration_members(tag) for tag in CONFIGURATION_TAGS}

    # -- Stream B candidate pools (spec section 12/15) -----------------------

    def candidate_pool_for_stream_b(self, b_tag: str) -> List[str]:
        """
        Stream B's FS candidate pool for `b_tag` ('B1'-'B4') - identical to
        its Stream A counterpart's own audited membership (B1<-A1, ...,
        spec section 12). FS1-FS5 select a subset of this same pool; it is
        never a different/expanded feature set from Stream A's.
        """
        if b_tag not in STREAM_B_TAGS:
            raise ValueError(f"Unknown Stream B tag '{b_tag}'. Expected one of {STREAM_B_TAGS}")
        return self.configuration_members(STREAM_B_TO_A[b_tag])

    def all_stream_b_candidate_pools(self) -> Dict[str, List[str]]:
        return {tag: self.candidate_pool_for_stream_b(tag) for tag in STREAM_B_TAGS}

    # -- Governed families (spec section 6) --------------------------------

    def governed_family(self, family: str) -> List[str]:
        """
        Return the feature names belonging to governed family G1-G5.

        G1 Exogenous-only:        target_derived == False, retained
        G2 Autoregressive-only:   CO2e lags/growth + seasonal indicators
        G3 Combined-safe:         every retained feature (exogenous + all
                                   safely-lagged target-derived, incl.
                                   intensity ratios)
        G4 Intensity-excluded:    G3 minus family == 'target_intensity'
        G5 Full audited pool:     all retained features. In this registry G5
                                   coincides with G3 exactly (there is no
                                   additional audited-but-unclassified
                                   bucket); the label is kept distinct for
                                   spec compliance and future extensibility
                                   rather than silently merged with G3.
        """
        retained = {n: e for n, e in self.entries.items() if e['retained_after_audit']}

        if family == 'G1':
            return [n for n, e in retained.items() if not e['target_derived']]
        if family == 'G2':
            return [
                n for n, e in retained.items()
                if e['family'] in ('target_lag', 'target_growth') or e['family'] == 'seasonal'
            ]
        if family == 'G3':
            return list(retained.keys())
        if family == 'G4':
            return [n for n, e in retained.items() if e['family'] != 'target_intensity']
        if family == 'G5':
            return list(retained.keys())

        raise ValueError(f"Unknown governed family '{family}'. Expected one of {GOVERNED_FAMILY_NAMES}")

    def all_governed_families(self) -> Dict[str, List[str]]:
        return {fam: self.governed_family(fam) for fam in GOVERNED_FAMILY_NAMES}
