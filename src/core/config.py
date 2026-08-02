"""
Configuration management for CO2 forecasting framework.
"""
import os
import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime


@dataclass
class DataConfig:
    """Data loading configuration."""
    input_path: str = "data/raw/data 1999-2025Q1.xlsx"
    date_column: str = "Quarter"  # or "Date"
    target_column: str = "CO2e"
    target_transform: str = "log"  # "log" or "delta_log"
    sheet_name: Optional[str] = None


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    lag_features: List[str] = field(default_factory=lambda: ["CO2e"])
    lag_orders: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    seasonality_type: str = "dummies"  # "dummies" or "sincos"
    include_covid_dummy: bool = True
    covid_start: str = "2020Q1"
    covid_end: str = "2021Q4"
    include_energy_crisis: bool = True
    energy_crisis_start: str = "2022Q1"
    energy_crisis_end: str = "2023Q4"
    # Rolling features configuration
    include_rolling_features: bool = False
    rolling_columns: List[str] = field(default_factory=lambda: ["CO2e"])
    rolling_windows: List[int] = field(default_factory=lambda: [4, 8])
    rolling_functions: List[str] = field(default_factory=lambda: ["mean", "std"])
    # Rate-of-change / growth features (Δlog)
    include_roc_features: bool = True
    include_target_growth: bool = True  # alias used in spec section 24 docs
    # TEC is fully removed per spec section 2 - never add it back here.
    roc_columns: List[str] = field(default_factory=lambda: ["GDP"])
    # Intensity/per-capita features - TARGET-DERIVED. Bug 3.1: these divide a
    # *lagged* CO2e by a *lagged* denominator (see features/engineering.py
    # create_intensity_features); the numerator/denominator are never
    # contemporaneous with the target being forecast. Opt-in and off by
    # default per spec 3.1 ("Set target-derived feature groups to opt-in
    # rather than silently enabled").
    include_intensity_features: bool = False
    # TEC is fully removed per spec section 2 - never add it back here.
    intensity_denominators: List[str] = field(default_factory=lambda: ["Population"])
    intensity_min_lag: int = 1  # must be >= 1; enforced in engineering.py
    # Weather features (HDD/CDD proxy)
    include_weather_features: bool = True
    temperature_column: str = "Air_Temp"
    hdd_base_temp: float = 18.0
    # NOTE: CEI is fully removed per spec section 2 (revised spec supersedes
    # the earlier CEI_lag1-governance approach, bug 3.2) - no CEI-related
    # config fields remain; raw CEI is dropped structurally in
    # engineer_features() regardless of any flag.
    # Predictor-governance layer (spec section 5): when True, the feature
    # matrix builder consults config/feature_registry.yaml and refuses to
    # include any feature not marked retained-after-audit for the requested
    # horizon.
    enforce_availability_registry: bool = True


@dataclass
class SplitConfig:
    """Cross-validation split configuration."""
    method: str = "walk_forward"  # "walk_forward" or "expanding"
    min_train_size: int = 40  # minimum quarters for training (outer)
    test_size: int = 4  # quarters per outer test fold
    horizons: List[int] = field(default_factory=lambda: [1, 2, 4])
    horizon_weights: Dict[int, float] = field(default_factory=lambda: {1: 0.5, 2: 0.3, 4: 0.2})
    # Forecasting strategy: 'direct' fits one model per horizon on
    # horizon-shifted targets (see features.create_direct_horizon_targets);
    # 'recursive' is only run as an explicit sensitivity experiment (spec
    # section 12 prefers direct models for the primary comparison).
    strategy: str = "direct"
    # Nested CV (spec section 11 / bug 3.5): feature selection, PSO tuning,
    # and any other model-selection step must run inside an INNER
    # expanding-window CV built strictly from the OUTER training fold - never
    # touching the outer test fold. These control the inner split.
    inner_min_train_size: int = 24
    inner_test_size: int = 4
    outer_scheme: str = "expanding"
    inner_scheme: str = "expanding"
    strict_temporal_order: bool = True
    # Panel 2 (common period across A1-A4, spec section 9) is much shorter
    # than Panel 1 (bounded by grid-data availability - ~29 quarters vs.
    # ~101 for the full raw-data period) - reusing Panel 1's
    # min_train_size/test_size would produce zero outer folds. These give
    # Panel 2 its own, smaller outer-fold sizing.
    panel2_min_train_size: int = 16
    panel2_test_size: int = 4
    # Panel 2's INNER tuning-fold sizing (used by build_tuning_cv_plan /
    # create_nested_walk_forward_splits when the outer plan being tuned for
    # is itself Panel 2's) - inner_min_train_size=24 would leave too few
    # pre-cutoff rows in a ~29-quarter common period to build even one
    # tuning fold.
    panel2_inner_min_train_size: int = 8
    panel2_inner_test_size: int = 4


@dataclass
class FSConfig:
    """Feature selection configuration."""
    vif_threshold: float = 10.0
    stability_threshold: float = 0.5  # Reduced from 0.6 for low feature counts
    vote_threshold: int = 2  # minimum methods agreeing
    top_k_features: int = 5  # for SHAP concentration
    evaluator_model: str = "lightgbm"  # or "catboost"
    # Linear FS specific thresholds (more lenient for small datasets)
    ridge_coef_threshold: float = 0.005  # Reduced from 0.01 for small datasets
    elasticnet_coef_threshold: float = 1e-8  # Reduced from 1e-10, more lenient
    # Hybrid FS settings
    run_hybrid_fs: bool = True  # Enable filter+wrapper+embedded hybrid
    hybrid_min_votes: int = 2  # Minimum methods agreeing in hybrid
    # Minimum features to select (prevents over-filtering)
    min_features: int = 3
    # The five primary feature-selection strategies (spec section 7). All
    # run strictly inside inner expanding-window folds of the outer training
    # data - see splits/nested_walk_forward.py. FS5 (consensus) is built
    # from the first four using `vote_threshold` as its minimum-votes rule.
    methods: List[str] = field(default_factory=lambda: [
        "linear_stability", "wrapper", "xgboost_shap",
        "permutation_stability", "consensus"
    ])


@dataclass
class OptimizationConfig:
    """Swarm optimization configuration."""
    optimizer: str = "pso"  # "pso" or "gwo"
    n_particles: int = 20
    n_iterations: int = 30
    seed: int = 42
    annual_penalty_threshold: float = 0.05  # 5% MAPE threshold
    annual_penalty_weight: float = 0.1
    # Primary optimisation metric (spec section 13): inner-fold MASE by
    # default rather than raw transformed-scale MAE.
    metric: str = "mase"
    # PSO/GWO must only ever be evaluated on INNER expanding-window folds of
    # the outer training data (bug 3.5 / section 11) - never on the outer
    # test fold being forecast.
    inertia: float = 0.7
    inertia_decay: float = 0.99
    cognitive_coefficient: float = 1.5
    social_coefficient: float = 1.5
    # False (default): the reduced-budget "sweep" mode - one shared
    # single-cutoff tuning plan (build_tuning_cv_plan) reused across every
    # outer fold, as scripts 01-04 already do. True: true per-outer-fold
    # nested retuning (optimize_model_nested/NestedFold, spec section 15/16)
    # - statistically ideal but far more PSO evaluations; reserved for
    # re-running Pareto-shortlisted configurations at full budget with
    # multiple seeds (spec section 16), not the full config x FS x model
    # sweep.
    nested_retuning: bool = False
    seeds_for_stability: List[int] = field(default_factory=lambda: [42, 43, 44])


@dataclass
class ModelConfig:
    """Model training configuration."""
    models: List[str] = field(default_factory=lambda: [
        "ridge", "random_forest", "catboost", "lightgbm", "lstm"
    ])
    lstm_lookback_options: List[int] = field(default_factory=lambda: [4, 8, 12])
    lstm_max_epochs: int = 200
    lstm_patience: int = 20
    lstm_dropout: float = 0.2
    # Use GPU for PSO optimization + final fits where the installed backend
    # supports it (CatBoost, LSTM/PyTorch). Auto-probed and falls back to
    # CPU per-backend if unavailable - see src/core/gpu.py. LightGBM's pip
    # wheel is CPU-only regardless of this flag (no bundled GPU build).
    use_gpu: bool = True


@dataclass
class MCDAConfig:
    """MCDA decision maker configuration."""
    method: str = "vikor"  # "vikor" or "topsis"
    use_pareto_filter: bool = True
    # Weights for FS evaluation criteria
    fs_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.30,
        "stability": 0.20,
        "shap_concentration": 0.15,
        "shap_stability": 0.20,
        "parsimony": 0.15
    })
    # Weights for final model selection
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "quarterly_mae": 0.35,
        "stability": 0.20,
        "annual_consistency": 0.25,
        "interpretability": 0.10,
        "parsimony": 0.10
    })
    vikor_v: float = 0.5  # VIKOR compromise parameter
    # Output top-N options for comparison
    top_n_options: int = 2  # Return top 2 FS options and models for comparison


@dataclass
class EvaluationConfig:
    """Evaluation metric configuration."""
    # MASE is the primary comparative metric (spec section 14) - scaled
    # relative to a seasonal-naive (lag-4) forecast computed from
    # training-only history.
    primary_metric: str = "mase"
    complementary_metrics: List[str] = field(default_factory=lambda: [
        "mae", "rmse", "smape", "r2"
    ])
    seasonal_period: int = 4  # quarterly data -> 4-quarter seasonal-naive
    include_naive_baselines: bool = True


@dataclass
class ConfigurationsConfig:
    """
    Data-configuration architecture (spec section 8): A1 raw, A2
    raw+engineered, A3 raw+grid, A4 raw+engineered+grid. `tags` maps each
    configuration name to the `configuration_membership` tag used in
    config/feature_registry.yaml - see FeatureRegistry.configuration_members.
    """
    names: List[str] = field(default_factory=lambda: [
        'A1_raw', 'A2_engineered', 'A3_grid_fusion', 'A4_full_fusion'
    ])
    tags: Dict[str, str] = field(default_factory=lambda: {
        'A1_raw': 'A1', 'A2_engineered': 'A2',
        'A3_grid_fusion': 'A3', 'A4_full_fusion': 'A4',
    })
    # 'panel2_common_period' is the PRIMARY comparison period for the whole
    # Stream A/B study (revised spec section 14): identical outer folds
    # across ALL of A1-A4/B1-B4, bounded by grid-data availability. See
    # src/splits/panels.py::PRIMARY_PERIOD_PANEL. 'panel1_full_period'
    # (longest valid window, A1/A2 only) is a documented SENSITIVITY-ONLY
    # comparison - excluded from the main global MCDM ranking.
    panels: List[str] = field(default_factory=lambda: ['panel1_full_period', 'panel2_common_period'])


@dataclass
class GridConfig:
    """
    Electricity-grid data fusion configuration (spec section 6). Source:
    GB National Grid ESO Carbon Intensity API (api.carbonintensity.org.uk) -
    public, keyless, half-hourly since ~2017-06. See src/grid/fetch.py
    (caching) and src/grid/aggregate.py (quarterly aggregation).
    """
    enabled: bool = True
    cache_dir: str = "data/raw/grid_cache"
    # Minimum fraction of expected half-hourly observations a quarter must
    # have before its aggregated grid features are trusted (spec 6.3).
    minimum_quarter_completeness: float = 0.95
    include_generation_mix: bool = True
    # The Carbon Intensity API exposes generation-mix *shares*, not absolute
    # demand in MW - true demand-weighted CI (spec 6.1) requires a separate
    # demand data source and is best-effort only (see Phase 3 notes in the
    # project plan / final report). Defaults to False until/unless such a
    # source is wired in, rather than silently fabricating a proxy.
    use_demand_weighted_intensity: bool = False
    # Fixed physical threshold (gCO2/kWh) for HighShare_q (spec 6.2).
    high_intensity_threshold_gco2_per_kwh: float = 300.0
    # Training-relative threshold percentile for ExtremeShare_q (spec 6.2) -
    # must be estimated only from the current training data when used.
    extreme_intensity_percentile: float = 90.0


@dataclass
class MobilityConfig:
    """
    Google COVID-19 mobility-shock data fusion configuration (spec section
    4). Source: Google's Global Mobility Report CSV (gstatic.com), public,
    keyless, daily national coverage ~2020-02-15 to mid-October 2022. See
    src/mobility/fetch.py (caching) and src/mobility/aggregate.py
    (quarterly aggregation + neutral-zero convention outside the window).
    """
    enabled: bool = True
    cache_dir: str = "data/raw/mobility"
    # Minimum fraction of expected daily observations (averaged across the
    # six categories) a quarter must have before its aggregated mobility
    # features are trusted (spec 4.3).
    minimum_quarter_completeness: float = 0.80
    # Last quarter eligible for the PRIMARY aggregation - the official data
    # end mid-way through 2022Q4, so partial 2022Q4 is a sensitivity-only
    # observation, not a primary one (spec 4.3).
    primary_cutoff_quarter: str = "2022Q3"
    # Sensitivity-only: include the partial 2022Q4 quarter in the primary
    # series instead of neutral-zeroing it (spec 4.3/29's mandatory
    # mobility sensitivity runs).
    include_partial_2022q4: bool = False
    # Mandatory sensitivity run (spec 4.4/29): exclude the mobility block
    # entirely, rather than only via ad-hoc registry membership editing.
    exclude_mobility_block: bool = False


@dataclass
class OwidConfig:
    """
    OWID annual renewable/low-carbon electricity share configuration (spec
    section 8). Source: Our World in Data grapher CSV exports
    (ourworldindata.org), public, keyless, annual. See src/owid/fetch.py
    (caching) and src/owid/lag.py (one-year-lag alignment to quarters).
    Independently-sourced structural predictor + external validation series
    for the primary NESO-derived Grid_RenewableShare/LowCarbonShare - not a
    replacement for them.
    """
    enabled: bool = True
    cache_dir: str = "data/raw/owid"


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: str = "outputs"
    figure_dpi: int = 300
    figure_dpi_high_res: int = 600
    figure_format: str = "png"
    figure_vector_formats: List[str] = field(default_factory=lambda: ["svg", "pdf"])
    save_models: bool = True
    save_predictions: bool = True


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    fs: FSConfig = field(default_factory=FSConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mcda: MCDAConfig = field(default_factory=MCDAConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    configurations: ConfigurationsConfig = field(default_factory=ConfigurationsConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)
    owid: OwidConfig = field(default_factory=OwidConfig)
    seed: int = 42
    run_id: Optional[str] = None
    feature_registry_path: str = "config/feature_registry.yaml"

    def __post_init__(self):
        if self.run_id is None:
            self.run_id = datetime.now().strftime("run_%Y%m%d_%H%M")
        # Process-wide GPU toggle - see src/core/gpu.py for why this is an
        # env var rather than threaded through ModelRegistry.create().
        from .gpu import set_gpu_requested
        set_gpu_requested(self.model.use_gpu)

    @property
    def run_dir(self) -> Path:
        return Path(self.output.base_dir) / "runs" / self.run_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None):
        if path is None:
            path = self.run_dir / "configs_snapshot" / "config.yaml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            data=DataConfig(**data.get('data', {})),
            features=FeatureConfig(**data.get('features', {})),
            splits=SplitConfig(**data.get('splits', {})),
            fs=FSConfig(**data.get('fs', {})),
            optimization=OptimizationConfig(**data.get('optimization', {})),
            model=ModelConfig(**data.get('model', {})),
            mcda=MCDAConfig(**data.get('mcda', {})),
            evaluation=EvaluationConfig(**data.get('evaluation', {})),
            output=OutputConfig(**data.get('output', {})),
            configurations=ConfigurationsConfig(**data.get('configurations', {})),
            grid=GridConfig(**data.get('grid', {})),
            mobility=MobilityConfig(**data.get('mobility', {})),
            owid=OwidConfig(**data.get('owid', {})),
            seed=data.get('seed', 42),
            run_id=data.get('run_id'),
            feature_registry_path=data.get('feature_registry_path', 'config/feature_registry.yaml')
        )


def create_run_directories(config: Config) -> Dict[str, Path]:
    """
    Create all output directories for a run, per spec section 30's tree.

    Beyond the original flat set (logs/tables/metrics/predictions/figures/
    models/configs_snapshot), adds the Stream A/B-aware subtree needed from
    Phase 6 (fold_predictions/stream_{A,B}, metrics/stream_{A,B},
    metrics/global_comparison) through Phase 8/10/11 (pareto_mcda/
    interpretability/figures/reports subtrees) - built once here so no
    later phase needs to duplicate directory-creation logic.
    """
    run_dir = config.run_dir
    dirs = {
        'root': run_dir,
        'logs': run_dir / 'logs',
        'tables': run_dir / 'tables',
        'metrics': run_dir / 'metrics',
        'predictions': run_dir / 'predictions',
        'figures': run_dir / 'figures',
        'models': run_dir / 'models',
        'configs_snapshot': run_dir / 'configs_snapshot',
        # spec section 30 tree
        'audit': run_dir / 'audit',
        'source_metadata': run_dir / 'source_metadata',
        'data_quality': run_dir / 'data_quality',
        'data_quality_mobility': run_dir / 'data_quality' / 'mobility',
        'data_quality_grid': run_dir / 'data_quality' / 'grid',
        'data_quality_owid': run_dir / 'data_quality' / 'owid',
        'feature_registry': run_dir / 'feature_registry',
        'selected_features': run_dir / 'selected_features',
        'fold_predictions': run_dir / 'fold_predictions',
        'fold_predictions_stream_A': run_dir / 'fold_predictions' / 'stream_A',
        'fold_predictions_stream_B': run_dir / 'fold_predictions' / 'stream_B',
        'metrics_stream_A': run_dir / 'metrics' / 'stream_A',
        'metrics_stream_B': run_dir / 'metrics' / 'stream_B',
        'metrics_global_comparison': run_dir / 'metrics' / 'global_comparison',
        'statistical_tests': run_dir / 'statistical_tests',
        'pareto_mcda_stream_A': run_dir / 'pareto_mcda' / 'stream_A',
        'pareto_mcda_stream_B': run_dir / 'pareto_mcda' / 'stream_B',
        'pareto_mcda_final': run_dir / 'pareto_mcda' / 'final',
        'interpretability_best_A': run_dir / 'interpretability' / 'best_A',
        'interpretability_best_B': run_dir / 'interpretability' / 'best_B',
        'interpretability_best_overall': run_dir / 'interpretability' / 'best_overall',
        'interpretability_regimes': run_dir / 'interpretability' / 'regimes',
        'figures_pdf_main': run_dir / 'figures' / 'pdf' / 'main',
        'figures_pdf_supplementary': run_dir / 'figures' / 'pdf' / 'supplementary',
        'figures_pdf_forecasting': run_dir / 'figures' / 'pdf' / 'forecasting',
        'figures_plot_data': run_dir / 'figures' / 'plot_data',
        'reports': run_dir / 'reports',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_latest_run_id(base_dir: str = "outputs") -> Optional[str]:
    """Find the most recent run_id by sorting run directories."""
    runs_dir = Path(base_dir) / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.name,
        reverse=True,
    )
    if run_dirs:
        return run_dirs[0].name
    return None


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()
