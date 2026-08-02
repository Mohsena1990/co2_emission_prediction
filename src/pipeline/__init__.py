"""
Experiment orchestration module (spec sections 12/15): runs the
config x FS x model x horizon grid end-to-end and produces fold-level,
machine-readable results every table/figure is generated from.
"""
from .experiment import run_configuration_model, run_naive_baseline, ExperimentCell

__all__ = ['run_configuration_model', 'run_naive_baseline', 'ExperimentCell']
