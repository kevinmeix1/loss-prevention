"""Synthetic data generation for loss-prevention experiments."""

from loss_prevention.data.generator import (
    FEATURE_COLUMNS,
    generate_customers,
    generate_experiment_dataset,
    generate_historical_assignments,
)

__all__ = [
    "FEATURE_COLUMNS",
    "generate_customers",
    "generate_historical_assignments",
    "generate_experiment_dataset",
]
