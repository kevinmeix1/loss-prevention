"""Shared fixtures with small synthetic datasets for fast tests."""

from __future__ import annotations

import pytest

from loss_prevention.data.generator import generate_customers, generate_experiment_dataset
from loss_prevention.models.bundle import train_all_models
from loss_prevention.pipeline import LossPreventionPipeline


@pytest.fixture(scope="session")
def small_bundle():
    return train_all_models(
        n_customers=600,
        n_historical=1800,
        seed=7,
        fit_causal_forest=True,
    )


@pytest.fixture(scope="session")
def small_customers():
    return generate_customers(600, seed=7)


@pytest.fixture(scope="session")
def pipeline(small_bundle, small_customers):
    return LossPreventionPipeline(bundle=small_bundle, customers=small_customers)


@pytest.fixture(scope="session")
def hist_df():
    _, df = generate_experiment_dataset(n_customers=500, n_historical=1200, seed=3)
    return df
