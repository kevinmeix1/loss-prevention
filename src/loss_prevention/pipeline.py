"""End-to-end loss prevention recommendation pipeline."""

from __future__ import annotations

from pathlib import Path

from loss_prevention.agent.recommender import LossPreventionAgent
from loss_prevention.data.generator import generate_customers, generate_experiment_dataset
from loss_prevention.models.bundle import ModelBundle, train_all_models
from loss_prevention.ranking.interventions import RankingConfig
from loss_prevention.schemas import (
    CounterfactualResult,
    CustomerContext,
    InterventionType,
    Recommendation,
)


class LossPreventionPipeline:
    def __init__(
        self,
        bundle: ModelBundle | None = None,
        customers: list[CustomerContext] | None = None,
        artifacts_dir: Path = Path("artifacts/models"),
    ):
        self.artifacts_dir = artifacts_dir
        self.bundle = bundle
        self.customers = customers or []
        self._agent: LossPreventionAgent | None = None

    def ensure_ready(self, seed: int = 42, n_customers: int = 2000, n_historical: int = 6000) -> None:
        if self.bundle is None:
            path = self.artifacts_dir / "model_bundle.joblib"
            if path.exists():
                self.bundle = ModelBundle.load(self.artifacts_dir)
            else:
                self.bundle = train_all_models(
                    n_customers=n_customers, n_historical=n_historical, seed=seed
                )
                self.bundle.save(self.artifacts_dir)
        if not self.customers:
            self.customers = generate_customers(n_customers, seed=seed)
        self._agent = LossPreventionAgent(self.bundle, self.customers)

    @property
    def agent(self) -> LossPreventionAgent:
        if self._agent is None:
            self.ensure_ready()
        assert self._agent is not None
        return self._agent

    def recommend(self, customer_id: str | None = None, method: str = "x_learner") -> Recommendation:
        self.ensure_ready()
        if customer_id is None:
            customer_id = self.customers[0].customer_id
        return self.agent.recommend(customer_id, method=method)

    def counterfactual(
        self, customer_id: str, intervention: InterventionType, method: str = "x_learner"
    ) -> CounterfactualResult:
        self.ensure_ready()
        raw = self.agent.tools.simulate_intervention(
            customer_id, intervention.value, method=method
        )
        return CounterfactualResult(**raw)

    def get_customer(self, customer_id: str) -> CustomerContext:
        self.ensure_ready()
        return self.agent.customers[customer_id]


def bootstrap_demo(seed: int = 42) -> LossPreventionPipeline:
    customers, _ = generate_experiment_dataset(n_customers=1500, n_historical=5000, seed=seed)
    bundle = train_all_models(n_customers=1500, n_historical=5000, seed=seed)
    return LossPreventionPipeline(bundle=bundle, customers=customers)
