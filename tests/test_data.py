from loss_prevention.data.generator import (
    FEATURE_COLUMNS,
    generate_customers,
    generate_historical_assignments,
    true_cate,
)
from loss_prevention.schemas import InterventionType


def test_generate_customers_shape():
    customers = generate_customers(100, seed=1)
    assert len(customers) == 100
    c = customers[0]
    assert c.customer_id.startswith("C")
    assert 0 <= c.hazard_exposure <= 1
    assert c.true_baseline_risk is not None


def test_historical_has_treatment_and_control():
    customers = generate_customers(200, seed=2)
    hist = generate_historical_assignments(customers, n=500, seed=2)
    treated = sum(1 for a in hist if a.treated)
    assert 50 < treated < 450
    assert all(set(a.features) == set(FEATURE_COLUMNS) for a in hist[:5])


def test_true_cate_no_intervention_zero():
    customers = generate_customers(20, seed=3)
    feats = {k: float(getattr(customers[0], k)) for k in FEATURE_COLUMNS}
    assert true_cate(InterventionType.NO_INTERVENTION, feats) == 0.0


def test_confounding_treated_higher_hazard():
    customers = generate_customers(800, seed=4)
    hist = generate_historical_assignments(
        customers, n=2000, seed=4, randomized_fraction=0.0
    )
    # Focus on inspection which is strongly selected on hazard
    insp = [a for a in hist if a.intervention == InterventionType.INSPECTION]
    t_haz = [a.features["hazard_exposure"] for a in insp if a.treated]
    c_haz = [a.features["hazard_exposure"] for a in insp if not a.treated]
    assert sum(t_haz) / len(t_haz) > sum(c_haz) / len(c_haz)
