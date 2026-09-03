# Autonomous Insurance Loss Prevention Recommendation Agent

Research-engineering system that answers:

> Given a customer / policy / property context, **what intervention is most likely to reduce future risk?**

It demonstrates predictive ML, uplift / causal modeling, intervention ranking, tool-using agents, explainability, synthetic experimentation, and responsible AI — on **entirely synthetic data**.

```
Customer → Risk Model → Risk Factors → Candidate Interventions
        → Uplift / Causal Model → Expected Impact → Constraint Filtering
        → Recommendation Ranking → AI Explanation → Customer Outcome
```

## Links

| Resource | Path |
|---|---|
| **Live Demo (local)** | [http://127.0.0.1:8010/](http://127.0.0.1:8010/) |
| **Demo Video** | [docs/demo/Loss_Prevention_Demo.mp4](docs/demo/Loss_Prevention_Demo.mp4) |
| **Demo Script** | [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) |
| **Technical PDF (EN + 中文)** | [docs/Loss_Prevention_Technical_Tutorial_EN_ZH.pdf](docs/Loss_Prevention_Technical_Tutorial_EN_ZH.pdf) |
| **Architecture** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **Uplift vs Prediction** | [docs/UPLIFT_VS_PREDICTION.md](docs/UPLIFT_VS_PREDICTION.md) |
| **Research Results** | [docs/RESEARCH.md](docs/RESEARCH.md) |
| **Glossary** | [docs/GLOSSARY.md](docs/GLOSSARY.md) |
| **Testing** | [docs/TEST_REPORT.md](docs/TEST_REPORT.md) |
| **QA Report** | [docs/QA_REPORT.md](docs/QA_REPORT.md) |
| **Changelog** | [CHANGELOG.md](CHANGELOG.md) |
| **OpenAPI** | [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs) |

## Core distinction

| Question | Model |
|---|---|
| Who is at risk? | Predictive: `P(loss \| X)` |
| Who will benefit from intervention T? | Uplift: `P(loss \| X, T=0) − P(loss \| X, T=1)` |

**Predicting who is at risk is not the same as predicting who will benefit from an intervention.**

## Quick start

```bash
cd loss-prevention
source ../.venv/bin/activate
pip install -e . --no-deps
# optional UI extras: pip install streamlit plotly playwright

loss-prevention train
loss-prevention serve-ui    # http://127.0.0.1:8010/
pytest -q --ignore=tests/e2e_playwright.py
LP_BASE_URL=http://127.0.0.1:8010 pytest tests/e2e_playwright.py -q
```

Other CLI commands: `recommend`, `counterfactual`, `evaluate`, `research`.

## UI

Production dashboard (not a placeholder):

1. Risk Overview — gauge + CATE ranking  
2. Risk Factors — baseline importances  
3. Interventions — constrained utility ranking  
4. Expected Impact — uplift-consistent P0/P1/CATE  
5. Evidence & Agent — grounded bullets + tool timeline  
6. Counterfactual — “What if intervention X?”  
7. Model Compare — S/T/X + causal forest  
8. Research Lab — deployment metrics + bias experiments  
9. Architecture — pipeline / tools / responsible-AI guards  

## Architecture modules

| Stage | Module | Role |
|---|---|---|
| Synthetic DGP | `data/generator.py` | Customers, RCT + observational assignments, heterogeneous CATEs, confounding |
| Predictive risk | `models/predictive.py` | Calibrated `P(future loss)` |
| Uplift meta-learners | `models/uplift.py` | S-learner, T-learner, X-learner |
| Causal forest | `models/causal_forest.py` | Honest forest CATE estimator |
| Ranking | `ranking/interventions.py` | Benefit, confidence, cost, burden, feasibility |
| Agent | `agent/` | Tool-calling recommender; **cannot invent treatment effects** |
| Explain | `explain/explainer.py` | Grounded bullets + trade-offs |
| Evaluation | `evaluation/` | AUC/Brier/ECE, Qini/AUUC, deployment sim, research experiments |
| API / UI | `api/app.py`, `ui/static/` | FastAPI + production static UI |

## Research experiments

- Selection bias / confounding  
- Treatment heterogeneity  
- Policy optimization (risk-only vs uplift)  
- Explore vs exploit  

## Responsible AI

- No protected characteristics in features or decisions  
- Hard eligibility / burden constraints the agent cannot override  
- Explanations cite only model outputs and observed features  
- Explicit prediction-vs-uplift separation in counterfactuals  

## Rebuild PDF / video

```bash
PYTHONPATH=src python docs/tutorial/generate_technical_pdf.py
# server must be running; screenshots from Playwright preferred
python scripts/generate_demo_video.py
```
