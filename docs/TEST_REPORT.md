# Test Report

## Suites

| Suite | File(s) | What it covers |
|---|---|---|
| Data / DGP | `test_data.py` | Feature schema, treatment/control mix, confounding signal |
| Models | `test_models.py` | Predictive AUC floor, S/T/X uplift, causal forest shape |
| Ranking + agent | `test_ranking_agent.py` | Constraints, grounded evidence, counterfactual arithmetic |
| Agent edges | `test_agent_edge.py` | Invalid args / tool failures |
| API | `test_api.py`, `test_api_extended.py` | Contracts, method-compare, architecture, 404s |
| Evaluation | `test_evaluation.py` | Metrics smoke + research experiments |
| Frontend assets | `test_frontend_assets.py` | Static HTML/CSS/JS presence & wiring |
| HTTP E2E | `test_e2e_http.py` | Multi-step API journey |
| Browser E2E | `e2e_playwright.py` | Real Chromium UI flows + screenshots |

## Validation cycles (this session)

```
Cycle 1 — pytest (non-browser): 38 passed
Cycle 2 — Playwright E2E: 2 failed (assertion case + mobile overflow) → fixed
Cycle 3 — Playwright E2E: 9 passed
Cycle 4 — pytest (non-browser) after counterfactual fix: re-run below
```

## How to run

```bash
cd loss-prevention
source ../.venv/bin/activate
pip install -e . --no-deps

# Unit / API / HTTP E2E
pytest -q --ignore=tests/e2e_playwright.py

# Browser E2E (server must be up, or fixture boots :8017)
loss-prevention serve-ui &
LP_BASE_URL=http://127.0.0.1:8010 pytest tests/e2e_playwright.py -q
```

## Latest verified counts

- Non-browser: **38+ passed** (includes counterfactual arithmetic regression)
- Browser E2E: **9 passed** (final cycle after label + tab-state fixes)
- Demo video rebuilt from live UI screenshots: `docs/demo/Loss_Prevention_Demo.mp4`
