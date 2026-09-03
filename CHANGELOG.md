# Changelog

## 1.1.0 — 2026-08-10 (portfolio polish)

### UI
- Production shell with design system (`app.css` / `app.js`): sidebar navigation, pipeline stages, gauges, CATE charts, agent timeline, toasts, loading/empty/error states
- Added Model Compare + Architecture views
- Auto-run recommendation on customer/method change
- Keyboard shortcuts Alt+1…9
- Mobile overflow fixes

### Correctness
- **Fix:** Counterfactual simulator now uses uplift-consistent `P0/P1/CATE` arithmetic; predictive score exposed separately as `p_loss_predictive`

### Testing
- Extended API / agent / frontend asset tests
- Playwright browser E2E with visual screenshots under `docs/qa/screenshots/`
- Regression for counterfactual arithmetic consistency

### Documentation & demos
- Bilingual technical PDF (`docs/Loss_Prevention_Technical_Tutorial_EN_ZH.pdf`)
- Glossary, QA report, test report, demo script
- Demo video regenerated from live UI screenshots

## 1.0.0 — 2026-08-10 (initial system)

- Synthetic DGP with confounding + RCT slice
- Predictive risk + S/T/X uplift + causal forest
- Constraint-aware ranking + grounded agent
- FastAPI + CLI + research experiments
