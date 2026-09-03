# Final QA Report — Loss Prevention Agent

Date: 2026-08-10  
Scope: production UI polish, interactive audit, automated tests, PDF, demo video.

## Application launch

| Check | Result |
|---|---|
| `GET /health` | PASS — `{"status":"ok"}` |
| UI at `/` | PASS — serves static production shell |
| OpenAPI `/docs` | PASS |

## Frontend interactive audit

| Control | Expected | Verified |
|---|---|---|
| Customer select | Auto-runs recommendation | PASS (Playwright) |
| Uplift method select | Auto-runs recommendation | PASS |
| Run recommendation | Reloads all panels + agent_trace | PASS |
| Research snapshot | Loads deploy + experiments | PASS |
| Side-nav tabs (9) | Switch panels, update title | PASS |
| Counterfactual Simulate | Consistent uplift P0/P1/CATE | PASS (after bugfix) |
| Method Compare | S/T/X (+ CF) table + chart | PASS |
| Alt+1…9 | Keyboard tab switch | Implemented |
| Mobile 390px | No horizontal overflow | PASS (after CSS fix) |
| Focus on primary button | Keyboard focusable | PASS |
| 404 unknown customer | API returns 404 | PASS |

## Bug found & fixed during QA

**Counterfactual metric inconsistency**  
Symptom: UI showed predictive `P(loss)=35.5%` as “baseline” and uplift treated `13.5%` as counterfactual, while Δ risk showed uplift CATE `0.6%` — arithmetic did not close.  
Fix: counterfactual baseline/treated/delta now all come from the uplift model; predictive score exposed separately as `p_loss_predictive`.  
Regression test: `test_counterfactual` asserts `P0 − P1 == CATE`.

## Visual QA

Screenshots under `docs/qa/screenshots/` (Playwright, 1440×900 + mobile):

- Overview, factors, interventions, impact, evidence, counterfactual, methods, research, architecture
- No clipped primary CTAs; pipeline stages readable; toast + status pill visible

## Accessibility (practical)

| Check | Result |
|---|---|
| Skip link | Present |
| Tab roles / aria-selected | Present on side-nav |
| Labels on selects | Present |
| Focus-visible styles | Present |
| Status live region | `#statusPill` / toast `aria-live` |
| Color-only meaning | Mitigated with badges + text |

NOT VERIFIED: full screen-reader walkthrough with VoiceOver.

## Performance notes

| Observation | Action |
|---|---|
| Research snapshot retrains/simulates (~tens of seconds) | Status pill + spinner; parallel fetch |
| Recommendation path is local sklearn inference | Acceptable for demo |
| No unnecessary LLM calls by default | By design |

## PDF QA

| Check | Result |
|---|---|
| Generated | `docs/Loss_Prevention_Technical_Tutorial_EN_ZH.pdf` (16 pages) |
| CJK present | PASS (text extraction) |
| Glossary / architecture / CATE / limitations | PASS |
| Diagrams present | PASS (`docs/tutorial/assets/`) |
| Claims match implementation | PASS — no external DB/auth claimed as implemented |

## Demo video QA

| Check | Result |
|---|---|
| Script | `docs/DEMO_SCRIPT.md` |
| Video | `docs/demo/Loss_Prevention_Demo.mp4` |
| Source frames | Live UI screenshots (not mocked charts) |

## Residual limitations (honest)

- Synthetic data only  
- Causal forest is educational-scale  
- Optional LLM narrative path requires `OPENAI_API_KEY` (not required for core)  
- No auth / persistence layer  
- Playwright browser download may need local Chromium cache on fresh machines  
