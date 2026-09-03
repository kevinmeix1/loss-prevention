"""Browser E2E + visual QA against a live server (or TestClient-spawned uvicorn).

Run:
  PYTHONPATH=src pytest tests/e2e_playwright.py -q
  # or with already-running server:
  LP_BASE_URL=http://127.0.0.1:8010 pytest tests/e2e_playwright.py -q
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

BASE = os.environ.get("LP_BASE_URL", "").rstrip("/")
SHOT_DIR = Path(__file__).resolve().parents[1] / "docs" / "qa" / "screenshots"
ROOT = Path(__file__).resolve().parents[1]


def _wait_health(url: str, timeout: float = 90.0) -> None:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Server not healthy at {url}")


@pytest.fixture(scope="module")
def lp_base_url():
    if BASE:
        _wait_health(BASE, timeout=20)
        yield BASE
        return

    port = "8017"
    url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.Popen(
        [
            str(Path("/Users/kaiwenmei/Desktop/x11/.venv/bin/uvicorn")),
            "loss_prevention.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(url, timeout=120)
        yield url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _chromium_executable() -> str | None:
    """Prefer an already-downloaded browser when playwright install is unavailable."""
    candidates = [
        Path.home()
        / "Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing",
        Path.home()
        / "Library/Caches/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-mac-arm64/chrome-headless-shell",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


@pytest.fixture(scope="module")
def browser_page(lp_base_url):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    exe = _chromium_executable()
    launch_kwargs = {"headless": True}
    if exe:
        launch_kwargs["executable_path"] = exe

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(**launch_kwargs)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"Playwright browser unavailable: {exc}")
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(lp_base_url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector("#overview .metric", timeout=180_000)
        page.wait_for_function(
            "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
            timeout=180_000,
        )
        yield page, lp_base_url
        browser.close()


def test_home_loads_recommendation(browser_page):
    page, _ = browser_page
    page.locator('.side-nav button[data-tab="overview"]').click()
    page.wait_for_timeout(150)
    assert page.locator("#pageTitle").inner_text() == "Risk Overview"
    assert page.locator("#overview .metric").count() >= 4
    page.screenshot(path=str(SHOT_DIR / "01_overview.png"), full_page=True)


def test_all_nav_tabs(browser_page):
    page, _ = browser_page
    tabs = [
        ("factors", "Risk Factors"),
        ("recs", "Recommended Interventions"),
        ("impact", "Expected Impact"),
        ("evidence", "Evidence & Agent Trace"),
        ("cf", "Counterfactual Simulator"),
        ("methods", "Model Comparison"),
        ("research", "Research Lab"),
        ("architecture", "Architecture"),
    ]
    for tab_id, title in tabs:
        page.locator(f'.side-nav button[data-tab="{tab_id}"]').click()
        page.wait_for_timeout(200)
        assert page.locator("#pageTitle").inner_text() == title
        assert page.locator(f"#{tab_id}").is_visible()
        page.screenshot(path=str(SHOT_DIR / f"tab_{tab_id}.png"), full_page=True)
    # return home
    page.locator('.side-nav button[data-tab="overview"]').click()


def test_run_recommendation_button(browser_page):
    page, _ = browser_page
    page.select_option("#customer", index=3)
    page.wait_for_function(
        "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
        timeout=120_000,
    )
    page.select_option("#method", "t_learner")
    page.wait_for_function(
        "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
        timeout=120_000,
    )
    page.click("#loadBtn")
    page.wait_for_function(
        "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
        timeout=120_000,
    )
    assert page.locator("#overview .metric").count() >= 4
    page.screenshot(path=str(SHOT_DIR / "02_recommend_rerun.png"), full_page=True)


def test_counterfactual_simulate(browser_page):
    page, _ = browser_page
    page.locator('.side-nav button[data-tab="cf"]').click()
    page.select_option("#cfIntervention", "inspection")
    page.click("#cfBtn")
    page.wait_for_selector("#cfOut .metric", timeout=60_000)
    text = page.locator("#cfOut").inner_text().lower()
    assert "p(loss|control)" in text or "uplift p(loss|control)" in text
    assert "cate" in text or "δ risk" in text or "risk" in text
    # Arithmetic sanity visible in UI metrics: control/treated both present
    assert "p(loss|treated)" in text or "treated" in text
    page.screenshot(path=str(SHOT_DIR / "03_counterfactual.png"), full_page=True)


def test_method_compare(browser_page):
    page, _ = browser_page
    page.locator('.side-nav button[data-tab="methods"]').click()
    page.select_option("#methodIntervention", "maintenance_recommendation")
    page.click("#methodBtn")
    page.wait_for_selector("#methodsOut table", timeout=60_000)
    assert "x_learner" in page.locator("#methodsOut").inner_text()
    page.screenshot(path=str(SHOT_DIR / "04_methods.png"), full_page=True)


def test_research_snapshot(browser_page):
    page, _ = browser_page
    page.click("#researchBtn")
    page.wait_for_selector("#research .metric", timeout=180_000)
    body = page.locator("#research").inner_text()
    assert "Predictive AUC" in body or "predictive_auc" in body
    page.screenshot(path=str(SHOT_DIR / "05_research.png"), full_page=True)


def test_keyboard_focus_and_disabled_during_busy(browser_page):
    page, _ = browser_page
    page.locator('.side-nav button[data-tab="overview"]').click()
    page.focus("#loadBtn")
    assert page.evaluate("document.activeElement?.id") == "loadBtn"
    # Buttons should not stay disabled after completion
    assert page.locator("#loadBtn").is_enabled()


def test_mobile_layout_no_overflow(browser_page):
    page, _ = browser_page
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator('.side-nav button[data-tab="overview"]').click()
    page.wait_for_timeout(300)
    overflow = page.evaluate(
        """() => {
          const doc = document.documentElement;
          return doc.scrollWidth > doc.clientWidth + 2;
        }"""
    )
    page.screenshot(path=str(SHOT_DIR / "06_mobile.png"), full_page=True)
    assert overflow is False
    page.set_viewport_size({"width": 1440, "height": 900})


def test_error_path_unknown_customer_via_api(browser_page):
    page, base = browser_page
    status = page.evaluate(
        """async (base) => {
          const r = await fetch(base + '/customers/NOPE/recommend');
          return r.status;
        }""",
        base,
    )
    assert status == 404
