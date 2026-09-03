"""Frontend asset and interaction contract tests (no browser required)."""

from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "loss_prevention" / "ui" / "static"


def test_static_files_exist():
    assert (STATIC / "index.html").exists()
    assert (STATIC / "app.css").exists()
    assert (STATIC / "app.js").exists()


def test_html_has_required_controls():
    html = (STATIC / "index.html").read_text()
    for token in [
        'id="customer"',
        'id="method"',
        'id="loadBtn"',
        'id="cfBtn"',
        'id="methodBtn"',
        'id="researchBtn"',
        'data-tab="overview"',
        'data-tab="cf"',
        'data-tab="research"',
        'data-tab="architecture"',
        'href="/assets/app.css"',
        'src="/assets/app.js"',
        'role="tablist"',
    ]:
        assert token in html, f"missing {token}"


def test_css_design_tokens():
    css = (STATIC / "app.css").read_text()
    for token in ["--accent", "--radius", ".btn-primary", ".btn:disabled", ".loading", ".empty", ".toast", ":focus-visible"]:
        assert token in css, f"missing {token}"


def test_js_has_error_and_loading_paths():
    js = (STATIC / "app.js").read_text()
    for token in [
        "setBusy",
        "toast",
        "loadRecommendation",
        "runCounterfactual",
        "loadMethodCompare",
        "runResearch",
        "alert-danger",
        "agent_trace",
        "window.LPApp",
    ]:
        assert token in js, f"missing {token}"
