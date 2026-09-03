#!/usr/bin/env python3
"""Generate a professional demo video from live UI screenshots + captions.

Prefer Playwright QA screenshots when present; otherwise render matplotlib scenes
from live API responses.

Usage:
  # with server running and screenshots already captured:
  python scripts/generate_demo_video.py

  # also refresh screenshots via Playwright (optional):
  LP_BASE_URL=http://127.0.0.1:8010 python scripts/generate_demo_video.py --capture
"""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOT_DIR = ROOT / "docs" / "qa" / "screenshots"
FRAME_DIR = ROOT / "docs" / "demo" / "frames"
OUT_VIDEO = ROOT / "docs" / "demo" / "Loss_Prevention_Demo.mp4"
BASE = "http://127.0.0.1:8010"

# Scene order for the product demo
SCENES = [
    ("01_overview.png", "Opening — Risk Overview", 4.5),
    ("tab_architecture.png", "Architecture pipeline", 3.5),
    ("02_recommend_rerun.png", "Run recommendation for selected customer", 4.0),
    ("tab_factors.png", "Risk factors explain baseline risk", 3.5),
    ("tab_recs.png", "Constraint-aware intervention ranking", 4.0),
    ("03_counterfactual.png", "What-if counterfactual simulator", 4.0),
    ("04_methods.png", "S/T/X-learner & causal forest compare", 4.0),
    ("05_research.png", "Research lab — bias & policy value", 4.5),
    ("01_overview.png", "Closing — prediction ≠ uplift", 3.5),
]


def get(path: str):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return json.loads(r.read().decode())


def capture_screenshots() -> None:
    from playwright.sync_api import sync_playwright

    exe = (
        Path.home()
        / "Library/Caches/ms-playwright/chromium-1228/chrome-mac-arm64/"
        "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    )
    SHOT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=str(exe), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector("#overview .metric", timeout=180_000)
        page.wait_for_function(
            "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
            timeout=180_000,
        )
        page.screenshot(path=str(SHOT_DIR / "01_overview.png"), full_page=True)
        for tab in [
            "architecture",
            "factors",
            "recs",
            "impact",
            "evidence",
            "cf",
            "methods",
            "research",
        ]:
            page.locator(f'.side-nav button[data-tab="{tab}"]').click()
            page.wait_for_timeout(250)
            page.screenshot(path=str(SHOT_DIR / f"tab_{tab}.png"), full_page=True)
        page.locator('.side-nav button[data-tab="overview"]').click()
        page.select_option("#customer", index=3)
        page.wait_for_function(
            "() => document.querySelector('#statusPill')?.textContent?.includes('Ready')",
            timeout=120_000,
        )
        page.screenshot(path=str(SHOT_DIR / "02_recommend_rerun.png"), full_page=True)
        page.locator('.side-nav button[data-tab="cf"]').click()
        page.select_option("#cfIntervention", "inspection")
        page.click("#cfBtn")
        page.wait_for_selector("#cfOut .metric", timeout=60_000)
        page.screenshot(path=str(SHOT_DIR / "03_counterfactual.png"), full_page=True)
        page.locator('.side-nav button[data-tab="methods"]').click()
        page.select_option("#methodIntervention", "maintenance_recommendation")
        page.click("#methodBtn")
        page.wait_for_selector("#methodsOut table", timeout=60_000)
        page.screenshot(path=str(SHOT_DIR / "04_methods.png"), full_page=True)
        page.click("#researchBtn")
        page.wait_for_selector("#research .metric", timeout=180_000)
        page.screenshot(path=str(SHOT_DIR / "05_research.png"), full_page=True)
        browser.close()


def caption_frame(src: Path, title: str, dest: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg

    img = mpimg.imread(src)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.imshow(img)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
    fig.patch.set_facecolor("#17352c")
    fig.text(
        0.02,
        0.96,
        title,
        color="white",
        fontsize=16,
        fontweight="bold",
        va="center",
        ha="left",
    )
    fig.text(
        0.98,
        0.96,
        "Loss Prevention Agent",
        color="#b7cdc3",
        fontsize=11,
        va="center",
        ha="right",
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_video(frames: list[tuple[Path, float]]) -> Path:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    list_path = FRAME_DIR / "ffmpeg_list.txt"
    lines = []
    for path, dur in frames:
        lines.append(f"file '{path.resolve()}'")
        lines.append(f"duration {dur}")
    # ffmpeg concat demuxer needs last file repeated
    lines.append(f"file '{frames[-1][0].resolve()}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(OUT_VIDEO),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return OUT_VIDEO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true", help="Refresh Playwright screenshots")
    args = parser.parse_args()
    if args.capture:
        capture_screenshots()

    missing = [name for name, _, _ in SCENES if not (SHOT_DIR / name).exists()]
    if missing:
        raise SystemExit(
            f"Missing screenshots: {missing}. Run Playwright E2E or pass --capture."
        )

    # Sanity: API should be alive for authenticity claim in metadata
    health = get("/health")
    assert health.get("status") == "ok"

    stamped: list[tuple[Path, float]] = []
    for i, (name, title, dur) in enumerate(SCENES, start=1):
        dest = FRAME_DIR / f"{i:02d}_{Path(name).stem}.png"
        caption_frame(SHOT_DIR / name, title, dest)
        stamped.append((dest, dur))

    out = build_video(stamped)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
