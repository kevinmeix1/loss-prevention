"""Generate architecture diagrams for the bilingual technical PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

ASSETS = Path(__file__).resolve().parent / "assets"

for _candidate in (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/STHeiti Light.ttc",
):
    if Path(_candidate).exists():
        font_manager.fontManager.addfont(_candidate)
        _prop = font_manager.FontProperties(fname=_candidate)
        plt.rcParams["font.family"] = _prop.get_name()
        plt.rcParams["axes.unicode_minus"] = False
        break


_FONT = None
for _candidate in (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
):
    if Path(_candidate).exists():
        _FONT = font_manager.FontProperties(fname=_candidate)
        break


def _box(ax, x, y, w, h, text, fc="#245c48", ec="#14241f", color="white", fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=color,
        fontsize=fontsize,
        wrap=True,
        fontproperties=_FONT,
    )


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#5d7268", lw=1.4))


def generate_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("System Architecture / 系统架构", fontsize=14, pad=12, fontproperties=_FONT)

    stages = [
        (0.3, 3.6, "Customer\n客户上下文"),
        (2.2, 3.6, "Risk Model\n风险预测"),
        (4.1, 3.6, "Uplift / CATE\n因果增益"),
        (6.0, 3.6, "Constraints\n约束过滤"),
        (7.9, 3.6, "Ranking\n干预排序"),
        (9.5, 3.6, "Agent Explain\n智能体解释"),
    ]
    for i, (x, y, t) in enumerate(stages):
        _box(ax, x, y, 1.5, 1.1, t, fc="#245c48" if i % 2 == 0 else "#2f6b54")
        if i < len(stages) - 1:
            _arrow(ax, x + 1.5, y + 0.55, stages[i + 1][0], y + 0.55)

    _box(ax, 1.5, 1.2, 2.4, 1.2, "Synthetic DGP\n合成数据生成\n+ confounding", fc="#f3e7d4", color="#14241f")
    _box(ax, 4.3, 1.2, 2.6, 1.2, "Meta-learners\nS / T / X\nCausal Forest", fc="#d9ebe3", color="#14241f")
    _box(ax, 7.3, 1.2, 2.6, 1.2, "Evaluation\nQini / AUUC\nPolicy value", fc="#e8eef1", color="#14241f")
    _arrow(ax, 2.7, 2.4, 2.9, 3.6)
    _arrow(ax, 5.6, 2.4, 4.9, 3.6)
    _arrow(ax, 8.6, 2.4, 8.6, 3.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_uplift_vs_prediction(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Prediction vs Treatment Effect / 预测 vs 干预效应", fontsize=13, fontproperties=_FONT)
    _box(ax, 0.5, 2.8, 3.8, 1.6, "Predictive model\n预测模型\nP(Y=1 | X)\nWho is at risk?\n谁有风险？", fc="#245c48")
    _box(ax, 5.5, 2.8, 3.8, 1.6, "Uplift / CATE\n增益模型\nP(Y|X,T=0)-P(Y|X,T=1)\nWho benefits?\n谁会受益？", fc="#b8925a")
    ax.text(
        5,
        1.5,
        "High risk customer may have low CATE for a given intervention\n高风险客户对某干预的增益可能很低",
        ha="center",
        va="center",
        fontsize=10,
        color="#14241f",
        fontproperties=_FONT,
        bbox=dict(boxstyle="round", facecolor="#f6f3ee", edgecolor="#c9d4ce"),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_agent_flow(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Agent Tool Flow / 智能体工具流", fontsize=13, fontproperties=_FONT)
    tools = [
        (0.4, 3.5, "get_risk_factors"),
        (2.7, 3.5, "get_evidence"),
        (5.0, 3.5, "uplift tools"),
        (7.3, 3.5, "rank + explain"),
    ]
    for i, (x, y, t) in enumerate(tools):
        _box(ax, x, y, 2.0, 0.9, t, fc="#1f4639", fontsize=8)
        if i < len(tools) - 1:
            _arrow(ax, x + 2.0, y + 0.45, tools[i + 1][0], y + 0.45)
    _box(ax, 2.5, 1.2, 5, 1.3,
         "LLM optional for narrative polish only\n大模型仅可选润色叙述\nCannot invent CATE / costs / probabilities\n不得编造增益、成本或概率",
         fc="#f6e4e0", color="#8f3d32", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_data_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Synthetic Experiment Pipeline / 合成实验流水线", fontsize=13, fontproperties=_FONT)
    steps = [
        (0.3, 1.5, "Customers\n客户特征"),
        (2.4, 1.5, "Assignment\n处理分配\n80% obs / 20% RCT"),
        (4.8, 1.5, "Outcomes\n损失结果\n+ severity"),
        (7.2, 1.5, "Train / Eval\n训练与评估"),
    ]
    for i, (x, y, t) in enumerate(steps):
        _box(ax, x, y, 1.9, 1.5, t, fc="#2f6b54" if i % 2 == 0 else "#3d7a62", fontsize=8)
        if i < len(steps) - 1:
            _arrow(ax, x + 1.9, y + 0.75, steps[i + 1][0], y + 0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def generate_all() -> dict[str, Path]:
    ASSETS.mkdir(parents=True, exist_ok=True)
    paths = {
        "architecture": ASSETS / "architecture.png",
        "uplift": ASSETS / "uplift_vs_prediction.png",
        "agent": ASSETS / "agent_flow.png",
        "data": ASSETS / "data_pipeline.png",
    }
    generate_architecture(paths["architecture"])
    generate_uplift_vs_prediction(paths["uplift"])
    generate_agent_flow(paths["agent"])
    generate_data_pipeline(paths["data"])
    return paths


if __name__ == "__main__":
    generate_all()
    print("Diagrams written to", ASSETS)
