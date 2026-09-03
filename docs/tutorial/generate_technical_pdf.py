#!/usr/bin/env python3
"""Generate bilingual (EN + 简体中文) technical tutorial PDF.

Run:
  PYTHONPATH=src python docs/tutorial/generate_technical_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagrams import generate_all  # noqa: E402

OUT_PDF = ROOT / "docs" / "Loss_Prevention_Technical_Tutorial_EN_ZH.pdf"
ASSETS = Path(__file__).resolve().parent / "assets"

FONT_REG = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ITAL = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
FONT_UNI = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_MONO = "/System/Library/Fonts/Supplemental/Courier New.ttf"


class BiPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(format="A4", unit="mm")
        self.set_auto_page_break(auto=True, margin=16)
        self.add_font("Body", "", FONT_REG)
        self.add_font("Body", "B", FONT_BOLD)
        self.add_font("Body", "I", FONT_ITAL)
        # Arial Unicode covers Latin + CJK for bilingual body text
        self.add_font("Uni", "", FONT_UNI)
        self.add_font("Uni", "B", FONT_UNI)
        self.add_font("Mono", "", FONT_MONO)
        self.chapter = ""

    def header(self) -> None:
        if self.page_no() <= 1:
            return
        self.set_font("Uni", "", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 6, "Loss Prevention Agent · 损失预防智能体", align="L")
        self.cell(0, 6, self.chapter[:40], align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(36, 92, 72)
        self.set_line_width(0.35)
        self.line(12, 12, 198, 12)
        self.ln(3)

    def footer(self) -> None:
        if self.page_no() <= 1:
            return
        self.set_y(-12)
        self.set_font("Body", "", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 8, f"{self.page_no() - 1}", align="C")

    def _center(self, text: str, size: int = 11, bold: bool = False) -> None:
        self.set_x(self.l_margin)
        self.set_font("Uni", "B" if bold else "", size)
        self.multi_cell(self.epw, size * 0.55, text, align="C")

    def cover(self) -> None:
        self.add_page()
        self.ln(36)
        self.set_text_color(20, 36, 31)
        self._center("Loss Prevention Recommendation Agent", size=22, bold=True)
        self.ln(2)
        self.set_text_color(36, 92, 72)
        self._center("保险损失预防推荐智能体 - 技术教程", size=14)
        self.ln(8)
        self.set_text_color(60, 80, 72)
        self._center("Bilingual technical tutorial", size=11)
        self._center("Predictive risk / Uplift CATE / Causal forests / Agent tooling", size=10)
        self._center("双语技术教程：风险预测 / 增益建模 / 因果森林 / 智能体工具", size=10)
        self._center("Synthetic data only / 仅使用合成数据", size=10)
        self.ln(16)
        self.set_draw_color(36, 92, 72)
        self.set_line_width(1.0)
        self.line(70, self.get_y(), 140, self.get_y())
        self.ln(10)
        self._center("Portfolio / research engineering project", size=10)
        self._center("面向技术面试与研究工程展示", size=10)

    def h1(self, en: str, zh: str) -> None:
        self.chapter = en
        self.add_page()
        self.set_x(self.l_margin)
        self.set_font("Uni", "B", 16)
        self.set_text_color(20, 36, 31)
        self.multi_cell(self.epw, 8, en)
        self.set_x(self.l_margin)
        self.set_font("Uni", "", 12)
        self.set_text_color(36, 92, 72)
        self.multi_cell(self.epw, 7, zh)
        self.ln(1)
        self.set_draw_color(36, 92, 72)
        self.set_line_width(0.7)
        self.line(self.l_margin, self.get_y(), self.l_margin + 55, self.get_y())
        self.ln(5)

    def h2(self, en: str, zh: str) -> None:
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Uni", "B", 12)
        self.set_text_color(20, 36, 31)
        self.multi_cell(self.epw, 6, en)
        self.set_x(self.l_margin)
        self.set_font("Uni", "", 10)
        self.set_text_color(36, 92, 72)
        self.multi_cell(self.epw, 5.5, zh)
        self.ln(1)

    def bi(self, en: str, zh: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Uni", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(self.epw, 5.2, en)
        self.set_x(self.l_margin)
        self.set_text_color(70, 90, 82)
        self.multi_cell(self.epw, 5.2, zh)
        self.ln(1.5)

    def bullet_bi(self, en: str, zh: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Uni", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(self.epw, 5.1, f"- {en}")
        self.set_x(self.l_margin)
        self.set_text_color(70, 90, 82)
        self.multi_cell(self.epw, 5.1, f"  {zh}")
        self.ln(0.6)

    def note(self, en: str, zh: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Uni", "B", 9)
        self.set_text_color(36, 92, 72)
        self.multi_cell(self.epw, 5, "KEY IDEA / 关键思想")
        self.set_x(self.l_margin)
        self.set_font("Uni", "", 9)
        self.set_text_color(30, 41, 59)
        self.multi_cell(self.epw, 4.8, en)
        self.set_x(self.l_margin)
        self.multi_cell(self.epw, 4.8, zh)
        self.ln(3)

    def formula(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Mono", "", 9)
        self.set_text_color(20, 36, 31)
        self.set_fill_color(246, 243, 238)
        self.multi_cell(self.epw, 5.5, text, fill=True)
        self.ln(2)

    def img(self, path: Path, w: float = 180) -> None:
        if path.exists():
            self.image(str(path), w=w)
            self.ln(4)

    def table_glossary(self, rows: list[tuple[str, str]]) -> None:
        col = min(90, self.epw / 2)
        self.set_x(self.l_margin)
        self.set_font("Uni", "B", 9)
        self.set_fill_color(36, 92, 72)
        self.set_text_color(255, 255, 255)
        self.cell(col, 7, "English", border=1, fill=True)
        self.cell(col, 7, "中文", border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_font("Uni", "", 8)
        self.set_text_color(30, 41, 59)
        for i, (en, zh) in enumerate(rows):
            self.set_x(self.l_margin)
            if i % 2 == 0:
                self.set_fill_color(248, 250, 248)
            else:
                self.set_fill_color(255, 255, 255)
            self.cell(col, 6.2, en[:48], border=1, fill=True)
            self.cell(col, 6.2, zh[:32], border=1, fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)


def build() -> Path:
    diagrams = generate_all()
    pdf = BiPDF()
    pdf.set_margins(12, 16, 12)
    pdf.cover()

    # Glossary first for translation consistency
    pdf.h1("0. Terminology Glossary", "0. 术语表")
    pdf.bi(
        "This glossary is the source of truth for bilingual terminology used throughout the tutorial.",
        "本术语表是全文中英术语一致性的基准。",
    )
    pdf.table_glossary(
        [
            ("Loss Prevention", "损失预防"),
            ("Predictive Risk Model", "预测风险模型"),
            ("Uplift Modeling", "增益建模 / 提升模型"),
            ("CATE (Conditional Average Treatment Effect)", "条件平均处理效应（CATE）"),
            ("Meta-learner (S/T/X)", "元学习器（S/T/X）"),
            ("Causal Forest", "因果森林"),
            ("Confounding / Selection Bias", "混杂 / 选择偏差"),
            ("Propensity Score", "倾向得分"),
            ("Intervention Ranking", "干预排序"),
            ("Customer Burden", "客户负担"),
            ("Feasibility", "可行性"),
            ("Constraint Filtering", "约束过滤"),
            ("Counterfactual Simulation", "反事实模拟"),
            ("Agent Tool Calling", "智能体工具调用"),
            ("Explainability", "可解释性"),
            ("Calibration / ECE", "校准 / 期望校准误差"),
            ("Qini / AUUC", "Qini 系数 / 增益曲线面积"),
            ("Policy Value", "策略价值"),
            ("Explore vs Exploit", "探索与利用"),
            ("Responsible AI", "负责任的人工智能"),
        ]
    )

    pdf.h1("1. Project Overview", "1. 项目概览")
    pdf.bi(
        "This project answers: given a customer/policy/property context, which intervention is most likely to reduce future insurance loss?",
        "本项目回答：在给定客户/保单/财产上下文时，哪一种干预最有可能降低未来保险损失？",
    )
    pdf.bi(
        "It is a research-engineering system demonstrating predictive ML, uplift/causal inference, recommender-style ranking, tool-using agents, explainability, and synthetic experimentation.",
        "它是一个研究工程系统，展示预测机器学习、增益/因果推断、推荐式排序、工具型智能体、可解释性与合成实验。",
    )
    pdf.h2("Key capabilities", "关键能力")
    for en, zh in [
        ("Calibrated P(future loss) risk scoring", "校准后的未来损失概率风险评分"),
        ("Per-intervention CATE via S/T/X-learners and causal forest", "基于 S/T/X 学习器与因果森林的逐干预 CATE"),
        ("Constraint-aware multi-objective ranking", "带约束的多目标干预排序"),
        ("Grounded agent explanations (no invented treatment effects)", "有依据的智能体解释（不编造处理效应）"),
        ("Interactive counterfactual simulator and research lab UI", "交互式反事实模拟器与研究实验界面"),
    ]:
        pdf.bullet_bi(en, zh)
    pdf.note(
        "Predicting who is at risk is not the same as predicting who will benefit from an intervention.",
        "预测谁有风险，并不等于预测谁会从干预中受益。",
    )

    pdf.h1("2. System Architecture", "2. 系统架构")
    pdf.img(diagrams["architecture"])
    pdf.bi(
        "Online path: Customer → Risk Model → Risk Factors → Candidate Interventions → Uplift/Causal Model → Expected Impact → Constraint Filtering → Ranking → AI Explanation.",
        "在线路径：客户 → 风险模型 → 风险因子 → 候选干预 → 增益/因果模型 → 期望影响 → 约束过滤 → 排序 → AI 解释。",
    )
    pdf.bi(
        "Implemented as a Python package (`loss_prevention`) with FastAPI service, static production UI, Typer CLI, and offline training/evaluation modules. There is no external database; state is synthetic and in-memory/model-artifact based.",
        "实现为 Python 包（`loss_prevention`），包含 FastAPI 服务、静态生产级 UI、Typer CLI，以及离线训练/评估模块。无外部数据库；状态基于合成数据与内存/模型产物。",
    )

    pdf.h1("3. Data Architecture", "3. 数据架构")
    pdf.img(diagrams["data"])
    pdf.bi(
        "All data are synthetic. The DGP creates customer features (maintenance, hazard, claims history, engagement, contactability, etc.) and historical intervention assignments.",
        "全部数据为合成数据。数据生成过程创建客户特征（维护、危险暴露、理赔历史、参与度、可联系性等）与历史干预分配。",
    )
    pdf.bi(
        "80% of assignments are observational with confounding: ops preferentially treats higher-risk and more contactable customers. 20% are RCT arms with propensity 0.5 for unconfounded reference.",
        "80% 分配为带混杂的观测数据：运营更倾向处理高风险、更易联系的客户。20% 为倾向得分=0.5 的随机对照试验臂，用作无混杂参照。",
    )
    pdf.h2("Why naive ATE misleads", "为何朴素 ATE 会误导")
    pdf.bi(
        "Because treated and control groups differ systematically in baseline hazard, a naive control-minus-treated loss-rate comparison mixes selection differences with true effects.",
        "由于处理组与对照组在基线危险上系统性不同，朴素的“对照损失率−处理损失率”会把选择差异与真实效应混在一起。",
    )

    pdf.h1("4. Machine Learning", "4. 机器学习")
    pdf.h2("Predictive model", "预测模型")
    pdf.formula("mu(X) = P(Y=1 | X)")
    pdf.bi(
        "Implemented as gradient boosting with isotonic calibration. Trained preferentially on untreated rows to approximate baseline risk without treatment contamination.",
        "实现为带等渗校准的梯度提升。优先在未处理样本上训练，以近似不受干预污染的基线风险。",
    )
    pdf.h2("Uplift / CATE", "增益 / CATE")
    pdf.formula("tau(X,T) = P(Y=1 | X, T=0) - P(Y=1 | X, T=1)")
    pdf.bi(
        "Positive tau means absolute risk reduction. Meta-learners: S-learner (single model with T), T-learner (separate outcome models), X-learner (imputed effects + propensity weighting).",
        "正的 tau 表示绝对风险下降。元学习器：S（含处理指示的单模型）、T（处理/对照分离模型）、X（效应填补 + 倾向加权）。",
    )
    pdf.img(diagrams["uplift"])
    pdf.h2("Causal forest", "因果森林")
    pdf.bi(
        "Educational honest forest: structure sample vs estimation sample; splits encourage treatment-effect heterogeneity; leaf CATE is control-minus-treated loss rate.",
        "教学用途的诚实森林：结构样本与估计样本分离；分裂鼓励处理效应异质性；叶节点 CATE 为对照与处理损失率之差。",
    )
    pdf.h2("Metrics", "指标")
    for en, zh in [
        ("ROC-AUC, Brier, log loss, ECE for prediction/calibration", "ROC-AUC、Brier、对数损失、ECE 用于预测/校准"),
        ("Qini and AUUC for uplift ranking quality", "Qini 与 AUUC 用于增益排序质量"),
        ("Policy value = mean oracle CATE under a decision policy", "策略价值 = 决策策略下 oracle CATE 均值"),
    ]:
        pdf.bullet_bi(en, zh)

    pdf.h1("5. AI / LLM Architecture", "5. AI / 大模型架构")
    pdf.bi(
        "The LLM is optional and disabled unless OPENAI_API_KEY is set. Deterministic model services own all numeric scores.",
        "大模型为可选项，仅在设置 OPENAI_API_KEY 时启用。所有数值分数由确定性模型服务给出。",
    )
    pdf.bi(
        "If enabled, the LLM may rewrite narrative text but is instructed not to invent probabilities, CATEs, costs, or confidence scores.",
        "若启用，大模型可润色叙述，但被明确要求不得编造概率、CATE、成本或置信度。",
    )
    pdf.note(
        "This separation is intentional: LLMs are good at communication; bad as silent inventers of causal effects.",
        "这种分离是有意设计：大模型擅长表达，不适合默默编造因果效应。",
    )

    pdf.h1("6. Agent Architecture", "6. 智能体架构")
    pdf.img(diagrams["agent"])
    pdf.bi(
        "LossPreventionAgent orchestrates tools: inspect risk factors, retrieve evidence, score candidates, call uplift/causal tools, compare interventions, explain trade-offs, emit structured Recommendation.",
        "LossPreventionAgent 编排工具：检查风险因子、检索证据、给候选打分、调用增益/因果工具、比较干预、解释权衡、输出结构化 Recommendation。",
    )
    for en, zh in [
        ("get_risk_factors / get_evidence_summary", "获取风险因子 / 证据摘要"),
        ("get_uplift_estimates / get_ranked_interventions", "获取增益估计 / 排序结果"),
        ("simulate_intervention (counterfactual)", "干预反事实模拟"),
        ("compare_interventions", "比较两种干预"),
    ]:
        pdf.bullet_bi(en, zh)

    pdf.h1("7. Ranking & Constraints", "7. 排序与约束")
    pdf.formula(
        "utility = w_b*CATE + w_c*confidence + w_f*feasibility - w_cost*cost_norm - w_burden*burden"
    )
    pdf.bi(
        "Hard filters: recent-contact suppression, annual intervention caps, fraud/specialist suitability, burden limits for low-engagement customers. no_intervention is first-class.",
        "硬约束：近期联系抑制、年度干预上限、欺诈/专家适用性、低参与客户负担限制。无干预（no_intervention）是一等选项。",
    )

    pdf.h1("8. Engineering Architecture", "8. 工程架构")
    pdf.bi(
        "Package layout: data/, models/, ranking/, agent/, explain/, evaluation/, api/, ui/static/. CLI entrypoint: loss-prevention. API serves both JSON and the production UI.",
        "包结构：data/、models/、ranking/、agent/、explain/、evaluation/、api/、ui/static/。CLI 入口：loss-prevention。API 同时提供 JSON 与生产 UI。",
    )
    pdf.h2("Important API endpoints (implemented)", "重要 API 端点（已实现）")
    for en, zh in [
        ("GET /customers/{id}/recommend — recommendation + agent_trace", "推荐结果 + 智能体轨迹"),
        ("POST /counterfactual — what-if simulator", "反事实模拟"),
        ("GET /customers/{id}/method-compare — S/T/X/CF comparison", "元学习器比较"),
        ("GET /research/deploy|experiments — offline research metrics", "离线研究指标"),
        ("GET /architecture — pipeline/tool metadata for UI", "架构元数据"),
    ]:
        pdf.bullet_bi(en, zh)

    pdf.h1("9. Evaluation & Testing", "9. 评估与测试")
    pdf.bi(
        "Automated tests cover synthetic DGP properties, predictive/uplift/causal models, ranking constraints, agent grounding, API contracts, static UI assets, and HTTP end-to-end journeys.",
        "自动化测试覆盖合成数据性质、预测/增益/因果模型、排序约束、智能体证据落地、API 契约、静态 UI 资源，以及 HTTP 端到端旅程。",
    )
    pdf.bi(
        "Research experiments investigate selection bias, treatment heterogeneity, policy optimization, and explore-vs-exploit diversity/value tradeoffs.",
        "研究实验考察选择偏差、处理异质性、策略优化，以及探索-利用的多样性/价值权衡。",
    )

    pdf.h1("10. Design Decisions", "10. 设计决策")
    pdf.h2("Why uplift instead of risk-only targeting?", "为何不用纯风险定向？")
    pdf.bi(
        "Risk-only policies over-spend on unresponsive high-risk customers. Uplift ranking optimizes expected risk reduction under cost/burden constraints.",
        "纯风险策略会在无响应高风险客户上过度花费。增益排序在成本/负担约束下优化期望风险下降。",
    )
    pdf.h2("Why not let the LLM choose the intervention numerically?", "为何不让大模型直接给出数值决策？")
    pdf.bi(
        "Causal effects require experimental/observational identification assumptions. LLMs do not estimate CATEs from first principles in this system; inventing them would be unsafe for a decision product.",
        "因果效应需要识别假设。本系统中大模型不会凭空估计 CATE；编造数值对决策产品不安全。",
    )
    pdf.h2("Why synthetic data?", "为何使用合成数据？")
    pdf.bi(
        "Insurance microdata are confidential. A transparent DGP with oracle CATEs enables honest evaluation of confounding and policy value.",
        "保险微观数据涉密。带 oracle CATE 的透明数据生成过程，可诚实评估混杂与策略价值。",
    )

    pdf.h1("11. Limitations", "11. 局限")
    for en, zh in [
        ("Synthetic DGP ≠ real-world claims processes", "合成数据生成过程 ≠ 真实理赔过程"),
        ("Observational CATE estimates remain assumption-dependent", "观测 CATE 估计仍依赖假设"),
        ("Causal forest is educational, not econml-production scale", "因果森林为教学实现，非生产级 econml 规模"),
        ("No persistent customer database or auth layer", "无持久客户库与鉴权层"),
        ("Optional LLM path not required for core recommendations", "核心推荐不依赖可选大模型路径"),
    ]:
        pdf.bullet_bi(en, zh)

    pdf.h1("12. Future Improvements", "12. 未来改进")
    for en, zh in [
        ("Integrate real RCT logs / doubly robust estimators", "接入真实 RCT 日志 / 双重稳健估计"),
        ("Online bandit allocation with delayed claim outcomes", "带延迟理赔结果的在线老虎机分配"),
        ("Model registry, drift monitors, and canary policies", "模型注册、漂移监控与金丝雀策略"),
        ("Stronger accessibility tooling and browser E2E (Playwright) in CI", "更强无障碍工具与 CI 中的浏览器 E2E（Playwright）"),
    ]:
        pdf.bullet_bi(en, zh)

    pdf.h1("13. How to Run", "13. 运行方式")
    pdf.formula(
        "cd loss-prevention\n"
        "source ../.venv/bin/activate\n"
        "pip install -e . --no-deps\n"
        "loss-prevention train\n"
        "loss-prevention serve-ui\n"
        "# open http://127.0.0.1:8010/\n"
        "pytest -q"
    )
    pdf.bi(
        "Open the UI to walk Risk Overview -> Interventions -> Counterfactual -> Model Compare -> Research Lab.",
        "打开 UI，依次体验风险总览 -> 干预排序 -> 反事实 -> 模型比较 -> 研究实验。",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PDF))
    return OUT_PDF


if __name__ == "__main__":
    out = build()
    print(f"Wrote {out}")
