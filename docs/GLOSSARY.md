# Bilingual Terminology Glossary / 中英术语表

Source of truth for documentation, UI copy, and the technical PDF.

| English | 中文 |
|---|---|
| Loss Prevention | 损失预防 |
| Predictive Risk Model | 预测风险模型 |
| Uplift Modeling | 增益建模 / 提升模型 |
| CATE (Conditional Average Treatment Effect) | 条件平均处理效应（CATE） |
| Meta-learner (S/T/X) | 元学习器（S/T/X） |
| Causal Forest | 因果森林 |
| Confounding / Selection Bias | 混杂 / 选择偏差 |
| Propensity Score | 倾向得分 |
| Intervention Ranking | 干预排序 |
| Customer Burden | 客户负担 |
| Feasibility | 可行性 |
| Constraint Filtering | 约束过滤 |
| Counterfactual Simulation | 反事实模拟 |
| Agent Tool Calling | 智能体工具调用 |
| Explainability | 可解释性 |
| Calibration / ECE | 校准 / 期望校准误差 |
| Qini / AUUC | Qini 系数 / 增益曲线面积 |
| Policy Value | 策略价值 |
| Explore vs Exploit | 探索与利用 |
| Responsible AI | 负责任的人工智能 |
| No Intervention | 无干预 |
| Grounded Explanation | 有依据的解释 |
| Synthetic Data Generating Process (DGP) | 合成数据生成过程 |

## Core distinction / 核心区分

```
Prediction:  P(loss | X)           → who is at risk?
预测：       P(损失 | X)            → 谁有风险？

Uplift:      P(loss | X,T=0) − P(loss | X,T=1) → who benefits?
增益：       P(损失 | X,T=0) − P(损失 | X,T=1) → 谁会受益？
```
