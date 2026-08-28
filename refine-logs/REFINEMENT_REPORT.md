# Refinement Report

## Outcome

方法已从“Fisher 排序 + 多指标 gate”收缩为 **NC-PTQ**：一个可证伪的 no-cancellation constrained ternarization 命题。

## Key decisions

1. 不声称首次 cross-layer PTQ；CAT-Q 和 Cross-Layer Error Compensation 已覆盖该空间。
2. 唯一主创新是 per-layer local no-regression feasible set，目标是防止 calibration-specific cancellation。
3. 不继续逐层扫描；R053 后转向 R054 机理审计。
4. 不因 R054 单一负结果否定 hard-`T`；只否定“误差相消是主要迁移瓶颈”。

## Status

- Anchor: preserved.
- Focus: tight.
- Method maturity: mechanism-ready, not paper-ready.
- Highest risk: local no-regression may reject almost every useful joint move.

## Outputs

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/REVIEW_SUMMARY.md`
- `refine-logs/score-history.md`
