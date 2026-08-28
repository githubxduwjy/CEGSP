# R058：checkpoint veto 的冻结复现实验

## 问题

R057A 中仅 H0/default 的 `hard_l11` 在 gate 与 untouched test 的 W2/C4 mean-NLL、CVaR10 全部改善，但因 gate 上 W2 layer-11 NMSE 相对 official 增加 `+1.064576e-4` 被 checkpoint 约束否决。R058 不放宽 epsilon，也不重新挑参；它检验这一 veto 是否会在独立 calibration seed 和全新序列上重复出现。

## 冻结设计

- 固定超参数：`calib_nsamples=8, blocksize=128, max_steps=4, validation_fraction=0.25`。
- calibration seed：1；不搜索 seed。
- 固定目标候选：`hard_l11`；`hard_l10` 与 composition 仅作为完整输出，不参与选择。
- 层窗口：`(10,11)`。
- 新序列：120–135；gate=120–127，untouched test=128–135。
- 数据：WikiText2 与 C4。
- epsilon 继续为 0；禁止根据结果修改。
- 预期原始评分行数：`4 candidates × 2 datasets × 16 sequences = 128`。

## 机器判定

1. `INVALID`：配置、128 行、序列、split、finite 或 nonfinite 审计失败。
2. `REJECT_CANDIDATE`：固定 `hard_l11` 在 gate 的任一 W2/C4 mean-NLL 或 CVaR10 delta > 0；此时不解释 test。
3. `FAIL_FUNCTIONAL_GENERALIZATION`：gate 四项均 <=0，但 untouched test 任一项 >0。
4. `SUPPORT_VETO_OVERCONSERVATIVE`：gate/test 八项功能 delta 全部 <=0，但 gate 上任一 layer10/11 NMSE delta >0，重复“功能泛化成功却被 checkpoint veto”现象。
5. `SUPPORT_HARD_L11_NO_VETO`：gate/test 功能指标全部 <=0，且 checkpoint NMSE 全部 <=0；支持 H0/hard_l11 的可复现性，但不支持 veto 过严解释。

R058 只判断 checkpoint veto 是否稳定错杀，不把 R057A 事后改判为成功，也不授权直接启动原 R057B。若支持 veto 过严，下一步应预注册带统计容差或 Pareto dominance 的新 gate，并用全新数据验证；不得直接把 `1.064576e-4` 写成 epsilon。
