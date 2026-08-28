# R057A 结果分析

## 预注册判定

`INCONCLUSIVE_OVERCONSERVATIVE`。五个配置共 640/640 条评分记录，`nonfinite_total=0`，机器选择过程标记 `selection_uses_test=false`。没有非 official 候选同时通过 checkpoint eligibility 与双分布功能 eligibility，因此冻结选择为 `official/H0`；按计划不得运行 R057B。

## 关键数值

| 配置 | 改变量 | hard_l11 gate 最差功能 delta | hard_l11 test 最差功能 delta | 结论 |
|---|---|---:|---:|---|
| H0 | default | -0.004003 | -0.025126 | 功能 gate/test 全部改善，但 checkpoint veto |
| H1 | max_steps=2 | +0.082282 | +0.048562 | 更新不足，功能失败 |
| H2 | max_steps=8 | +0.016391 | -0.016522 | gate 失败，不能用 test 翻案 |
| H3 | nsamples=16 | +0.009560 | +0.075422 | 更多校准样本未改善泛化 |
| H4 | blocksize=64 | +0.075311 | +0.022563 | 更细块粒度明显退化 |

H0 `hard_l11` 的 untouched test deltas 为：W2 mean/CVaR `-0.025126/-0.119563`，C4 mean/CVaR `-0.086018/-0.275001`。它被拒绝的直接原因是 W2 layer-11 checkpoint NMSE `+0.000106458`，而 C4 checkpoint NMSE 为 `-0.000457080`。

## 解释边界

R057A 说明超参数确实影响 hard-T 的功能行为：默认 4 步显著优于 2/8 步、nsamples16 和 blocksize64。但它没有证明“调参即可稳定解决三值 PTQ”，因为只有一个配置呈现功能成功，且 calibration seed 尚未复制。

更重要的是，结果提出一个新、可证伪的问题：严格的逐 checkpoint 零退化约束可能会否决能够跨 W2/C4 泛化的候选。这个结论目前只是单配置现象，不能通过事后放宽 epsilon 确认。R058 因此固定 H0/hard_l11，在 seed1 与全新序列上复制 checkpoint-veto 模式。

## Harness 事件

H0–H2 完成后，H3 首次因 runner 未导出既有 `PT2_DATA_ROOT` 而退出。修复仅恢复环境变量并加入按完整 `metrics.json` 跳过的断点续跑；H0–H2 未重算，失败退出标记被保留为 `EXIT_CODE.failed_h3_data_path`，最终运行 `EXIT_CODE=0`。
