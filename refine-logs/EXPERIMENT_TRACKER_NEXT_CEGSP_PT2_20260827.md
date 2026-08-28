# 下一阶段实验 Tracker：CEGSP vs PT²

**状态：** 全部等待人工审核；当前没有云端任务被本 tracker 启动。

| Run ID | 阶段 | 目的 | 系统 | 模型/协议 | 指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| CEGSP-12A | A | 官方 PT² 协议复现 | FP16 / PT²-ATQ / PT²-ATQ+SSR | OPT-350M；128×2048 calibration；2048 PPL | W2/C4 PPL、finite、日志和环境审计 | MUST | DONE_BASELINE_REPRO_FAIL |
| CEGSP-12B | B | 独立公平比较 | FP16 / direct / PT² / CEGSP | OPT-350M；同 dtype、同 decoder Linear 范围、同 token budget | W2/C4 holdout NLL/PPL、paired delta、time、VRAM | MUST | BLOCKED_ON_12A_DIAGNOSIS |
| CEGSP-12C | C | 规模与架构迁移 | FP16 / direct / PT² / CEGSP | OPT-1.3B、Pythia-1B | W2/C4 NLL、PPL、有效 bit、cost | MUST | BLOCKED_ON_12B |
| CEGSP-12D | D | 三值特异性与简洁性 | random-support / signflip-only / support-only / CEGSP-joint | OPT-350M + Pythia-1B；固定 k25 | holdout NLL、CI、编辑统计 | MUST | BLOCKED_ON_12C |

## 启动纪律

- 12A 完成前不运行 12B–12D。
- 每次云端启动前提交实际命令和配置，等待用户明确批准。
- 不运行 PT²+CEGSP 组合，不使用 test 选择参数。
