# CEGSP P4-R：QAT transition audit 与 edit-matched one-step baseline 实验方案

日期：2026-08-28  
目标：补上 P4 中 one-step QAT 对照偏弱的问题，进入顶会投稿导向的机制闭环。

## 1. 去重检查

| 既有实验 | 已回答 | 本轮是否重复 |
|---|---|---|
| P3A/P3B | fixed-rule 新 offset 与跨架构验证 | 不重复 |
| P4 | direct / CEGSP / one-step / 10,50-step QAT gap-cost 初版 | 不重复；P4-R 是修复 QAT baseline，而不是重新证明 CEGSP |
| P0 | OPT-125M 小模型 QAT gap 与 score-validity | 不重复；本轮在 OPT-350M 主设置上做 transition audit |
| P2/P3 random controls | CE-guided edit 强于 random | 不重复；本轮关注 QAT latent update 是否跨 ternary boundary |

## 2. 本次核心问题

P4 发现：

- CEGSP 改善 direct；
- 10-step QAT 是有效 reference；
- 但 one-step QAT 的 validation-best eta 为 0。

因此本轮回答：

1. One-step QAT 失败是否因为 latent update 没有产生足够 ternary state transition？
2. 当 one-step QAT 的离散变化数量与 CEGSP 近似匹配时，谁更好？
3. QAT steps 增加时，W2 validation、W2 untouched、C4 transfer 是否分离？

## 3. 固定设置

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| 数据 | Wikitext-2 + C4 validation |
| fit / val / W2 untouched / C4 untouched | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| CEGSP | canonical support relocation, top-6 |
| group size | 128 |
| threshold factor | 0.7 |
| max edits | 64 |
| one-step eta sweep | 1e-6 至 1e-1 |
| multi-step QAT | steps={5,10,20,50}, eta={0.001,0.003,0.01} |
| 选择规则 | eta 只按 W2 validation 选择，untouched/C4 只报告 |

## 4. 必须记录的 transition 类型

对每个 one-step eta 和 multi-step point 记录：

- changed_total；
- 0→+；
- 0→−；
- +→0；
- −→0；
- +→−；
- −→+。

这能验证 CEGSP 的机制解释：

> CEGSP 直接执行合法三值离散 transition，而 one-step QAT 必须依赖 latent FP update 恰好跨过 quantizer boundary。

## 5. 判据

### Strong PASS

- CEGSP 优于 edit-matched one-step QAT；
- CEGSP 恢复 multi-step QAT gap 的正比例；
- CEGSP 成本明显低于 multi-step QAT；
- transition audit 显示 one-step eta 存在明显 boundary threshold。

### PASS

- CEGSP 与 edit-matched one-step QAT 接近，但成本更低、无 optimizer、无 latent trajectory；
- 或 one-step QAT 更强，但需要显著更多/不可控的 discrete changes。

### Risk

- properly tuned one-step QAT 显著强于 CEGSP，且变化数量/成本相近；
- 此时才考虑后续改进 CEGSP 的 layer selector 或 multi-round，而不是现在提前修改方法。

## 6. 本次云端运行

Run ID：

`CEGSP-V2-P4R-OPT350M-QAT-TRANSITION-OFFSET2`

本地脚本：

`/home/x1shan/文档/ChatGPT/PTQ_paper/remote-tools/cegsp_v2_p4r_qat_transition_4090.py`

远端结果目录：

`/root/tqgsp-runs/CEGSP-V2-P4R-OPT350M-QAT-TRANSITION-OFFSET2/`

本轮不执行 P5 strong PTQ，因为 strong baseline 仍需先完成协议审计，避免跑出不可比结果。
