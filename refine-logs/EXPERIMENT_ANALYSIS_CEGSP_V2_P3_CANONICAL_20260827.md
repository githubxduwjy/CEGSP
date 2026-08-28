# CEGSP-V2-P3 Canonical Fixed-Rule Validation 实验报告

日期：2026-08-27  
远端队列：`CEGSP_V2_P3_CANONICAL_VALIDATE`  
远端日志：`/root/tqgsp-runs/CEGSP-V2-P3-CANONICAL-VALIDATE/console.log`  
远端脚本：`/root/tqgsp-work/run_p3_canonical_validate.sh`  
本地脚本备份：`/home/x1shan/文档/ChatGPT/PTQ_paper/remote-tools/run_p3_canonical_validate.sh`

## 1. 实验目的

本轮 P3 不做新模块、不换新方向，只验证一个问题：

> 当方法、编辑对象、预算选择规则全部冻结后，换新的 calibration/validation offset，CEGSP 是否仍然改善 untouched data？

冻结的 canonical rule：

```text
Direct ternary
→ one quantized-point CE gradient
→ Q/K
→ support relocation primary
→ small fixed top-k
→ untouched evaluation
```

本轮不使用 QAT teacher、不做 optimizer step、不使用 untouched test 选择参数。

## 2. 运行完整性

前一次 P3 screen 是空跑：screen 消失、GPU 空闲、console.log 为 0 字节。  
本轮先确认该问题，然后用远端脚本文件重启同一 P3 预注册实验。

完成状态：

| Run | 状态 | result.json | elapsed |
|---|---|---|---:|
| P3A | complete | `/root/tqgsp-runs/CEGSP-V2-P3A-OPT350M-CANONICAL-OFFSET2/result.json` | 84.55 s |
| P3B | complete | `/root/tqgsp-runs/CEGSP-V2-P3B-PYTHIA1B-CANONICAL-OFFSET1/result.json` | 45.41 s |

GPU：

| Run | GPU | peak memory |
|---|---|---:|
| P3A | RTX 4090 24GB | 1.20 GB |
| P3B | RTX 4090 24GB | 4.21 GB |

## 3. P3A：OPT-350M 新 offset + Wikitext/C4

配置：

| 项目 | 设置 |
|---|---:|
| 模型 | `facebook/opt-350m` |
| adapter | OPT separate Q/K linear |
| layers | 0–23 |
| fit / val / untouched W2 / untouched C4 | 8 / 8 / 64 / 32 |
| offsets | fit=8192, val=8192, C4=16384 |
| k | 6 |
| max edits | 64 |
| dtype | bf16 |

Direct ternary baseline：

| split | NLL |
|---|---:|
| val | 8.4695 |
| untouched W2 | 8.4652 |
| untouched C4 | 8.1304 |

核心 patch set：

| 方法 | selected layers / edits | val ΔNLL | W2 untouched ΔNLL | C4 untouched ΔNLL |
|---|---|---:|---:|---:|
| support top6 | [13,17,14,19,23,16] | -0.1798 | -0.1686 | **-0.2901** |
| signflip top6 | [17,13,23,19,14,9] | -0.1554 | -0.1514 | -0.2665 |
| joint top6 | 13:support,17:signflip,23:signflip,14:support,19:support,16:support | **-0.1907** | **-0.1837** | -0.2742 |
| random joint top6 | random layers/actions | -0.0004 | +0.0002 | +0.0012 |
| random candidate on CE joint layers | same CE layers, random actions | -0.0006 | -0.0005 | +0.0001 |

P3A 判定：

- W2 untouched 改善：yes；
- C4 untouched 改善：yes；
- random control 近零：yes；
- 固定新 offset 下改善：yes。

P3A 是 strong positive。

## 4. P3B：Pythia-1B 新 offset + Wikitext

配置：

| 项目 | 设置 |
|---|---:|
| 模型 | `EleutherAI/pythia-1b` |
| adapter | GPT-NeoX fused QKV row-slice |
| layers | 0–15 |
| fit / val / untouched W2 | 8 / 8 / 32 |
| offsets | fit=4096, val=4096 |
| k | 4 |
| max edits | 64 |
| dtype | bf16 |
| C4 | disabled due offline cache |

Direct ternary baseline：

| split | NLL |
|---|---:|
| val | 9.1137 |
| untouched W2 | 8.8771 |

核心 patch set：

| 方法 | selected layers / edits | val ΔNLL | W2 untouched ΔNLL |
|---|---|---:|---:|
| support top4 | [4,5,6,3] | **-0.3299** | -0.2919 |
| signflip top4 | [7,6,10,8] | -0.2121 | -0.2326 |
| joint top4 | 3:support,4:support,5:support,6:support | **-0.3299** | -0.2919 |
| support on signflip-selected layers | [7,6,10,8] | -0.2994 | **-0.3749** |
| random joint top4 | random layers/actions | -0.0056 | -0.0035 |
| random candidate on CE joint layers | same CE layers, random actions | +0.0003 | +0.0003 |

P3B 判定：

- W2 untouched 改善：yes；
- 跨 Pythia 新 offset 改善：yes；
- random control 近零：yes；
- support relocation 仍强于 signflip：yes。

P3B 是 strong positive for Wikitext split robustness。

## 5. 预注册 gate 判定

贴文建议的 Strong PASS 条件：

1. OPT-350M 新 offset：
   - W2 untouched 改善；
   - C4 untouched 改善。
2. Pythia-1B 新 offset：
   - W2 untouched 改善。
3. 不使用 untouched test 选择 k/layer。

本轮结果：

| 条件 | 结果 |
|---|---|
| OPT-350M W2 untouched 改善 | PASS |
| OPT-350M C4 untouched 改善 | PASS |
| Pythia-1B W2 untouched 改善 | PASS |
| random control 近零 | PASS |
| 不用 untouched 选参 | PASS |

总判定：`STRONG_PASS_CANONICAL_FIXED_RULE_VALIDATION`

## 6. 研究意义

P3 是目前 CEGSP 最关键的一组验证，因为它不再回答“某个模块有没有用”，而是回答：

> 固定 CEGSP 规则后，换 calibration/test offset 是否仍然有效？

本轮结果支持以下更稳的论文表述：

1. CEGSP 不是只在某个 split 上有效；
2. CEGSP 不是只在 OPT-family 上有效；
3. 量化点 CE gradient 的选择信号显著强于 random；
4. 当前最可靠的三值特异性来自 support relocation，即 `{−α,0,+α}` 中零态/非零态支撑的重分配；
5. CEGSP 可以在 4090 24GB 上低成本验证，运行时间远低于 QAT 类训练过程。

## 7. 边界与不能过度声称的内容

1. 还不能声称超过最新 ternary PTQ，因为强 PTQ baseline 仍需审计。
2. P3B 没有 C4，因为远端离线缓存限制；Pythia 的跨域泛化还没验证。
3. 当前 k 是固定小预算验证，不等于已经解决所有模型上的 automatic stopping。
4. joint 在 OPT-350M 上略优于 support，但 Pythia 上退化为 support-only；因此主 claim 仍应放在 support relocation，而不是 mixed/joint。

## 8. 下一步建议

P3 已经给出强正证据，下一步不建议继续做零散小消融。建议转入论文级闭环：

1. **强基线审计**
   - 解决 direct ternary 与 PT²/其他 ternary PTQ 是否可比；
   - 在该 gate 前不写 SOTA claim。

2. **QAT gap / cost 同场实验**
   - direct / CEGSP / one-step QAT / small-step QAT；
   - 报告 NLL、wall-clock、peak memory；
   - 证明 CEGSP 是 PTQ 级成本，而不是 QAT teacher。

3. **canonical main matrix**
   - OPT-350M、Pythia-1B 至少两模型；
   - W2/C4 尽量齐全；
   - NLL 与 PPL 同报。

本轮结论：P3 固定规则验证通过，CEGSP 方向应继续，但下一阶段必须优先补强基线，而不是继续发散模块。
