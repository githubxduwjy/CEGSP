# Experiment Plan: TDBT2 Support-Transport PTQ

日期：2026-08-26

状态：预注册下一步实验；不自动启动云端。

## Problem

固定三值码本下，PTQ 一次性选择的离散 codeword 与预训练模型函数之间存在 compatibility gap。今天的诊断提示，QAT 的早期改善更像 `0 <-> nonzero` support reallocation，而不是显式 sign flip。因此下一步不做完整全模型 TDBT，也不使用 QAT teacher；先验证三值 support transport 是否在相同预算下比 endpoint-only 或 one-shot gradient 更有用。

## Method Thesis

TDBT2-SupportTransport 在固定三值码本和固定 nonzero budget 下做成对支撑运输：

```text
donor:    nonzero -> zero
receiver: zero -> nonzero with sign(W_fp)
```

候选只有在中间路径满足 local trust region 时才进入 beam；最终以 Q/K 与 V/O composed-operator distortion 作为主判据。

## Legacy Evidence Used

允许使用：

- 今天发现的 explicit next-token CE 修正；
- 4090 24GB 上 OPT-125M/350M 的可运行经验；
- QAT trajectory 中 support transition 主导这一现象，作为 hypothesis motivation；
- FP32 path metric 比 BF16 更能分辨小 barrier 差异这一工程提示。

不允许使用：

- 旧 G4090 结果中的 pass/fail gate；
- 旧 R014-R058 的 projection mask、epsilon、layer choice；
- QAT checkpoint、QAT logits、QAT latent weights 或 QAT state prior；
- “zero-mediated sign flip 是核心机制”的默认假设。

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: 三值 support transport 在 strict PTQ 设置下有独立价值 | 证明方法不是普通低比特 beam search 或更多计算 | 在相同候选数、forward 次数和 alpha refit 下，TDBT2-F/G 相对 endpoint-only beam 在 held-out operator distortion 上改善 >=5%，且至少 3/4 个 layer-operator pair 方向一致 | B1, B2 |
| C2: 收益来自三值 support 机制，而不是 QAT teacher 或 sign flip 叙事 | 回应“这个方法低 bit 都能套用”的质疑 | support-swap 优于 sign-flip-only；不使用 QAT 信息；binary control 或 no-zero variant 不显示同等收益 | B2, B3 |

Anti-claims to rule out:

- 改善只是更多候选搜索；
- 改善只在 calibration fit 上发生；
- one-shot quantized-gradient 已经足够；
- support transport 只降低 path barrier，但不改善 endpoint operator；
- 方法实际依赖 QAT teacher。

## Experiment Blocks

### Block B0: Harness And Metric Sanity

- Claim tested: 实验实现和指标可靠。
- Why this block exists: 避免今天发现的 metric bug 再次污染结论。
- Dataset / split / task: Wikitext-2 train/validation，seq 128，OPT-125M，layer 0，Q/K 与 V/O 各一组。
- Compared systems: FP reference、direct ternary PTQ、identity reload、single candidate patch。
- Metrics: explicit next-token CE sanity、D_QK、D_VO、finite、reload parity、peak VRAM。
- Setup details: no QAT；FP32 operator metric；group size 128；threshold factor 0.7 固定为工程默认，不作为研究 gate。
- Success criterion: 全部 finite；reload 后指标 bitwise 或 tolerance 内一致；single patch 能改变 operator metric。
- Failure interpretation: harness failure，不评价 idea。
- Table / figure target: appendix sanity table。
- Priority: MUST-RUN。

### Block B1: Main Support-Transport Feasibility

- Claim tested: support transport 是否优于 endpoint-only 和 one-shot。
- Why this block exists: 这是下一步最小判别实验。
- Dataset / split / task:
  - Model: OPT-350M。
  - Layers: 0、7、15、23。
  - Operators: Q/K composed attention logits；V/O composed value-output map。
  - fit-A: Wikitext-2 train 固定 32 windows，seq 128。
  - val-B: Wikitext-2 validation 固定 16 windows。
  - untouched-W: Wikitext-2 validation 后续不重叠 16 windows。
  - untouched-C: 若 C4 可用，C4 validation 16 windows；若不可用，记录 dataset failure，不用 fallback 冒充 C4。
- Compared systems:
  - `PTQ-direct`: groupwise ternary PTQ。
  - `QG-one-shot`: 一次 quantized-point gradient 排序，直接应用 top support swaps，无 path barrier。
  - `endpoint-beam`: 同样 beam width 和候选预算，只看 endpoint D_QK/D_VO，不看中间 barrier。
  - `TDBT2-F`: forward-only salience 候选 + support transport + barrier。
  - `TDBT2-G`: one quantized-point gradient 候选 + support transport + barrier。
- Metrics:
  - Primary: held-out `D_QK` 和 `D_VO` 相对 `PTQ-direct` 的百分比改善。
  - Decisive comparison: `TDBT2-F/G` vs `endpoint-beam`。
  - Secondary: local output NMSE、max path barrier、accepted swap count、zero-rate drift、wall-clock、forward/backward count、peak VRAM。
- Setup details:
  - group size 128；
  - threshold factor 0.7；
  - beam width 4；
  - max accepted swaps per layer/operator 64；
  - candidate pool 512 donor/receiver pairs；
  - local trust `tau = 1.05`；
  - final alpha refit exactly 1 pass；
  - no QAT artifacts loaded。
- Success criterion:
  - Diagnostic pass: `TDBT2-G` or `TDBT2-F` improves held-out operator distortion over `endpoint-beam` by >=5% in at least 3/4 layer-operator pairs, and untouched-W direction does not reverse。
  - Paper-worthy direction: same as above, plus untouched-C direction agrees when C4 is available, and wall-clock <= direct PTQ 3x。
- Failure interpretation:
  - If endpoint-beam matches TDBT2: path barrier is not needed。
  - If QG-one-shot matches TDBT2: multi-step transport is unnecessary。
  - If fit improves but val/untouched fails: support transport overfits calibration。
  - If sign-flip-only matches support-swap: ternary support claim is weak。
- Table / figure target: main Table 1 candidate；path-barrier vs endpoint scatter for Figure 2。
- Priority: MUST-RUN。

### Block B2: Ternary Mechanism Isolation

- Claim tested: 收益是否来自三值 support，而不是通用 sign / low-bit search。
- Dataset / split / task: 只在 B1 中改善最稳定的 2 个 layer-operator pair 上运行，仍使用预注册 split。
- Compared systems:
  - `support-swap`: donor nonzero -> zero + receiver zero -> nonzero。
  - `sign-flip-only`: nonzero sign flip，不改变 support。
  - `no-barrier support-swap`: 去掉 path trust，仅 endpoint。
  - `random budget-matched support-swap`: same accepted count。
  - `binary-like no-zero control`: 禁止 zero receiver，只允许 active sign updates。
- Metrics: D_QK/D_VO、path barrier、local NMSE、accepted count、cost。
- Setup details: 与 B1 同候选预算；不得新增 search pass。
- Success criterion: support-swap 在 held-out operator distortion 上优于 sign-flip-only 和 binary-like control >=5%，且 no-barrier 版本不能完全复现。
- Failure interpretation:
  - 若 sign-flip-only 同等好，TDBT2 不能主张三值 support 特性。
  - 若 no-barrier 同等好，删除 path-planning claim，仅保留 endpoint support selection。
- Table / figure target: ablation table。
- Priority: MUST-RUN if B1 diagnostic pass，否则不运行。

### Block B3: Cost And Overfitting Audit

- Claim tested: 方法仍属于 PTQ 级，而不是隐性 calibration training。
- Dataset / split / task: 汇总 B1/B2。
- Compared systems: PTQ-direct、QG-one-shot、endpoint-beam、TDBT2-F/G。
- Metrics: wall-clock ratio、forward count、backward count、peak VRAM、fit-val gap、untouched degradation。
- Success criterion: strict variant wall-clock <= direct PTQ 3x；不保存或读取 QAT artifacts；untouched-W NLL 不比 endpoint-beam 恶化超过 0.02。
- Failure interpretation: 若超过成本门槛，TDBT2 降级为重型 reconstruction PTQ，不作为主方法。
- Table / figure target: appendix cost table；main text 用一句话报告。
- Priority: MUST-RUN。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 写/检查 `tdbt2_support_transport_4090.py` | local lint + CPU dry-run if possible | 参数、split、no-QAT invariant 全部可审计 | 0 GPU h | 实现复杂度 |
| M1 | OPT-125M sanity | TDBT2-01A | finite + reload parity + metric changes after patch | 10-25 min | metric bug |
| M2 | OPT-350M primary | TDBT2-02A | B1 diagnostic pass or clear fail | 45-120 min | C4 unavailable / 4090 OOM |
| M3 | Mechanism ablation | TDBT2-02B | B2 support-specific pass | 30-90 min | ablation too noisy |
| M4 | Audit and report | TDBT2-02-report | nonfinite/split/cost/result completeness checked | CPU only | overclaiming |

## First Runs To Prepare

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status |
|---|---|---|---|---|---|---|---|
| TDBT2-01A | M1 | support-transport harness sanity | OPT-125M layer0 Q/K,V/O | W2 train/validation | finite, D_QK, D_VO, parity, VRAM | MUST | READY_NOT_STARTED |
| TDBT2-02A | M2 | main support-transport feasibility | OPT-350M layers 0/7/15/23 | fit-A/val-B/untouched-W/C | D_QK, D_VO, barrier, cost | MUST | BLOCKED_ON_TDBT2_01A |
| TDBT2-02B | M3 | ternary mechanism ablation | best 2 layer-operator pairs from B1 chosen by pre-rule | same as B1 | support-vs-sign, no-barrier, binary-like | CONDITIONAL | BLOCKED_ON_B1_PASS |

Pre-rule for selecting B2 pairs:

```text
Select the two layer-operator pairs with largest val-B improvement of TDBT2-G over endpoint-beam,
but only among pairs whose untouched-W direction is non-negative.
If fewer than two pairs satisfy this, do not run B2.
```

## Compute And Data Budget

- Total must-run if B1 fails: about 1-2.5 GPU-hours。
- Total if B1 passes and B2 runs: about 2-4 GPU-hours。
- Peak VRAM target: <=21.5 GiB on RTX 4090。
- Data: Wikitext-2 required；C4 optional but preferred. If C4 loading fails, record as missing external dataset and do not replace with synthetic text。
- Human input: no hyperparameter decision during run。

## Fresh Falsification Gate

This experiment rejects the current TDBT2 support-transport idea if any of the following holds:

1. `endpoint-beam` matches or beats `TDBT2-F/G` on val-B and untouched-W；
2. `QG-one-shot` matches or beats `TDBT2-F/G` under the same backward budget；
3. improvements appear only on fit-A；
4. support-swap does not beat sign-flip-only or binary-like no-zero control in B2；
5. strict variant needs QAT artifacts or exceeds direct PTQ 3x wall-clock。

If rejected, the next research move is not to tune `tau` or beam width. The correct move is to close the support-transport method claim and return to either calibration-data alignment or a different ternary-native mechanism.

## Final Checklist

- [x] Main claim is independent of old R014-R058 gates。
- [x] No QAT teacher, logits, checkpoint, latent weights, or state prior。
- [x] The experiment tests support transport, not old zero-mediated sign-flip as default。
- [x] Endpoint-only and one-shot gradient are mandatory baselines。
- [x] Fit/validation/untouched split is explicit。
- [x] Cost and overfitting audit are part of the gate。

