# Experiment Tracker: TDBT Clean-Room

日期：2026-08-26

状态：未启动新实验。本 tracker 只用于 clean-room TDBT 后续验证，不承接 R014-R058 或早期 `G4090-TDBT-*` 的 gate。

## Namespace Rules

- 新 run id 使用 `TDBT2-*`。
- 新结果目录使用 `results/remote-runs/TDBT2-*`。
- 每个 run 必须在启动前写明 `Legacy evidence used` 和 `Fresh falsification gate`。
- 旧实验只能作为 reference，不作为 pass/fail 的一部分。

## Runs

| Run ID | Purpose | Legacy evidence allowed | Fresh gate | Status | Notes |
|---|---|---|---|---|---|
| TDBT2-00 | 建立 clean-room 隔离协议 | None | 文档完成；后续实验不得从旧 TDBT plan 直接启动 | PASS_DOC | 见 `TDBT_CLEANROOM_PROTOCOL.md` |
| TDBT2-01 | 独立重写 TDBT method thesis | 只允许引用 Li2026 理论动机、correct CE harness 约束和 4090 资源边界 | 方法先从三值支撑/极性结构推出，不由旧结果倒推 | PASS_DOC | `METHOD_NOTE_TDBT2_SUPPORT_TRANSPORT_20260826.md` |
| TDBT2-02 | 预注册最小机制实验 | 只允许引用旧结果作为风险提示 | 新 plan 明确 split、metric、baselines、cost gate、stop condition | READY_NOT_STARTED | `EXPERIMENT_PLAN_TDBT2_SUPPORT_TRANSPORT_4090.md` |
| TDBT2-01A | support-transport harness sanity | correct CE harness、4090 资源边界 | OPT-125M layer0 Q/K,V/O finite + metric/parity sanity | PASS_SANITY | Wikitext Arrow cache；finite；no QAT artifacts；patch changed metrics；FP val NLL 3.9316、PTQ val NLL 9.6132 |
| TDBT2-02A | support-transport primary feasibility | 只使用 split/metric/显存经验作为风险提示 | OPT-350M layers 0/7/15/23；TDBT2-F/G vs endpoint-beam/QG-one-shot | FAIL_PATH_GATE | Complete in 242.4s；Wikitext Arrow cache；no QAT artifacts；TDBT2-F/G identical to endpoint-greedy on 8/8 pairs；see `EXPERIMENT_ANALYSIS_TDBT2_02A_20260826.md` |
| TDBT2-02B | ternary mechanism ablation | B1 结果只用于按预规则选择 2 个 pair | support-swap beats sign-flip/no-barrier/binary-like controls | NOT_RUN_GATE_FAIL | B1 path/barrier gate failed；do not run under this plan |

## Legacy Quarantine Summary

早期 `G4090-TDBT-00/01/02*` 和 R014-R058 均保持为 historical/diagnostic artifacts。它们可以回答“为什么要小心”，不能回答“新 TDBT 是否成立”。

## Post-TDBT Pivot Runs

| Run ID | Purpose | Legacy evidence allowed | Fresh gate | Status | Notes |
|---|---|---|---|---|---|
| TQGSP-01A | 验证量化点梯度三值支撑投影；同时测试 ternary specificity、NLL transfer 与 cost | 只允许引用 TDBT2-02A 的负结论作为转向依据 | `TQGSP-support-G` 在 untouched operator NMSE 上优于 random/forward/NZ-signflip；patched NLL 不明显劣化；输出成本分解 | DIAGNOSTIC_CONTROL_MISMATCH | 完成；发现 signflip 64 edits vs support-G 11–14 edits，不作机制结论 |
| TQGSP-01B | 修正 01A 的 edit-budget mismatch；公平比较 support-swap 与 signflip/random/forward | 只允许使用 01A 作为 harness diagnostic | matched 64-edit budget 下重新执行同一 gate | PASS_MECHANISM_MIXED_TRANSFER | 完成；operator gate 3/4 pass；untouched NLL -0.00558，val NLL +0.00160；见 `EXPERIMENT_ANALYSIS_TQGSP_01B_20260826.md` |
| TQGSP-02A | 检验 operator gain 能否预测 CE/NLL；做 CE-aware Q/K 层选择 | 只允许使用 01B 作为机制依据 | CE-selected patch set 在 untouched NLL 上不劣于 direct PTQ；operator gain 与 val NLL delta 关系可解释 | PASS_CE_GATE_FAIL_PROXY | 完成；operator proxy 相关性近 0 且 wrong-signed；ce-selected untouched NLL -0.0157；见 `EXPERIMENT_ANALYSIS_TQGSP_02A_20260826.md` |
| CEGSP-01A | 直接使用部署三值点 CE 梯度做 support projection，并与 CE signflip control 比较 | 只允许使用 TQGSP-02A 的 proxy 失败作为转向依据 | support-selected 改善 untouched NLL；与 signflip-selected 比较决定是否 support-only 或 joint sign/support | PASS_TOPK_STRONG_ALL_LAYER_FAIL | 完成；support-topk untouched NLL -0.2720，signflip-topk -0.2701，全层叠加劣化；见 `EXPERIMENT_ANALYSIS_CEGSP_01A_20260826.md` |
| CEGSP-01B | k-sweep：support-only、signflip-only、joint-best 的层预算稳定性 | 只允许使用 CEGSP-01A 的 top-k 强正与 all-layer 失败作为动机 | 多个 k 改善 untouched NLL；joint/support/signflip 胜负决定方法形态 | PASS_JOINT_TOPK_BEST | 完成；joint top-6 untouched NLL -0.2807 最优；k=24 全部劣化；见 `EXPERIMENT_ANALYSIS_CEGSP_01B_20260826.md` |
| CEGSP-02A | split/offset 稳定性验证，防止单次结果牵引方向 | 只允许使用 CEGSP-01B 的 top-k 强正作为待验证假设 | 3 个 offset 上至少一个 family/k 稳定改善 untouched NLL | PASS_ROBUST_SMALL_K | 完成；k=4/6 在 3 个 offset 上稳定改善；joint 有价值但不总是最优；见 `EXPERIMENT_ANALYSIS_CEGSP_02A_20260826.md` |
| CEGSP-03A | WikiText 选择到 C4 untouched 的跨数据迁移验证 | 只允许使用 CEGSP-02A 的稳定 k=4/6；C4 不参与选择 | joint top4/top6 在 val、WikiText untouched、C4 untouched 均不退化 | PASS_C4_TRANSFER_O0 | 完成；joint top6 W2 untouched -0.280708、C4 untouched -0.174876；全层编辑仍退化；见 `EXPERIMENT_ANALYSIS_CEGSP_03A_20260826.md` |
| CEGSP-03B | 在 O1/O2 复现 C4 transfer，避免 O0 单点结论 | 完全复用 03A 方法与 k=4/6；不根据 C4 选层或调参 | O1/O2 上至少一个 joint top-k 同时改善 val/W2/C4 | PASS_C4_TRANSFER_3OFFSETS | 完成；O0/O1/O2 全部 top-k family 均 3/3 改善 W2 与 C4；all-layer 仍不稳定；见 `EXPERIMENT_ANALYSIS_CEGSP_03B_20260826.md` |
| CEGSP-04A | edit-budget/cost sensitivity，检查 64 edits 是否偶然 | 固定 O0，扫 max-edits 16/32/64/128；k=4/6；C4 只报告 | 非单点稳定预算区间，top-k 同时改善 val/W2/C4 | PASS_BUDGET_ROBUST | 完成；所有 top-k family 在全部预算点均改善，runtime 49.47–54.16s；all-layer 高预算退化；见 `EXPERIMENT_ANALYSIS_CEGSP_04A_20260826.md` |
| CEGSP-04B | OPT-125M second-model sanity check | 缓存模型；12 层；比例预算 k=2/3；O0/O1/O2；C4 只报告 | 至少一个 family/k 在 2/3 offset 同时改善 val/W2/C4 | PASS_SECOND_MODEL | 完成；所有 top-k family 3/3 改善；joint top3 aggregate W2 -0.253209、C4 -0.301906；all-layer 在 125M 也改善但不作为 scale-robust 默认；见 `EXPERIMENT_ANALYSIS_CEGSP_04B_20260826.md` |
| CEGSP-05A | larger untouched holdout，排除 8-batch 小样本偶然 | O0；W2/C4 untouched 从 8 增到 32 batches；OPT-350M 与 OPT-125M | 每个模型至少一个 top-k family 同时改善 val/W32/C4-32 | PASS_LARGER_HOLDOUT | 完成；joint top6/top3 均通过，OPT-350M joint top6 W32 -0.255838、C4-32 -0.210322；见 `EXPERIMENT_ANALYSIS_CEGSP_05A_20260826.md` |
| CEGSP-05B | random-control，区分 CE-gradient 机制与任意随机三值编辑 | O0；同 05A 的 W2/C4 32-batch holdout；两模型；每模型 3 次 random repeat | CE joint top-k 同时优于 random joint 均值（W32 与 C4-32） | PASS_CE_OVER_RANDOM | 完成；OPT-350M k=4/6、OPT-125M k=2/3 均通过，CE 相对 random 均值优势 0.156–0.410 NLL；见 `EXPERIMENT_ANALYSIS_CEGSP_05B_20260826.md` |
| CEGSP-06A | matched controls，拆分 CE 选层/类型收益与 CE 层内候选收益 | O0；OPT-350M；同 05B 的 W2/C4 32-batch holdout；3 次 random repeat | CE joint 优于 random joint；matched controls 能解释 layer/type 与 candidate 两部分贡献 | PASS_MATCHED_MECHANISM | 完成；CE joint top6 W32 -0.255838、C4 -0.210322；random candidates on CE layers 近 0，CE candidates on random layers 仍有收益；见 `EXPERIMENT_ANALYSIS_CEGSP_06A_20260827.md` |
| CEGSP-07A | ternary specificity，同层比较 zero-support relocation 与 nonzero-only signflip | O0；OPT-350M；W2/C4 32-batch holdout；k=4/6；3 次 random repeat | CE joint 继续通过；support/signflip same-layer 对照决定是否可声称 zero-support relocation 是核心机制 | PASS_TERNARY_MODULE | 完成；same-layer support 在 k=4/6 的 W32/C4 上均优于 signflip，但 joint 仍最合理；见 `EXPERIMENT_ANALYSIS_CEGSP_07A_20260827.md` |
| CEGSP-07B | second-model ternary specificity，在 OPT-125M 上复现 07A 同层 support/signflip 对照 | O0；OPT-125M；W2/C4 32-batch holdout；k=2/3；3 次 random repeat | CE joint 继续通过；same-layer support/signflip 对照检查三值模块是否跨模型 | PASS_SECOND_MODEL_TERNARY_MODULE | 完成；OPT-125M 上 same-layer support 在 k=2/3 的 W32/C4 上均优于 signflip，CE joint 仍最优；见 `EXPERIMENT_ANALYSIS_CEGSP_07B_20260827.md` |
| CEGSP-08A | downstream sanity，检查 NLL 改善是否在 cloze last-token accuracy 上不退化/有信号 | OPT-125M/350M；W2/C4 32-batch holdout；LAMBADA-style 128 examples | CE joint 保持 NLL 改善且 cloze top1/top5 不双双退化 | DIAGNOSTIC_FLOOR_EFFECT | 完成；direct ternary 的 LAMBADA top1/top5 已归零，hard accuracy 无分辨率；350M cloze NLL 改善，125M cloze NLL 略差；见 `EXPERIMENT_ANALYSIS_CEGSP_08A_20260827.md` |
| CEGSP-09A | scale validation，停止继续小消融，在 OPT-1.3B 上验证 CEGSP 是否仍优于 direct ternary | OPT-1.3B；W2/C4 32-batch holdout；k=8/12；无 random/cloze | 至少一个 CE joint top-k 同时改善 W32 与 C4，且 4090 成本可接受 | PASS_SCALE | 完成；joint top8/top12 均改善 W32 与 C4；support relocation 仍强于 signflip；runtime 130.85s、peak 4.75GB；见 `EXPERIMENT_ANALYSIS_CEGSP_09A_20260827.md` |
| CEGSP-09B | larger scale validation，在 OPT-2.7B 上验证 CEGSP 是否仍优于 direct ternary | OPT-2.7B；W2/C4 24-batch holdout；k=12/16；无 random/cloze | 至少一个 CE joint top-k 同时改善 W24 与 C4，且 4090 成本可接受 | PASS_LARGER_SCALE | 完成；joint top16 W24 -0.731652、C4-24 -0.649337；runtime 199.14s、peak 7.96GB；见 `EXPERIMENT_ANALYSIS_CEGSP_09B_20260827.md` |
| CEGSP-10A | architecture adapter + cross-family validation，在 GPT-NeoX/Pythia-1B 上验证 CEGSP 是否迁移 | Pythia-1B；fused QKV row-slice adapter；W2/C4 24-batch holdout；k=4/8；strict PTQ | 至少一个 CE joint top-k 同时改善 W24 与 C4，adapter metadata、finite metrics 与 4090 cost 完整 | PASS_CROSS_ARCH | 正式复跑通过；joint top4 W24 -0.590785、C4-24 -0.386891；runtime 64.93s、peak 3.92GiB；见 `EXPERIMENT_ANALYSIS_CEGSP_10A_20260827.md` |
