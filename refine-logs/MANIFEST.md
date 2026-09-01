# Refine Logs Manifest

| Date | Artifact | Purpose |
|---|---|---|
| 2026-08-24 | `EXPERIMENT_PLAN_QAT_GAUGE_4090_20260824_174500.md` | 4090 版 claim-driven 实验计划快照 |
| 2026-08-24 | `EXPERIMENT_PLAN_QAT_GAUGE_4090.md` | 4090 版实验计划固定入口 |
| 2026-08-24 | `EXPERIMENT_TRACKER_QAT_GAUGE_4090.md` | 新方向执行 tracker |
| 2026-08-24 | `EXPERIMENT_TRACKER.md` | 追加新方向索引，保留 R014–R058 历史 |
| 2026-08-24 | `EXPERIMENT_PLAN_TERNARY_GAP_4090_20260824_200122.md` | 三值 QAT–PTQ gap 证明、归因与弥合方案快照 |
| 2026-08-24 | `EXPERIMENT_PLAN_TERNARY_GAP_4090.md` | 三值 gap 方案固定入口 |
| 2026-08-24 | `EXPERIMENT_TRACKER_TERNARY_GAP_4090_20260824_200122.md` | 三值 gap 方案执行 tracker 快照 |
| 2026-08-24 | `EXPERIMENT_TRACKER_TERNARY_GAP_4090.md` | 三值 gap 方案固定 tracker |
| 2026-08-26 | `EXPERIMENT_PLAN_TDBT_4090_20260826_102533.md` | Ternary Discrete Basin Transport 4090 方案快照 |
| 2026-08-26 | `EXPERIMENT_PLAN_TDBT_4090.md` | TDBT 方案固定入口 |
| 2026-08-26 | `EXPERIMENT_TRACKER_TDBT_4090_20260826_102533.md` | TDBT 方案执行 tracker 快照 |
| 2026-08-26 | `EXPERIMENT_TRACKER_TDBT_4090.md` | TDBT 方案固定 tracker |
| 2026-08-26 | `EXPERIMENT_LOG_G4090_20260826.md` | 42137 RTX 4090 云端 TDBT 首轮原始结果与方向判断 |
| 2026-08-26 | `Li2026_PAPER_DIGEST_AND_TDBT_UPDATE.md` | Li et al. 2026 论文理解、与现有实验的对齐审计及下一轮修正 |
| 2026-08-26 | `TDBT_EVIDENCE_NEXT_STEPS_20260826.md` | TDBT 当前可靠证据、不可支持 claim、思路修改与下一步最小验证计划 |
| 2026-08-26 | `TDBT_CLEANROOM_PROTOCOL_20260826.md` | TDBT clean-room 隔离协议快照：旧实验只作参考，不驱动新 idea/gate |
| 2026-08-26 | `TDBT_CLEANROOM_PROTOCOL.md` | TDBT clean-room 隔离协议固定入口 |
| 2026-08-26 | `EXPERIMENT_TRACKER_TDBT_CLEANROOM.md` | TDBT2 clean-room 后续实验 tracker，隔离旧 R014-R058/G4090 gates |
| 2026-08-26 | `METHOD_NOTE_TDBT2_SUPPORT_TRANSPORT_20260826.md` | TDBT2 support-transport clean-room 方法锚点 |
| 2026-08-26 | `EXPERIMENT_PLAN_TDBT2_SUPPORT_TRANSPORT_4090_20260826.md` | TDBT2 support-transport 4090 下一步实验计划快照 |
| 2026-08-26 | `EXPERIMENT_PLAN_TDBT2_SUPPORT_TRANSPORT_4090.md` | TDBT2 support-transport 4090 下一步实验计划固定入口 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_TDBT2_02A_20260826.md` | TDBT2-02A 结果分析：support-swap 有信号但 path/barrier gate 失败 |
| 2026-08-26 | `EXPERIMENT_PLAN_TQGSP_01A_4090_20260826.md` | TQG-SP 最小验证计划：ternary support projection、NLL transfer 与成本分解 |
| 2026-08-26 | `EXPERIMENT_PLAN_TQGSP_01B_BUDGET_MATCHED_4090_20260826.md` | TQG-SP 预算匹配修正版验证计划：修正 01A control edit 数不公平问题 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_TQGSP_01B_20260826.md` | TQGSP-01B 结果分析：budget-matched support projection 机制通过，NLL 迁移弱正/混合 |
| 2026-08-26 | `EXPERIMENT_PLAN_TQGSP_02A_CE_SELECTION_4090_20260826.md` | TQGSP-02A 预注册计划：检验 operator proxy 到 CE/NLL 的层选择迁移 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_TQGSP_02A_20260826.md` | TQGSP-02A 结果分析：operator proxy 失败，CE-aware selection 成功，建议转向 CE-gradient support projection |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_01A_4090_20260826.md` | CEGSP-01A 预注册计划：直接用部署三值点 CE 梯度做 support projection，并比较 signflip control |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_01A_20260826.md` | CEGSP-01A 结果分析：CE-gradient top-k 编辑强正，全层叠加失败，提示需要 k-sweep 与 joint edit selection |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_01B_KSWEEP_4090_20260826.md` | CEGSP-01B 预注册计划：support/signflip/joint 的 k-sweep 层预算稳定性实验 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_01B_20260826.md` | CEGSP-01B 结果分析：joint top-6 最优，确认 CE-gradient joint ternary edit selection 是当前最佳方向 |
| 2026-08-26 | `RESEARCH_DIRECTION_LOCK_CEGSP_20260826.md` | PTQ 1.58-bit 研究方向冻结协议：防止单次实验导致过度转向或钻牛角尖 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_02A_ROBUSTNESS_4090_20260826.md` | CEGSP-02A 预注册计划：3 个 calibration/validation offset 的稳健性验证 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_02A_20260826.md` | CEGSP-02A 结果分析：3 offset 稳健性通过，k=4/6 为稳定预算区间 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_03A_C4_TRANSFER_4090_20260826.md` | CEGSP-03A 预注册计划：WikiText 选择到 C4 untouched 的跨数据迁移验证 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_03A_20260826.md` | CEGSP-03A 结果分析：joint top4/top6 跨到 C4 仍改善，全层编辑继续失败 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_03B_C4_TRANSFER_OFFSETS_4090_20260826.md` | CEGSP-03B 预注册计划：在 offset 4096/8192 上复现 WikiText-to-C4 transfer |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_03B_20260826.md` | CEGSP-03B 结果分析：三 offset 下 top-k CE-gradient 三值编辑均迁移到 C4，全层编辑不稳定 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_04A_EDIT_BUDGET_4090_20260826.md` | CEGSP-04A 预注册计划：扫描 max-edits 16/32/64/128 的预算-收益曲线 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_04A_20260826.md` | CEGSP-04A 结果分析：top-k 对 edit budget 不敏感，高预算 all-layer 编辑退化 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_04B_OPT125M_4090_20260826.md` | CEGSP-04B 预注册计划：在缓存的 OPT-125M 上做第二模型 sanity check |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_04B_20260826.md` | CEGSP-04B 结果分析：OPT-125M 三 offset 全部通过，支持第二模型诊断但暴露 all-layer scale interaction |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_05A_LARGER_HOLDOUT_4090_20260826.md` | CEGSP-05A 预注册计划：将 untouched W2/C4 从 8 batches 扩到 32 batches 做大样本复核 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_05A_20260826.md` | CEGSP-05A 结果分析：32-batch untouched 下两模型 top-k 仍通过，下一步需 random/edit-control NLL 对照 |
| 2026-08-26 | `EXPERIMENT_PLAN_CEGSP_05B_RANDOM_CONTROL_4090_20260826.md` | CEGSP-05B 预注册计划：同预算 random support/signflip + validation top-k 强对照 |
| 2026-08-26 | `EXPERIMENT_ANALYSIS_CEGSP_05B_20260826.md` | CEGSP-05B 完整性审计与随机对照分析：两模型、两档 k 均通过 CE joint 优于 random joint 均值的主 gate |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_06A_MATCHED_CONTROLS_4090_20260827.md` | CEGSP-06A 预注册计划：拆分 CE 选层/类型与层内候选质量的 matched controls |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_06A_20260827.md` | CEGSP-06A 结果分析：CE 层内候选质量是主要收益来源，CE 选层/类型进一步增强 |
| 2026-08-27 | `CEGSP_TWO_DAY_SYNTHESIS_AND_NEXT_RUN_20260827.md` | CEGSP 两天实验结果综合：当前支持 claim、缺口与 07A 下一步 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_07A_TERNARY_SPECIFICITY_4090_20260827.md` | CEGSP-07A 预注册计划：同层比较 support relocation 与 nonzero-only signflip |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_07A_20260827.md` | CEGSP-07A 结果分析：同层 support relocation 优于 nonzero-only signflip，支持三值特异性模块 claim |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_07B_OPT125M_TERNARY_SPECIFICITY_4090_20260827.md` | CEGSP-07B 预注册计划：在 OPT-125M 上复现 support/signflip 同层三值特异性对照 |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_07B_20260827.md` | CEGSP-07B 结果分析：OPT-125M 复现同层 support relocation 优于 nonzero-only signflip |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_08A_CLOZE_SANITY_4090_20260827.md` | CEGSP-08A 预注册计划：两模型 LAMBADA-style last-token cloze sanity eval |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_08A_20260827.md` | CEGSP-08A 结果分析：LAMBADA hard accuracy 出现 floor effect，350M cloze NLL 改善但不能声明 accuracy gain |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_09A_OPT13B_SCALE_4090_20260827.md` | CEGSP-09A 预注册计划：OPT-1.3B 上做规模验证，停止继续小消融 |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_09A_20260827.md` | CEGSP-09A 结果分析：OPT-1.3B scale gate 通过，joint top-k 改善 Wikitext/C4，support relocation 仍为核心模块 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_09B_OPT27B_SCALE_4090_20260827.md` | CEGSP-09B 预注册计划：OPT-2.7B 上继续做大模型规模验证 |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_09B_20260827.md` | CEGSP-09B 结果分析：OPT-2.7B larger scale gate 通过，joint top16 大幅改善 Wikitext/C4 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_10A_PYTHIA1B_CROSS_ARCH_4090_20260827.md` | CEGSP-10A 预注册计划：GPT-NeoX/Pythia-1B fused-QKV architecture adapter 跨架构验证 |
| 2026-08-27 | `EXPERIMENT_ANALYSIS_CEGSP_10A_20260827.md` | CEGSP-10A 正式复跑分析：Pythia-1B cross-family gate 通过，joint top4 改善 Wikitext/C4 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_PAPER_20260827_093516.md` | 论文级 CEGSP 总验证方案：主表、QAT gap、三值特异性、泛化、下游和成本闭环 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_PAPER.md` | 论文级 CEGSP 总验证方案最新副本 |
| 2026-08-27 | `EXPERIMENT_TRACKER_CEGSP_PAPER_20260827_093516.md` | 论文级 CEGSP 分组实验 tracker |
| 2026-08-27 | `EXPERIMENT_TRACKER_CEGSP_PAPER.md` | 论文级 CEGSP 分组实验 tracker 最新副本 |
| 2026-08-27 | `EXPERIMENT_PLAN_CEGSP_PAPER_20260827_094411.md` | 强基线审计与 direct/strong + CEGSP 2×2 组合验证修订版 |
| 2026-08-27 | `EXPERIMENT_TRACKER_CEGSP_PAPER_20260827_094500.md` | 强基线、组合 gate 与公平性审计 tracker 修订版 |
| 2026-08-27 | `AFTERNOON_BATCH_PLAN_CEGSP_20260827_095620.md` | 整合既有 CEGSP 证据与强三值 PTQ 对照的下午 2×2 验证批次 |
| 2026-08-27 | `AFTERNOON_BATCH_PLAN_CEGSP.md` | 下午论文级验证批次固定入口 |
| 2026-08-27 | `CEGSP_EXPERIMENT_SYNTHESIS_20260827.md` | 云端扫描 `/root/tqgsp-runs/*/result.json` 后生成的 CEGSP/TQGSP/TDBT 实验整理报告 |
| 2026-08-27 | `CEGSP_EXPERIMENT_SUMMARY_20260827.csv` | 云端生成并同步回本地的逐 run 指标、成本与路径 CSV 汇总 |
| 2026-08-27 | `CEGSP_11A_11A2_STRONG_BASELINE_AUDIT_20260827.md` | OPT-350M 上 official PT² ATQ/ATQ+SSR 的 matched compact 与 long-calib 强基线审计结果 |
| 2026-08-27 | `EXPERIMENT_PLAN_NEXT_CEGSP_PT2_20260827_142217.md` | CEGSP 作为独立三值 PTQ 与 PT² 的官方协议公平比较方案，等待人工审核 |
| 2026-08-27 | `EXPERIMENT_PLAN_NEXT_CEGSP_PT2.md` | 上述下一阶段实验方案最新副本 |
| 2026-08-27 | `EXPERIMENT_TRACKER_NEXT_CEGSP_PT2_20260827.md` | CEGSP vs PT² 分阶段实验 tracker，全部等待人工审核 |
| 2026-08-27 | `CEGSP_12A_OFFICIAL_BASELINE_AUDIT_20260827.md` | OPT-350M 上 PT² native 128×2048 复现、干净 FP16 reference 与异常结果审计 |
| 2026-09-01 | `EXPERIMENT_PLAN_CEGSP_P9D1_RESIDUAL_LANDSCAPE_20260901.md` | P9-D1 预注册：ordinary affine 与真实 PT² 初始化的单步残差离散景观诊断 |
| 2026-09-01 | `EXPERIMENT_ANALYSIS_CEGSP_P9D1_RESIDUAL_LANDSCAPE_20260901.md` | P9-D1 原始结果与完整性分析：PT² 后 W2 可利用残差减少；rho 对两种 initializer 均偏弱，不触发 D2 |
| 2026-09-01 | `results/remote-runs/cegsp_p9d1_residual_landscape_llama2_7b_a100_20260901_42028/p9d1_result.json` | A100 P9-D1 1024 个单步 candidate 的原始 JSON 结果 |
| 2026-09-01 | `EXPERIMENT_PLAN_CEGSP_P8A_ONLY_PREP_20260901.md` | P8-only downstream 准备协议：六任务 screen、冻结规则、42079 服务器启动前 gate |
| 2026-09-01 | `remote-tools/run_p8a_downstream.sh` | P8-only 远端 check/launch wrapper；默认 check，不自动启动正式实验 |
| 2026-08-27 | `results/remote-runs/CEGSP-12A-OFFICIAL-OPT350M/result.json` | CEGSP-12A 官方 ATQ/ATQ+SSR 聚合配置与结果 |
| 2026-08-27 | `results/remote-runs/CEGSP-12A-FP16-CLEAN-OPT350M/result.json` | 官方数据/evaluator 下的干净 FP16 reference |
