# P5-C0 实验完整性审计

日期：2026-08-28  
审计对象：[P5-C0 result.json](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/cegsp_p5c0_pt2_numeric_health_opt350m_20260828_42143/result.json)

## 外部审计状态

本轮请求的独立外部 reviewer 在约 300 秒内未返回，已终止；因此不冒充外部 reviewer 给出 verdict，外部审计状态为 `REVIEW_UNAVAILABLE`。以下是基于脚本、原始结果和预注册协议的确定性完整性检查，不能替代独立 reviewer。

## 确定性检查

### A. 数据来源：PASS

官方评测使用 PT² 的 Wikitext-2/C4 loader；compact 评测使用真实 Wikitext-2/C4 validation 数据。没有使用模型输出作为 ground truth，也没有使用 QAT teacher。

### B. 分数归一化：PASS

结果同时保存官方 PPL、compact 原始 NLL/PPL 和 layer/block 原始统计；没有使用模型自身 max/min/mean 对指标做归一化。

### C. 结果存在性与一致性：PASS

原始 JSON 存在，状态为 `complete`；记录了 clean FP16、ATQ、ATQ+SSR，144 个线性模块和 1728 个 block。报告中的数值均来自该 JSON。

### D. 指标调用链：PASS

脚本实际调用了 PT² `quant_sequential`、官方 `opt_eval` 和 compact `evaluate_nll`；layer/block finite、T 合法性和 output reconstruction 统计均写入结果文件。

### E. 范围：WARN

这是单模型、单 seed、两个官方状态的数值审计，不是跨模型或多 seed 的 strong-baseline 结论。后续文字不能使用“全面证明 PT² 实现均有问题”等超出范围的表述。

### F. 评测类型：self-supervised proxy

语言模型 next-token NLL/PPL 使用真实语料 token 作为自监督标签；它不是人工标注下游任务，也不是由另一个模型生成的 synthetic ground truth。

## 结论

确定性完整性状态：`PASS_WITH_SCOPE_WARNING`。  
外部 reviewer 状态：`REVIEW_UNAVAILABLE`。  
实验质量判定：`NUMERICAL_HEALTH_FAIL`，不是 phantom result 或 evaluator-only failure。

## Claim impact

- C1：P5-C0 已完整审计官方 OPT-350M PT² 状态的 finite、模块覆盖、T legality 和 evaluator direction —— **supported**。
- C2：当前 PT² reproduction 是健康且可作为主 strong baseline 的状态 —— **unsupported**。
- C3：CEGSP 在 direct/ordinary affine ternary initialization 上仍有独立证据 —— **由 P5-C0 不改变；应引用 P5-B/P5-C 既有边界**。
- C4：CEGSP 改善健康的 strong PT² —— **unsupported**。
