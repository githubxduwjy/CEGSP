# CEGSP-12A：PT² 官方协议复现与 baseline 审计

**日期：** 2026-08-27  
**云端：** `xj-member.bitahub.com:42120`  
**GPU：** NVIDIA RTX 4090 24GB  
**模型：** `facebook/opt-350m`  
**PT² commit：** `9e943e6`

## 1. 实验目的

本实验只复现 PT² 官方 ATQ/ATQ+SSR 流程，不运行 CEGSP，也不把该结果解释为 CEGSP 与 PT² 的性能比较。目的为锁定官方 baseline 的数据、校准、评测和代码行为。

使用官方数据准备脚本生成：

- WikiText-2 train/test；
- C4 train/validation；
- 数据目录：`/root/PT2-LLM-full/pt2_llm/data`。

## 2. 预注册配置核验

| 项目 | 配置 |
|---|---|
| calibration dataset | WikiText-2 |
| `nsamples` | 128 |
| `calib_seqlen` | 2048 |
| `ppl_seqlen` | 2048 |
| `blocksize` | 128 |
| `percdamp` | 0.01 |
| dtype | PT² 自动加载为 FP16 |
| 量化范围 | 24 decoder layers / 144 decoder Linear modules |
| ATQ | 官方 `atq`，即 ITF + AGA |
| SSR | 官方 `GPTQ_SSR` |
| QAT teacher | 无 |
| optimizer step | 无 |

云端 Transformers 为 4.46.3，而 PT² 仓库的 OPT 前向调用带有当前 OPT layer 不接受的 `position_embeddings` 参数。因此使用了单独的 `pt2_official_native_compat.py` wrapper，仅移除该不支持的 keyword，未修改 ATQ、GPTQ 或 SSR 逻辑。

## 3. 结果

| system | WikiText-2 PPL | C4 PPL | quantization time | 状态 |
|---|---:|---:|---:|---|
| clean FP16 reference | 22.0046 | 22.5898 | 0.0 s | valid |
| official `quantize.py ... fp16` | 9970.2998 | 8861.2852 | 66.1 s | invalid reference |
| PT² ATQ | 13044.4258 | 11384.0186 | 67.4 s | finite but severe degradation |
| PT² ATQ+SSR | 15917.3828 | 13408.7412 | 83.9 s | finite but severe degradation |

所有 PT² PPL 均为 finite。ATQ+SSR 相比 ATQ 进一步恶化约 22.0%（WikiText-2）和 17.8%（C4）。

## 4. 关键实现发现

### 4.1 官方 `fp16` 入口不是干净 reference

官方 `quantize.py` 将 `fp16` 传给 `TernaryQuantizer`，但仍然进入 `quant_sequential`。由于 `gptaq=True`，GPTAQ 补偿分支仍会修改后续列，即使当前量化误差为零。因此该入口产生的 9970/8861 PPL 不能作为 FP16 基线。

干净 reference 通过以下方式获得：

- 使用官方 `get_model` 加载模型；
- 使用官方 `get_loaders` 加载测试数据；
- 使用官方 `opt_eval` 评测；
- 不调用 `quant_sequential`。

其 22.00/22.59 PPL 与 OPT-350M 的正常 FP16 量级一致，因此评测链路基本正常。

### 4.2 PT² 官方 native 结果仍然异常差

这次已经满足 128×2048 calibration 和 2048 PPL evaluation，因而排除了此前 compact protocol 中“校准 token 太少”的主要疑点。但 PT² ATQ 仍达到 13044/11384 PPL，ATQ+SSR 达到 15917/13409 PPL。

这说明当前问题不再能简单归因于 compact evaluation。仍需进一步确认：

- PT² 代码在 OPT-350M 上是否本来就不稳定；
- 官方论文主要结果是否集中在 LLaMA/Qwen 等模型，OPT 分支是否只是接口支持而非强结果范围；
- Transformers 4.46.3 的兼容 wrapper 是否需要更严格地复现官方 4.44.2 环境；
- PT² 的 GPTAQ/ATQ 实现是否存在与 OPT 架构相关的实现问题。

## 5. Gate 判定

| gate | 判定 | 说明 |
|---|---|---|
| 数据准备完成 | PASS | 官方 `prepare_data.py` 完成 WikiText-2/C4 save-to-disk |
| 128×2048 calibration | PASS | 日志和命令可核验 |
| 官方 ATQ 执行 | PASS | 24 层完成，finite |
| 官方 ATQ+SSR 执行 | PASS | 24 层完成，finite |
| clean FP16 reference | PASS | 独立 reference 为 22.00/22.59 |
| native PT² 强 baseline 复现 | FAIL | PT² 相对 clean FP16 严重退化 |
| CEGSP vs PT² 公平比较 | BLOCKED | PT² baseline 还需要定位异常，尚未运行 CEGSP |

## 6. 研究解释

本实验不能支持“CEGSP 优于 PT²”。它只支持以下结论：

1. PT² 官方代码可以在 4090 上按 native calibration/evaluation 流程运行；
2. 官方 `fp16` 入口不能直接当作 FP16 reference；
3. 当前 OPT-350M native PT² 结果异常差，baseline reproduction 尚未通过；
4. CEGSP 仍然是独立方法，不能因为 baseline 异常就被降级为 PT² 组件。

## 7. 后续限制

在 12A baseline reproduction 问题解决前，不启动 12B，也不运行 `PT²+CEGSP`。下一步若继续，必须先做一次明确的 environment/code-path diagnosis：优先使用 PT² requirements 中的 Transformers 4.44.2，或在当前环境中逐项对比官方模型前向、quantizer 输出和 GPTAQ compensation；不得用事后挑选的配置替代失败结果。

## 原始产物

- [native log](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/CEGSP-12A-OFFICIAL-OPT350M/CEGSP-12A-OFFICIAL-OPT350M.log)
- [clean FP16 result](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/CEGSP-12A-FP16-CLEAN-OPT350M/result.json)
- [aggregate result](/home/x1shan/文档/ChatGPT/PTQ_paper/results/remote-runs/CEGSP-12A-OFFICIAL-OPT350M/result.json)
