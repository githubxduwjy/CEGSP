# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 机器审计 | driver/CUDA/RAM/disk/network | Cloud 4090 | system inventory | MUST | DONE | 2026-08-21：RTX 4090 24564 MiB，driver 580.76.05；503 GiB RAM；128 CPU；GPU 空闲；root overlay 30 GiB，仅 3.9 GiB 可用，主要为现有 PT2 目录 22 GiB |
| R000A | M0 | 旧环境验收 | existing Qwen3-compatible PT² env | Qwen3-8B | imports/CUDA witness/history | MUST | DONE | commit 9e943e6；CUDA witness 通过；但 Transformers/Accelerate/Tokenizer 偏离官方 pin；历史 ATQ+SSR PPL=59.5678/204.4103 异常 |
| R001 | M0 | 环境冻结 | official PT² commit + pinned env | N/A | import/unit smoke | MUST | BLOCKED | 官方 spec 已写入 `.aris/compute`；须先腾出或挂载至少 15–20 GiB 持久空间，再建隔离 env |
| R003Q | M0 | 评估链诊断 | FP16 HF-standard sanity | Qwen3-8B; first 2×2048 tokens | CE/PPL/time/VRAM | MUST | DONE | PPL 9.2493；13s；peak 19654 MiB；产物 `results/R003Q_20260821_084250`；首次包装因缺 `/usr/bin/time` 0s 失败并保存 |
| R003QF | M0 | 评估链诊断 | FP16 PT² qwen_eval full | Qwen3-8B; full WikiText2 | PPL/time/VRAM | MUST | DONE | PPL 9.7278；54s；peak 7610 MiB；产物 `results/R003QF_20260821_084600`；说明历史 59.5678 主要是量化后退化，不是 FP16 评估链失效 |
| R002 | M0 | 数据准备 | WikiText2/C4/tokenizer | Dev model | sample hashes/counts | MUST | TODO | 不在日志打印 token |
| R003 | M0 | 评估校验 | FP16 | TinyLlama 1.1B | W2/C4 PPL | MUST | TODO | 对照 HF 原始模型 |
| R004 | M0 | 量化 smoke | ternary-init | TinyLlama; 8×256 | PPL/VRAM/time | MUST | TODO | 最小可运行配置 |
| R004Q | M0 | Qwen 量化 smoke | ternary-init | Qwen3-8B; 8×512 | W2/C4 PPL/VRAM/time | MUST | DONE | PPL 337833.5312/214931.8281；265s；peak 11776 MiB；产物 `results/R004Q_20260821_084830`；朴素三值化完全崩溃 |
| R005 | M0 | ATQ smoke | ATQ no SSR | TinyLlama; 16×512 | PPL/VRAM/time | MUST | TODO | ITF+AGA 路径 |
| R005Q | M0 | Qwen ATQ smoke | ATQ no SSR | Qwen3-8B; 8×512 | W2/C4 PPL/VRAM/time | MUST | DONE | PPL 817.2192/2663.9712；296s；peak 11198 MiB；产物 `results/R005Q_20260821_085500`；显著优于 ternary-init 但仍不可用 |
| R006 | M0 | full smoke | ATQ + SSR | TinyLlama; 16×512 | PPL/VRAM/time | MUST | TODO | 验证保存/加载 |
| R006Q | M0 | Qwen full smoke | ATQ + SSR | Qwen3-8B; 8×512 | W2/C4 PPL/VRAM/time | MUST | DONE | PPL 1077.3230/2415.8220；785s；peak 11198 MiB；产物 `results/R006Q_20260821_090130`；同预算下 W2 劣于 no-SSR，C4 小幅改善，不能用小校准集代替 full baseline |
| R007 | M1 | 显存剖析 | PT² full | LLaMA-2-7B; 8×512 | peak VRAM/RAM | MUST | TODO | 每阶段采样 nvidia-smi |
| R008 | M1 | FP16 anchor | FP16 | LLaMA-2-7B | W2/C4 PPL | MUST | TODO | 论文/官方参考 |
| R009 | M1 | reduced baseline | ATQ no SSR | LLaMA-2-7B; 32×1024 | PPL/time/VRAM | MUST | TODO | 非论文配置，标清 |
| R010 | M1 | reduced full baseline | ATQ + SSR | LLaMA-2-7B; 32×1024 | PPL/time/VRAM | MUST | TODO | 与 R009 隔离 SSR |
| R011 | M1 | cache optimization test | shared/subset FP cache | LLaMA-2-7B | memory parity/output parity | MUST | TODO | 必须与原实现数值一致 |
| R012 | M1 | near-full baseline | ATQ + SSR | LLaMA-2-7B; 64×2048 | PPL/time/VRAM | MUST | TODO | 先于 128 samples |
| R013 | M1 | official reproduction | ATQ + SSR | LLaMA-2-7B; 128×2048 | W2/C4 PPL | MUST | TODO | 目标接近 11.56/24.38 |
| R014 | M2 | adjacent control | adjacent Haar + shared ATQ | sampled blocks | HF energy/zero/NMSE/x-error | MUST | TODO | 无 pairing search |
| R015 | M2 | random control | random pairing Haar | same blocks | same metrics | MUST | TODO | 固定随机种子 |
| R016 | M2 | negative control | dissimilar pairing Haar | same blocks | same metrics | MUST | TODO | 验证因果方向 |
| R017 | M2 | SSR-order test | SSR order + adjacent Haar | same blocks | same metrics | MUST | TODO | 检查 order 是否足够 |
| R018 | M2 | explicit matching | cosine matching Haar | same blocks | same metrics | MUST | TODO | M2 go/no-go 核心 |
| R019 | M3 | band grid | matching Haar + separate grids | TinyLlama | W2/C4 PPL/layer stats | MUST | TODO | mu_H=0 |
| R020 | M3 | random full control | random Haar + separate grids | TinyLlama | PPL/layer stats | MUST | TODO | 控制额外参数 |
| R021 | M3 | PT² full comparison | PT² SSR baseline | TinyLlama | PPL/layer stats | MUST | TODO | 同校准集 |
| R022 | M3 | joint alignment | matching Haar + joint AGA | TinyLlama | PPL/condition/fallback | MUST | TODO | 3×3 solve + damping |
| R023 | M3 | primary-model screening | best Haar variant | LLaMA-2-7B reduced/full | W2/C4 PPL | MUST | TODO | M3 stop/go gate |
| R024 | M4 | bit accounting | PT² vs Haar method | LLaMA-2-7B | effective/physical bpw | MUST | TODO | scale/shift/perm/padding |
| R025 | M4 | bit-matched result | best constrained variant | LLaMA-2-7B | PPL at equal bpw | MUST | TODO | 主张 C2 |
| R026 | M4 | final seed 0 | PT² + best | LLaMA-2-7B | PPL/all diagnostics | MUST | TODO | 冻结配置后运行 |
| R027 | M4 | final seed 1 | PT² + best | LLaMA-2-7B | PPL/all diagnostics | MUST | TODO | calibration seed |
| R028 | M4 | final seed 2 | PT² + best | LLaMA-2-7B | PPL/all diagnostics | MUST | TODO | calibration seed |
| R029 | M4 | layer failure map | best method | LLaMA-2-7B | per-layer gains/failures | NICE | TODO | appendix figure |
| R030 | M5 | transform microbench | block-local P+H | RTX 4090 | us/GB/s | NICE | TODO | 准确率 gate 后 |
| R031 | M5 | ternary path prototype | factorized Haar GEMM | RTX 4090 | kernel latency | NICE | TODO | PyTorch/Triton first |
| R032 | M5 | end-to-end benchmark | PT² vs Haar path | best model | prefill/decode tok/s | NICE | TODO | systems claim gate |
