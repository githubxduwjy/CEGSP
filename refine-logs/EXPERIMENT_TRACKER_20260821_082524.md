# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Model / Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | 机器审计 | driver/CUDA/RAM/disk/network | Cloud 4090 | system inventory | MUST | TODO | 收到连接后只读审计 |
| R001 | M0 | 环境冻结 | official PT² commit + pinned env | N/A | import/unit smoke | MUST | TODO | clone --recurse-submodules；记录 SHA |
| R002 | M0 | 数据准备 | WikiText2/C4/tokenizer | Dev model | sample hashes/counts | MUST | TODO | 不在日志打印 token |
| R003 | M0 | 评估校验 | FP16 | TinyLlama 1.1B | W2/C4 PPL | MUST | TODO | 对照 HF 原始模型 |
| R004 | M0 | 量化 smoke | ternary-init | TinyLlama; 8×256 | PPL/VRAM/time | MUST | TODO | 最小可运行配置 |
| R005 | M0 | ATQ smoke | ATQ no SSR | TinyLlama; 16×512 | PPL/VRAM/time | MUST | TODO | ITF+AGA 路径 |
| R006 | M0 | full smoke | ATQ + SSR | TinyLlama; 16×512 | PPL/VRAM/time | MUST | TODO | 验证保存/加载 |
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
