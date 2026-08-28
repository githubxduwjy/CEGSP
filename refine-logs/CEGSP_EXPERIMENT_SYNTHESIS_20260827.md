# CEGSP Experiment Synthesis after Cloud整理

Generated on cloud: 2026-08-27T13:27:47

## Scope

- Scanned `/root/tqgsp-runs/*/result.json`.
- This report summarizes the existing TQGSP/CEGSP/TDBT-family cloud results available on this instance.
- No new experiment was launched during this整理 step.
- Strong ternary PTQ baseline runs `CEGSP-11A/11B` are not present on this cloud instance yet; current evidence is therefore still mainly direct-ternary plus CEGSP, with TDBT/QAT-gap history as background.

## High-level Status

- Total valid result files: 32; CEGSP-prefixed result files: 28.
- CEGSP runs with validation improvement: 28/28.
- CEGSP runs with simultaneous Wikitext/C4 untouched improvement: 23/28 where both holdouts are reported.
- Observed CEGSP runtime range: 19.9s to 199.1s on reported configs.

## Compact Result Table

| Run | Model | Best patch | dVal | dW | dC4 | Time(s) | Mem GiB | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| CEGSP-01A | facebook/opt-350m | ce-signflip-topk-qk | -0.3021 | -0.2701 |  | 19.9 | 1.12 | val improve only/partial |
| CEGSP-01B | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2807 |  | 22.5 | 1.12 | val improve only/partial |
| CEGSP-02A-O0 | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2807 |  | 20.3 | 1.12 | val improve only/partial |
| CEGSP-02A-O1 | facebook/opt-350m | ksweep-joint-top12-qk | -0.3507 | -0.4114 |  | 20.2 | 1.12 | val improve only/partial |
| CEGSP-02A-O2 | facebook/opt-350m | ksweep-joint-top4-qk | -0.2089 | -0.1944 |  | 20.4 | 1.12 | val improve only/partial |
| CEGSP-03A-C4TRANSFER | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2807 | -0.1749 | 51.3 | 1.12 | W+C4 improve |
| CEGSP-03B-O1 | facebook/opt-350m | ksweep-joint-top6-qk | -0.2679 | -0.2687 | -0.2957 | 50.2 | 1.12 | W+C4 improve |
| CEGSP-03B-O2 | facebook/opt-350m | ksweep-joint-top4-qk | -0.2089 | -0.1944 | -0.1497 | 51.1 | 1.12 | W+C4 improve |
| CEGSP-04A-E128 | facebook/opt-350m | cegsp-support-topk-qk | -0.3754 | -0.3240 | -0.1847 | 54.2 | 1.12 | W+C4 improve |
| CEGSP-04A-E16 | facebook/opt-350m | ce-signflip-all-qk | -0.1411 | -0.1288 | -0.1829 | 49.5 | 1.12 | W+C4 improve |
| CEGSP-04A-E32 | facebook/opt-350m | cegsp-support-topk-qk | -0.2104 | -0.1831 | -0.1607 | 50.9 | 1.12 | W+C4 improve |
| CEGSP-04A-E64 | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2807 | -0.1749 | 52.4 | 1.12 | W+C4 improve |
| CEGSP-04B-OPT125M-O0 | facebook/opt-125m | cegsp-support-all-qk | -0.4164 | -0.3911 | -0.4719 | 44.8 | 0.98 | W+C4 improve |
| CEGSP-04B-OPT125M-O1 | facebook/opt-125m | cegsp-support-all-qk | -0.3339 | -0.3546 | -0.3452 | 44.3 | 0.98 | W+C4 improve |
| CEGSP-04B-OPT125M-O2 | facebook/opt-125m | cegsp-support-all-qk | -0.5545 | -0.5069 | -0.5114 | 50.5 | 0.98 | W+C4 improve |
| CEGSP-05A-OPT125M-O0-U32 | facebook/opt-125m | cegsp-support-all-qk | -0.4164 | -0.4626 | -0.5595 | 48.1 | 0.98 | W+C4 improve |
| CEGSP-05A-OPT350M-O0-U32 | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2558 | -0.2103 | 57.1 | 1.12 | W+C4 improve |
| CEGSP-05B-OPT125M-O0-U32-RANDOM | facebook/opt-125m | cegsp-support-all-qk | -0.4164 | -0.4626 | -0.5595 | 57.2 | 0.98 | W+C4 improve |
| CEGSP-05B-OPT350M-O0-U32-RANDOM | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2558 | -0.2103 | 86.0 | 1.12 | W+C4 improve |
| CEGSP-06A-OPT350M-O0-U32-MATCHED | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2558 | -0.2103 | 88.3 | 1.12 | W+C4 improve |
| CEGSP-07A-OPT350M-O0-U32-TERNARYSPEC | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2558 | -0.2103 | 92.1 | 1.12 | W+C4 improve |
| CEGSP-07B-OPT125M-O0-U32-TERNARYSPEC | facebook/opt-125m | cegsp-support-all-qk | -0.4164 | -0.4626 | -0.5595 | 64.0 | 0.98 | W+C4 improve |
| CEGSP-08A-OPT125M-O0-U32-CLOZE | facebook/opt-125m | cegsp-support-all-qk | -0.4164 | -0.4626 | -0.5595 | 63.1 | 0.98 | W+C4 improve |
| CEGSP-08A-OPT350M-O0-U32-CLOZE | facebook/opt-350m | ksweep-joint-top6-qk | -0.3226 | -0.2558 | -0.2103 | 79.4 | 1.12 | W+C4 improve |
| CEGSP-09A-OPT13B-O0-U32-SCALE | facebook/opt-1.3b | cegsp-support-selected-qk | -0.2702 | -0.2855 | -0.1027 | 130.9 | 4.43 | W+C4 improve |
| CEGSP-09B-OPT27B-O0-U32-SCALE | facebook/opt-2.7b | ce-signflip-all-qk | -0.6944 | -0.6803 | -0.7459 | 199.1 | 7.41 | W+C4 improve |
| CEGSP-10A-PYTHIA1B-CROSSARCH | EleutherAI/pythia-1b | matched-support-on-signflip-layers-top4-qk | -0.6430 | -0.6577 | -0.4620 | 105.3 | 3.92 | W+C4 improve |
| CEGSP-10A-PYTHIA1B-CROSSARCH-RERUN | EleutherAI/pythia-1b | matched-support-on-signflip-layers-top4-qk | -0.6430 | -0.6577 | -0.4620 | 64.9 | 3.92 | W+C4 improve |

## Evidence That Looks Solid

1. Direct ternary is consistently repairable by a single CE-gradient guided discrete edit pass on the available OPT-family and Pythia-family runs.
2. The Pythia-1B cross-architecture rerun supports that the architecture adapter path is viable beyond OPT-family; the rerun reports validation, Wikitext holdout, and C4 holdout improvements with about 3.92 GiB peak allocation and about 65 seconds runtime.
3. Matched-control and ternary-specific runs provide evidence that support relocation is not merely arbitrary extra computation; however, this is still mechanism evidence, not final competitiveness evidence.

## What Is Not Yet Proven

1. CEGSP has not yet been compared on this instance against a reproduced strong ternary PTQ baseline such as PT² ATQ under matched model, calibration, bit accounting, and split.
2. The current report cannot claim SOTA or superiority over latest ternary PTQ. It can only claim robust improvement over the canonical direct ternary starting point and support for a low-cost repair-layer hypothesis.
3. The cost story is promising but incomplete: current result files report wall-clock and peak memory, but the strong-baseline comparison must also count calibration tokens, forward/backward passes, candidate evaluations, and effective bpw.

## Recommended Next Cloud Batch

1. Run `CEGSP-11A-AUDIT-OPT350M`: FP16, direct ternary, PT² ATQ full, and optional PT² SSR under identical splits and token budgets.
2. If 11A is finite and comparable, run `CEGSP-11B-2X2-OPT350M`: direct, direct+CEGSP, PT², PT²+CEGSP with frozen PT² scale/offset and edited ternary state only.
3. Report accuracy and cost together: NLL/PPL, paired deltas, bootstrap CI, wall-clock, peak memory, calibration token budget, effective bpw, and number of backward passes.

## Decision Rule

- If CEGSP beats or matches PT² at much lower quantization cost, frame it as an efficient independent PTQ method.
- If PT² remains better but PT²+CEGSP improves over PT², frame CEGSP as a quantized-point function-repair layer for strong ternary PTQ.
- If both standalone and PT²-composed versions fail, stop adding modules and narrow the claim to mechanism analysis over direct ternary only.

CSV artifact: `/root/tqgsp-runs/CEGSP_EXPERIMENT_SUMMARY_20260827.csv`
