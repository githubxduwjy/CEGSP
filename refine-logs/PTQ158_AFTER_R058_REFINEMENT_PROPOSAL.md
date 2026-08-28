# Anchored pivot proposal after R058

## Problem anchor

- Bottom-line problem: design a post-training ternary (1.58-bit) method that preserves the pretrained LLM's function, not merely its weight matrix, without turning PTQ into QAT.
- Must-solve bottleneck: FP16-to-ternary support mismatch; locally attractive ternary assignments often change the quantized autoregressive trajectory and are distribution-sensitive.
- Non-goals: small cosmetic changes to PT²; projection-mask enumeration; post-hoc epsilon tuning; claiming speed or exact 1.58 bpw without implementation/accounting.
- Constraints: LLaMA-2-7B on a 24 GB RTX 4090; small calibration sets; frozen pretrained weights except quantization parameters/ternary assignments; PTQ-scale cost.
- Success condition: preregistered candidate selection transfers to untouched Wikitext2 and C4, improves or preserves model-level function relative to official PT²/GPTQ, and generalizes beyond one layer/window/seed.

## Anchor and simplicity check

- The evidence supports a local hard-T mechanism but rejects local reconstruction as a sufficient objective and rejects continued finite-sample gate engineering as the main method.
- The proposed pivot changes the optimization signal, not the problem: it makes a short quantized window's end-to-end function shape the ternary assignment while retaining local constraints.
- Dominant contribution: cross-layer function-preserving ternary support optimization under a local trust region.
- Intentionally excluded: coordinate-system search, layer-mask search, adaptive epsilon, large calibration-data recipes, full-model joint optimization, and multiple new trainable modules.

## Proposed method: Windowed Function-Preserving Ternarization (WFPT)

For a frozen two-layer window starting at layer `l`, let `T_j in {-1,0,+1}` and quantized weights `W_j^q = alpha_j T_j + mu_j`. Initialize all `T_j, alpha_j, mu_j` from official PT²/ATQ. Optimize only the current window while all preceding layers run in their already-quantized context and all following layers stay unchanged.

The objective is

`L = L_end(h_{l+2}^q, h_{l+2}^{fp-target}) + lambda * sum_{j=l}^{l+1} L_local(h_{j+1}^q, h_{j+1}^{fp-target}) + beta * D(T,T0)`.

- `L_end`: normalized MSE plus cosine drift at the window boundary, evaluated on mixed W2/C4 fit sequences in the true quantized prefix context.
- `L_local`: the same reconstruction signal at each layer boundary; it prevents compensating cross-layer changes from destroying an individual layer.
- `D(T,T0)`: sparse flip budget or Hamming penalty relative to the PT² initializer.
- No language-model test NLL is used for optimization. A disjoint validation split uses worst-domain window-boundary loss to stop or fall back to `T0`; final mean/CVaR NLL is untouched evaluation only.

The first solver is bounded hard block-coordinate descent: at each step, generate a small set of activation/Hessian-ranked support flips, evaluate their exact two-layer objective in one batched forward pass, accept only the best objective-decreasing tranche, then refit `alpha,mu`. This directly uses the cross-layer signal to choose support changes, unlike R047--R058 where a locally formed candidate was only filtered after construction.

## Backup solver, not a parallel contribution

If hard coordinate descent cannot find stable improvements under the same objective, use ternary logits with temperature annealing and a straight-through or Gumbel-style estimator, followed by hard projection and mandatory endpoint evaluation. The scientific claim remains the window objective; the relaxation is only a solver. If the hard endpoint loses the soft gain, this branch stops.

## Calibration-data role

Use a frozen 50/50 W2/C4 mixture for the first claim test. Model-generated/task traces are excluded from the core method because they would change the claim to task-adaptive PTQ. Calibration alignment can be a later backup only if the method is explicitly reframed and evaluated as task-adaptive.

## Minimal validation blocks (plans only)

1. **Objective identifiability, one frozen window (10,11):** official PT², local hard-T from R042c, post-hoc gated hard-T, and WFPT with identical flip/evaluation budgets. Fit/validation/test sequence identities are frozen. Primary gate: WFPT must improve worst-domain untouched window-end loss and must not regress either domain's mean or CVaR NLL. This distinguishes objective shaping from gate filtering.
2. **Local-term necessity:** compare `L_end` only versus `L_end + lambda L_local` at one preregistered lambda set by scale normalization, not a sweep. The retained local term must reduce layer-level degradation without erasing the window-end gain.
3. **One replication only after blocks 1--2 pass:** same frozen method at an early window (0,1) and a new calibration seed. Require directionally consistent worst-domain results and no nonfinite outputs. Do not expand to all layers before this passes.

## Stop conditions

- Stop WFPT if block 1 cannot beat both local hard-T and post-hoc gate under matched candidate/evaluation budget on untouched worst-domain metrics.
- Stop the local-trust-region formulation if block 2 shows the local term gives no stability benefit or simply suppresses all useful updates.
- Stop the broader cross-layer objective direction if the frozen method fails the early-window/seed replication or if cost exceeds a preregistered PTQ budget (proposed cap: <=3x official quantization time for a two-layer window) without a clear accuracy gain.
- Move to the soft-to-hard backup only after a diagnosed search/optimization failure, not after a generalization failure.
