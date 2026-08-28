#!/usr/bin/env python3
"""Fair-control confirmation for validation-gated ternary refinement (R042c).

This deliberately runs before full GPTQ integration.  It asks whether changing
the discrete ternary assignment T can improve unseen activation-weighted output
error when proposal, acceptance, and test activations are kept disjoint.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from pt2_llm.data import get_loaders
from pt2_llm.quantizer import solve_closed_form_alpha_mu, ternary_init, update_ternary


@torch.no_grad()
def weight_itf(weight: torch.Tensor, max_iter: int = 100):
    mean, alpha, ternary = ternary_init(weight)
    ternary = ternary.float()
    for _ in range(max_iter):
        previous = ternary.clone()
        alpha, mean = solve_closed_form_alpha_mu(ternary, weight)
        ternary = update_ternary(weight, alpha, mean).float()
        if torch.equal(ternary, previous):
            break
    return ternary, alpha.float(), mean.float()


@torch.no_grad()
def activation_grid(
    ternary: torch.Tensor,
    weight: torch.Tensor,
    covariance: torch.Tensor,
    fallback_alpha: torch.Tensor,
    fallback_mean: torch.Tensor,
):
    """Solve min_{alpha,mu} ||(W-alpha*T-mu) X|| row-wise."""
    t = ternary.double()
    w = weight.double()
    s = covariance.double()
    one = torch.ones((s.shape[0], 1), device=s.device, dtype=s.dtype)

    a = torch.sum((t @ s) * t, dim=1)
    b = (t @ s @ one).squeeze(-1)
    d = (one.T @ s @ one).squeeze()
    y1 = torch.sum((w @ s) * t, dim=1)
    y2 = (w @ s @ one).squeeze(-1)
    denom = a * d - b.square()

    alpha = (d * y1 - b * y2) / denom
    mean = (a * y2 - b * y1) / denom
    valid = torch.isfinite(alpha) & torch.isfinite(mean) & (denom.abs() > 1e-12)
    alpha = torch.where(valid, alpha, fallback_alpha.double())
    mean = torch.where(valid, mean, fallback_mean.double())
    return alpha.float(), mean.float(), int((~valid).sum().item())


@torch.no_grad()
def reconstruct(ternary: torch.Tensor, alpha: torch.Tensor, mean: torch.Tensor):
    return alpha[:, None] * ternary + mean[:, None]


@torch.no_grad()
def row_loss(weight: torch.Tensor, quantized: torch.Tensor, covariance: torch.Tensor):
    error = weight - quantized
    return torch.sum((error @ covariance) * error, dim=1)


@torch.no_grad()
def aggregate_nmse(weight: torch.Tensor, quantized: torch.Tensor, covariance: torch.Tensor):
    numerator = row_loss(weight, quantized, covariance).sum()
    denominator = torch.sum((weight @ covariance) * weight).clamp_min(1e-30)
    return float((numerator / denominator).item())


@torch.no_grad()
def refine_ternary(
    weight: torch.Tensor,
    ternary: torch.Tensor,
    proposal_covariance: torch.Tensor,
    validation_covariance: torch.Tensor | None,
    max_steps: int,
):
    """One sensitivity-ranked coordinate proposal per row and step.

    If validation_covariance is supplied, a proposal must reduce both proposal
    and validation loss.  Coordinates already proposed are locked so the search
    cannot oscillate or repeatedly tune one threshold-adjacent weight.
    """
    current_t = ternary.clone().float()
    fallback_alpha, fallback_mean = solve_closed_form_alpha_mu(current_t, weight)
    alpha, mean, fallback_rows = activation_grid(
        current_t, weight, proposal_covariance, fallback_alpha, fallback_mean
    )
    locked = torch.zeros_like(current_t, dtype=torch.bool)
    accepted_total = 0
    proposed_total = 0
    steps_run = 0

    states = torch.tensor((-1.0, 0.0, 1.0), device=weight.device)
    diag = torch.diag(proposal_covariance).view(1, -1)

    for step in range(max_steps):
        quantized = reconstruct(current_t, alpha, mean)
        error = weight - quantized
        error_s = error @ proposal_covariance

        delta = alpha[:, None, None] * (
            states.view(1, 1, 3) - current_t[:, :, None]
        )
        gain = 2.0 * delta * error_s[:, :, None] - delta.square() * diag[:, :, None]
        gain = gain.masked_fill(locked[:, :, None], -torch.inf)
        gain = gain.masked_fill(
            states.view(1, 1, 3) == current_t[:, :, None], -torch.inf
        )

        flat_gain = gain.reshape(gain.shape[0], -1)
        best_gain, best_flat = flat_gain.max(dim=1)
        best_col = torch.div(best_flat, 3, rounding_mode="floor")
        best_state = states[best_flat.remainder(3)]
        rows = torch.arange(weight.shape[0], device=weight.device)
        locked[rows, best_col] = True

        proposal_t = current_t.clone()
        proposal_t[rows, best_col] = best_state
        prop_alpha, prop_mean, fallback = activation_grid(
            proposal_t, weight, proposal_covariance, alpha, mean
        )
        fallback_rows += fallback
        proposal_q = reconstruct(proposal_t, prop_alpha, prop_mean)

        current_fit = row_loss(weight, quantized, proposal_covariance)
        proposal_fit = row_loss(weight, proposal_q, proposal_covariance)
        accept = torch.isfinite(best_gain) & (proposal_fit < current_fit - 1e-12)
        if validation_covariance is not None:
            current_val = row_loss(weight, quantized, validation_covariance)
            proposal_val = row_loss(weight, proposal_q, validation_covariance)
            accept &= proposal_val < current_val - 1e-12

        proposed_total += int(torch.isfinite(best_gain).sum().item())
        accepted = int(accept.sum().item())
        accepted_total += accepted
        steps_run = step + 1
        if accepted == 0:
            break

        current_t = torch.where(accept[:, None], proposal_t, current_t)
        alpha = torch.where(accept, prop_alpha, alpha)
        mean = torch.where(accept, prop_mean, mean)

    return reconstruct(current_t, alpha, mean), current_t, {
        "steps": steps_run,
        "proposed": proposed_total,
        "accepted": accepted_total,
        "acceptance_rate": accepted_total / max(proposed_total, 1),
        "changed_fraction": float((current_t != ternary).float().mean().item()),
        "fallback_rows": fallback_rows,
    }


def covariance(activation_chunks, device):
    x = torch.cat(activation_chunks, dim=0).to(device=device, dtype=torch.float32)
    s = x.T @ x
    s /= max(x.shape[0], 1)
    return s, x.shape[0]


def starts_for_width(width: int, blocksize: int, count: int):
    max_start = width - blocksize
    if count == 1:
        return [0]
    starts = [
        int(round(i * max_start / (count - 1)) // blocksize * blocksize)
        for i in range(count)
    ]
    return sorted(set(min(max_start, max(0, value)) for value in starts))


def summarize(rows):
    baseline = "fixed_t_allcal"
    variants = (
        "itf",
        "fixed_t_fit",
        "ungated_allcal_refine",
        "ungated_fit_refit",
        "gated_no_refit",
        "gated_refit",
    )
    summary = {}
    for variant in variants:
        improvements = []
        wins = []
        fit_improvements = []
        for row in rows:
            base_test = row[f"{baseline}_test_nmse"]
            value_test = row[f"{variant}_test_nmse"]
            improvement = (base_test - value_test) / max(base_test, 1e-30)
            improvements.append(improvement)
            wins.append(value_test < base_test)
            base_fit = row[f"{baseline}_fit_nmse"]
            value_fit = row[f"{variant}_fit_nmse"]
            fit_improvements.append((base_fit - value_fit) / max(base_fit, 1e-30))
        summary[variant] = {
            "n_blocks": len(rows),
            "median_test_improvement_vs_fixed_t_allcal": statistics.median(improvements),
            "mean_test_improvement_vs_fixed_t_allcal": statistics.fmean(improvements),
            "test_win_rate_vs_fixed_t_allcal": statistics.fmean(wins),
            "median_fit_improvement_vs_fixed_t_allcal": statistics.median(fit_improvements),
        }

    gated = summary["gated_refit"]
    ungated = summary["ungated_fit_refit"]
    gate = {
        "median_test_improvement_target": 0.02,
        "test_win_rate_target": 0.65,
        "gated_not_worse_than_matched_ungated_target": True,
        "pass": gated["median_test_improvement_vs_fixed_t_allcal"] >= 0.02
        and gated["test_win_rate_vs_fixed_t_allcal"] >= 0.65
        and gated["median_test_improvement_vs_fixed_t_allcal"]
        >= ungated["median_test_improvement_vs_fixed_t_allcal"],
    }
    return {"by_variant": summary, "gate": gate}


def parse_layers(text: str):
    return [int(value) for value in text.split(",") if value]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,10,20,31")
    parser.add_argument("--nsamples", type=int, default=12)
    parser.add_argument("--tokens-per-sample", type=int, default=128)
    parser.add_argument("--blocks-per-module", type=int, default=2)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.nsamples < 4:
        raise ValueError("At least four samples are required for fit/validation/test splitting")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = "cuda:0"
    layers = parse_layers(args.layers)
    projection_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )

    print(f"Loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    targets = {}
    for layer_idx in layers:
        named = dict(model.model.layers[layer_idx].named_modules())
        for projection in projection_names:
            targets[f"layer_{layer_idx}.{projection}"] = named[projection]

    activation_chunks = {name: [] for name in targets}
    handles = []
    for name, module in targets.items():
        def hook(_, inputs, __, key=name):
            flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(args.tokens_per_sample, flat.shape[0])
            index = torch.linspace(0, flat.shape[0] - 1, count, device=flat.device).long()
            activation_chunks[key].append(flat[index].float().cpu())
        handles.append(module.register_forward_hook(hook))

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=2048,
    )
    with torch.inference_mode():
        for sample_idx, batch in enumerate(dataloader):
            model(batch[0].to(device), use_cache=False)
            print(f"Captured sample {sample_idx + 1}/{args.nsamples}", flush=True)
    for handle in handles:
        handle.remove()

    weights = {name: module.weight.detach().cpu().clone() for name, module in targets.items()}
    targets.clear()
    model.cpu()
    del model
    torch.cuda.empty_cache()

    n_fit = max(1, int(args.nsamples * args.fit_fraction))
    n_validation = max(1, int(args.nsamples * args.validation_fraction))
    if n_fit + n_validation >= args.nsamples:
        n_validation = args.nsamples - n_fit - 1
    if n_validation < 1:
        raise ValueError("Split leaves no validation sample")

    rows = []
    for module_name in sorted(weights):
        chunks = activation_chunks[module_name]
        fit_chunks = chunks[:n_fit]
        validation_chunks = chunks[n_fit:n_fit + n_validation]
        test_chunks = chunks[n_fit + n_validation:]
        weight_cpu = weights[module_name]

        for block_start in starts_for_width(
            weight_cpu.shape[1], args.blocksize, args.blocks_per_module
        ):
            block_end = block_start + args.blocksize
            weight = weight_cpu[:, block_start:block_end].to(device=device, dtype=torch.float32)
            fit_s, fit_tokens = covariance(
                [chunk[:, block_start:block_end] for chunk in fit_chunks], device
            )
            validation_s, validation_tokens = covariance(
                [chunk[:, block_start:block_end] for chunk in validation_chunks], device
            )
            test_s, test_tokens = covariance(
                [chunk[:, block_start:block_end] for chunk in test_chunks], device
            )
            allcal_s = (
                fit_s * fit_tokens + validation_s * validation_tokens
            ) / (fit_tokens + validation_tokens)

            ternary, itf_alpha, itf_mean = weight_itf(weight)
            variants = {}
            variants["itf"] = reconstruct(ternary, itf_alpha, itf_mean)

            fit_alpha, fit_mean, fit_fallback = activation_grid(
                ternary, weight, fit_s, itf_alpha, itf_mean
            )
            variants["fixed_t_fit"] = reconstruct(ternary, fit_alpha, fit_mean)

            all_alpha, all_mean, all_fallback = activation_grid(
                ternary, weight, allcal_s, itf_alpha, itf_mean
            )
            variants["fixed_t_allcal"] = reconstruct(ternary, all_alpha, all_mean)

            ungated_allcal_q, _, ungated_allcal_info = refine_ternary(
                weight, ternary, allcal_s, None, args.max_steps
            )
            ungated_fit_q, ungated_fit_t, ungated_fit_info = refine_ternary(
                weight, ternary, fit_s, None, args.max_steps
            )
            gated_q, gated_t, gated_info = refine_ternary(
                weight, ternary, fit_s, validation_s, args.max_steps
            )

            # Selection and nuisance-parameter estimation must be separated.
            # Both matched variants select T using fit activations, then freeze T
            # and refit only alpha/mu on fit+validation before untouched testing.
            ungated_fallback_alpha, ungated_fallback_mean = solve_closed_form_alpha_mu(
                ungated_fit_t, weight
            )
            ungated_refit_alpha, ungated_refit_mean, ungated_refit_fallback = activation_grid(
                ungated_fit_t,
                weight,
                allcal_s,
                ungated_fallback_alpha,
                ungated_fallback_mean,
            )
            gated_fallback_alpha, gated_fallback_mean = solve_closed_form_alpha_mu(
                gated_t, weight
            )
            gated_refit_alpha, gated_refit_mean, gated_refit_fallback = activation_grid(
                gated_t,
                weight,
                allcal_s,
                gated_fallback_alpha,
                gated_fallback_mean,
            )
            variants["ungated_allcal_refine"] = ungated_allcal_q
            variants["ungated_fit_refit"] = reconstruct(
                ungated_fit_t, ungated_refit_alpha, ungated_refit_mean
            )
            variants["gated_no_refit"] = gated_q
            variants["gated_refit"] = reconstruct(
                gated_t, gated_refit_alpha, gated_refit_mean
            )

            row = {
                "module": module_name,
                "layer_type": module_name.split(".", 1)[1],
                "block_start": block_start,
                "blocksize": args.blocksize,
                "fit_tokens": fit_tokens,
                "validation_tokens": validation_tokens,
                "test_tokens": test_tokens,
                "fixed_t_fit_fallback_rows": fit_fallback,
                "fixed_t_allcal_fallback_rows": all_fallback,
                "ungated_allcal_changed_fraction": ungated_allcal_info["changed_fraction"],
                "ungated_allcal_acceptance_rate": ungated_allcal_info["acceptance_rate"],
                "ungated_fit_changed_fraction": ungated_fit_info["changed_fraction"],
                "ungated_fit_acceptance_rate": ungated_fit_info["acceptance_rate"],
                "gated_changed_fraction": gated_info["changed_fraction"],
                "gated_acceptance_rate": gated_info["acceptance_rate"],
                "ungated_refit_fallback_rows": ungated_refit_fallback,
                "gated_refit_fallback_rows": gated_refit_fallback,
            }
            for variant, quantized in variants.items():
                row[f"{variant}_fit_nmse"] = aggregate_nmse(weight, quantized, fit_s)
                row[f"{variant}_validation_nmse"] = aggregate_nmse(
                    weight, quantized, validation_s
                )
                row[f"{variant}_test_nmse"] = aggregate_nmse(weight, quantized, test_s)
            rows.append(row)
            print(
                f"Measured {module_name} block {block_start}; "
                f"gated change={gated_info['changed_fraction']:.4%}",
                flush=True,
            )
            del weight, fit_s, validation_s, test_s, allcal_s
            torch.cuda.empty_cache()

    result = {
        "config": vars(args),
        "split_samples": {
            "fit": n_fit,
            "validation": n_validation,
            "test": args.nsamples - n_fit - n_validation,
        },
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "rows": rows,
        "summary": summarize(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(result["summary"], indent=2), encoding="utf-8"
    )
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
