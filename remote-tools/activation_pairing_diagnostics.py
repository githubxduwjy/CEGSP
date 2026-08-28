#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from haar_mechanism_diagnostics import (
    SQRT2_INV,
    haar_inverse,
    haar_transform,
    measure,
    pairing_order,
    quantize_coefficients,
    stable_seed,
    summarize,
)
from pt2_llm.data import get_loaders


@torch.no_grad()
def activation_weighted_hf_order(weight: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """Greedy matching that minimizes a pairwise activation-weighted Haar high-band proxy."""
    w = weight.float()
    x = acts.float()
    w2 = w.square().sum(dim=0, keepdim=True)
    x2 = x.square().mean(dim=0, keepdim=True)
    dw2 = (w2.T + w2 - 2.0 * (w.T @ w)).clamp_min_(0.0)
    dx2 = (x2.T + x2 - 2.0 * ((x.T @ x) / max(1, x.shape[0]))).clamp_min_(0.0)
    score = dw2 * dx2
    n = score.shape[0]
    available = torch.ones(n, dtype=torch.bool, device=score.device)
    pairs = []
    for _ in range(n // 2):
        valid = available[:, None] & available[None, :]
        valid.fill_diagonal_(False)
        masked = score.masked_fill(~valid, torch.inf)
        flat = torch.argmin(masked)
        i = int(flat // n)
        j = int(flat % n)
        pairs.extend((i, j))
        available[i] = False
        available[j] = False
    return torch.tensor(pairs, device=weight.device, dtype=torch.long)


@torch.no_grad()
def activation_covariance_order(weight: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    """Pair columns with similar weights and similar activation traces."""
    weight_cos = F.normalize(weight.float(), dim=0).T @ F.normalize(weight.float(), dim=0)
    acts_cos = F.normalize(acts.float(), dim=0).T @ F.normalize(acts.float(), dim=0)
    score = weight_cos * acts_cos
    n = score.shape[0]
    available = torch.ones(n, dtype=torch.bool, device=score.device)
    pairs = []
    for _ in range(n // 2):
        valid = available[:, None] & available[None, :]
        valid.fill_diagonal_(False)
        masked = score.masked_fill(~valid, -torch.inf)
        flat = torch.argmax(masked)
        i = int(flat // n)
        j = int(flat % n)
        pairs.extend((i, j))
        available[i] = False
        available[j] = False
    return torch.tensor(pairs, device=weight.device, dtype=torch.long)


@torch.no_grad()
def measure_custom_order(weight: torch.Tensor, acts: torch.Tensor, strategy: str, order: torch.Tensor, grid_mode: str):
    coeff, transformed_acts = haar_transform(weight, acts, order)
    half = coeff.shape[1] // 2
    high = coeff[:, half:]
    total_energy = coeff.float().square().sum().clamp_min(1e-30)
    hf_ratio = float((high.float().square().sum() / total_energy).item())
    high_abs = high.float().abs()
    normed = F.normalize(weight.float(), dim=0)
    pair_cosine = float((normed[:, order[0::2]] * normed[:, order[1::2]]).sum(dim=0).mean().item())
    exact_num = weight.float() @ acts.float().T - coeff.float() @ transformed_acts.float().T
    exact_den = (weight.float() @ acts.float().T).norm().clamp_min(1e-30)
    quantized, ternary, fallback = quantize_coefficients(coeff, transformed_acts, grid_mode)
    residual = coeff.float() - quantized.float()
    output_ref = coeff.float() @ transformed_acts.float().T
    output_err = residual @ transformed_acts.float().T
    reconstructed = haar_inverse(quantized, order)
    return {
        "strategy": strategy,
        "grid_mode": grid_mode,
        "pair_cosine_mean": pair_cosine,
        "hf_energy_ratio": hf_ratio,
        "hf_abs_p50": float(torch.quantile(high_abs.flatten(), 0.50).item()),
        "hf_abs_p90": float(torch.quantile(high_abs.flatten(), 0.90).item()),
        "hf_abs_p99": float(torch.quantile(high_abs.flatten(), 0.99).item()),
        "ternary_zero_rate": float((ternary == 0).float().mean().item()),
        "high_zero_rate": float((ternary[:, half:] == 0).float().mean().item()),
        "weight_nmse": float((residual.square().sum() / coeff.float().square().sum().clamp_min(1e-30)).item()),
        "activation_weighted_nmse": float((output_err.square().sum() / output_ref.square().sum().clamp_min(1e-30)).item()),
        "inverse_weight_nmse": float(((weight.float() - reconstructed.float()).square().sum() / weight.float().square().sum().clamp_min(1e-30)).item()),
        "haar_exact_rel_error": float((exact_num.norm() / exact_den).item()),
        "atq_fallback": fallback,
    }


def parse_layers(text: str):
    return [int(x) for x in text.split(",") if x]


def paired_improvements(rows, candidates):
    by_key = {(r["module"], r["block_start"], r["strategy"]): r for r in rows}
    block_keys = sorted({(r["module"], r["block_start"]) for r in rows})
    result = {}
    for candidate in candidates:
        values = []
        wins = []
        for module, block_start in block_keys:
            cand = by_key[(module, block_start, candidate)]
            rnd = by_key[(module, block_start, "random")]
            improvement = (
                rnd["activation_weighted_nmse"] - cand["activation_weighted_nmse"]
            ) / max(rnd["activation_weighted_nmse"], 1e-30)
            values.append(improvement)
            wins.append(improvement > 0)
        result[candidate] = {
            "n": len(values),
            "median_weighted_error_improvement_vs_random": statistics.median(values),
            "mean_weighted_error_improvement_vs_random": statistics.fmean(values),
            "weighted_error_win_fraction_vs_random": statistics.fmean(wins),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,10,20,31")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--tokens-per-sample", type=int, default=128)
    parser.add_argument("--blocks-per-module", type=int, default=2)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-mode", choices=("shared", "band"), default="band")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
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
    ).to("cuda:0")
    model.eval()
    model.config.use_cache = False
    targets = {}
    for layer_idx in layers:
        layer = model.model.layers[layer_idx]
        named = dict(layer.named_modules())
        for projection in projection_names:
            targets[f"layer_{layer_idx}.{projection}"] = named[projection]

    activations = {name: [] for name in targets}
    handles = []
    for name, module in targets.items():
        def hook(_, inputs, __, key=name):
            flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(args.tokens_per_sample, flat.shape[0])
            index = torch.linspace(0, flat.shape[0] - 1, count, device=flat.device).long()
            activations[key].append(flat[index].float().cpu())
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
            model(batch[0].to("cuda:0"), use_cache=False)
            print(f"Captured calibration sample {sample_idx + 1}/{args.nsamples}", flush=True)
    for handle in handles:
        handle.remove()

    weights = {name: module.weight.detach().cpu().clone() for name, module in targets.items()}
    targets.clear()
    model.cpu()
    del model
    torch.cuda.empty_cache()
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)

    rows = []
    base_strategies = ("identity", "adjacent", "random", "dissimilar", "ssr_order", "similarity")
    custom_strategies = ("activation_hf", "activation_cov")
    total = len(weights) * args.blocks_per_module * (len(base_strategies) + len(custom_strategies))
    completed = 0
    for module_name in sorted(weights):
        weight_cpu = weights[module_name]
        acts_cpu = activations[module_name]
        in_features = weight_cpu.shape[1]
        max_start = in_features - args.blocksize
        if args.blocks_per_module == 1:
            starts = [0]
        else:
            starts = [
                int(round(i * max_start / (args.blocks_per_module - 1)) // args.blocksize * args.blocksize)
                for i in range(args.blocks_per_module)
            ]
        starts = sorted(set(min(max_start, max(0, s)) for s in starts))
        for block_start in starts:
            block_end = block_start + args.blocksize
            weight = weight_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            acts = acts_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            for strategy in base_strategies:
                row = measure(
                    weight,
                    acts,
                    strategy,
                    stable_seed(f"{module_name}:{block_start}:{strategy}", args.seed),
                    args.grid_mode,
                )
                row["inverse_weight_nmse"] = row["weight_nmse"]
                row.update({
                    "module": module_name,
                    "layer_type": module_name.split(".", 1)[1],
                    "block_start": block_start,
                    "blocksize": args.blocksize,
                    "activation_tokens": acts.shape[0],
                })
                rows.append(row)
                completed += 1
            custom_orders = {
                "activation_hf": activation_weighted_hf_order(weight, acts),
                "activation_cov": activation_covariance_order(weight, acts),
            }
            for strategy, order in custom_orders.items():
                row = measure_custom_order(weight, acts, strategy, order, args.grid_mode)
                row.update({
                    "module": module_name,
                    "layer_type": module_name.split(".", 1)[1],
                    "block_start": block_start,
                    "blocksize": args.blocksize,
                    "activation_tokens": acts.shape[0],
                })
                rows.append(row)
                completed += 1
            if completed % 12 == 0 or completed == total:
                print(f"Measured {completed}/{total}", flush=True)
            del weight, acts
            torch.cuda.empty_cache()

    pair_summary = paired_improvements(rows, ("similarity", "dissimilar", "activation_hf", "activation_cov"))
    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summarize(rows),
        "pivot_summary": pair_summary,
        "gate": {
            "candidate": "activation_hf",
            "target_median_weighted_error_improvement_vs_random": 0.05,
            "target_win_fraction_vs_random": 0.70,
            "pass": pair_summary["activation_hf"]["median_weighted_error_improvement_vs_random"] >= 0.05
            and pair_summary["activation_hf"]["weighted_error_win_fraction_vs_random"] >= 0.70,
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    (out_dir / "pivot_summary.json").write_text(json.dumps(pair_summary, indent=2), encoding="utf-8")
    print(json.dumps({"pivot_summary": pair_summary, "gate": result["gate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
