#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from pt2_llm.data import get_loaders
from pt2_llm.quantizer import (
    solve_closed_form_alpha_mu,
    ternary_init,
    update_ternary,
)


SQRT2_INV = 1.0 / math.sqrt(2.0)


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + base) % (2**31)


@torch.no_grad()
def haar_transform(weight: torch.Tensor, acts: torch.Tensor, order: torch.Tensor):
    left = order[0::2]
    right = order[1::2]
    low_w = (weight[:, left] + weight[:, right]) * SQRT2_INV
    high_w = (weight[:, left] - weight[:, right]) * SQRT2_INV
    low_x = (acts[:, left] + acts[:, right]) * SQRT2_INV
    high_x = (acts[:, left] - acts[:, right]) * SQRT2_INV
    return torch.cat((low_w, high_w), dim=1), torch.cat((low_x, high_x), dim=1)


@torch.no_grad()
def haar_inverse(coeff: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    half = coeff.shape[1] // 2
    low, high = coeff[:, :half], coeff[:, half:]
    out = torch.empty_like(coeff)
    out[:, order[0::2]] = (low + high) * SQRT2_INV
    out[:, order[1::2]] = (low - high) * SQRT2_INV
    return out


@torch.no_grad()
def greedy_pair_order(weight: torch.Tensor, largest: bool) -> torch.Tensor:
    normed = F.normalize(weight.float(), dim=0)
    sim = normed.T @ normed
    n = sim.shape[0]
    available = torch.ones(n, dtype=torch.bool, device=sim.device)
    pairs = []
    for _ in range(n // 2):
        valid = available[:, None] & available[None, :]
        valid.fill_diagonal_(False)
        score = sim.masked_fill(~valid, -torch.inf if largest else torch.inf)
        flat = torch.argmax(score) if largest else torch.argmin(score)
        i = int(flat // n)
        j = int(flat % n)
        pairs.extend((i, j))
        available[i] = False
        available[j] = False
    return torch.tensor(pairs, device=weight.device, dtype=torch.long)


@torch.no_grad()
def pairing_order(strategy: str, weight: torch.Tensor, seed: int) -> torch.Tensor:
    n = weight.shape[1]
    if n % 2:
        raise ValueError(f"pairing requires an even column count, got {n}")
    if strategy == "adjacent":
        return torch.arange(n, device=weight.device)
    if strategy == "random":
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(seed)
        return torch.randperm(n, generator=gen, device=weight.device)
    if strategy == "similarity":
        return greedy_pair_order(weight, largest=True)
    if strategy == "dissimilar":
        return greedy_pair_order(weight, largest=False)
    if strategy == "ssr_order":
        normed = F.normalize(weight.float(), dim=0)
        mean_vec = normed.mean(dim=1, keepdim=True)
        score = (mean_vec.T @ normed).squeeze(0)
        return torch.argsort(score, descending=True)
    raise ValueError(strategy)


@torch.no_grad()
def shared_atq(weight: torch.Tensor, covariance: torch.Tensor):
    """PT2 ATQ (ITF + AGA) with ternary codes returned for diagnostics."""
    x = weight.float()
    mu, alpha, ternary = ternary_init(x)
    current = ternary.clone()
    for _ in range(100):
        previous = current.clone()
        alpha, mu = solve_closed_form_alpha_mu(current, x)
        current = update_ternary(x, alpha, mu)
        if torch.equal(current, previous):
            break

    t64 = current.to(torch.float64)
    s64 = covariance.to(torch.float64)
    x64 = x.to(torch.float64)
    one = torch.ones((s64.shape[0], 1), device=x.device, dtype=torch.float64)
    a = torch.sum((t64 @ s64) * t64, dim=1)
    b = (t64 @ s64 @ one).squeeze(-1)
    d = (one.T @ s64 @ one).squeeze()
    y1 = torch.sum((x64 @ s64) * t64, dim=1)
    y2 = (x64 @ s64 @ one).squeeze(-1)
    denom = a * d - b * b
    alpha_try = (d * y1 - b * y2) / denom
    mu_try = (a * y2 - b * y1) / denom
    fallback = not bool(torch.isfinite(alpha_try).all() and torch.isfinite(mu_try).all())
    if not fallback:
        alpha = alpha_try
        mu = mu_try
    quantized = alpha[:, None] * t64 + mu[:, None]
    return quantized.float(), current, fallback


@torch.no_grad()
def zero_center_atq(weight: torch.Tensor, covariance: torch.Tensor):
    """Zero-centered ternary ITF + activation-aware scale for the high band."""
    x = weight.float()
    threshold = 0.75 * x.abs().mean(dim=1, keepdim=True)
    ternary = torch.zeros_like(x, dtype=torch.int)
    ternary[x > threshold] = 1
    ternary[x < -threshold] = -1
    alpha = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(100):
        previous = ternary.clone()
        denom = ternary.float().square().sum(dim=1).clamp_min(1e-12)
        alpha = (ternary.float() * x).sum(dim=1) / denom
        safe_alpha = alpha.abs().clamp_min(1e-12)
        ternary = torch.round(x / safe_alpha[:, None]).clamp(-1, 1).to(torch.int)
        if torch.equal(ternary, previous):
            break

    t64 = ternary.to(torch.float64)
    s64 = covariance.to(torch.float64)
    x64 = x.to(torch.float64)
    numerator = torch.sum((x64 @ s64) * t64, dim=1)
    denominator = torch.sum((t64 @ s64) * t64, dim=1)
    alpha_try = numerator / denominator
    finite = torch.isfinite(alpha_try)
    fallback = not bool(finite.all())
    if fallback:
        alpha_try = torch.where(finite, alpha_try, alpha.to(torch.float64))
    quantized = alpha_try[:, None] * t64
    return quantized.float(), ternary, fallback


@torch.no_grad()
def quantize_coefficients(coeff: torch.Tensor, transformed_acts: torch.Tensor, grid_mode: str):
    covariance = transformed_acts.float().T @ transformed_acts.float()
    covariance /= max(1, transformed_acts.shape[0])
    if grid_mode == "shared":
        return shared_atq(coeff, covariance)
    if grid_mode != "band":
        raise ValueError(grid_mode)
    half = coeff.shape[1] // 2
    low_q, low_t, low_fallback = shared_atq(coeff[:, :half], covariance[:half, :half])
    high_q, high_t, high_fallback = zero_center_atq(coeff[:, half:], covariance[half:, half:])
    return (
        torch.cat((low_q, high_q), dim=1),
        torch.cat((low_t, high_t), dim=1),
        low_fallback or high_fallback,
    )


def tensor_percentile(x: torch.Tensor, q: float) -> float:
    return float(torch.quantile(x.float().flatten(), q).item())


@torch.no_grad()
def measure(weight: torch.Tensor, acts: torch.Tensor, strategy: str, seed: int, grid_mode: str):
    if strategy == "identity":
        coeff, transformed_acts = weight, acts
        order = None
        pair_cosine = None
        hf_ratio = None
        hf_p50 = hf_p90 = hf_p99 = None
    else:
        order = pairing_order(strategy, weight, seed)
        coeff, transformed_acts = haar_transform(weight, acts, order)
        half = coeff.shape[1] // 2
        high = coeff[:, half:]
        total_energy = coeff.float().square().sum().clamp_min(1e-30)
        hf_ratio = float((high.float().square().sum() / total_energy).item())
        high_abs = high.float().abs()
        hf_p50 = tensor_percentile(high_abs, 0.50)
        hf_p90 = tensor_percentile(high_abs, 0.90)
        hf_p99 = tensor_percentile(high_abs, 0.99)
        normed = F.normalize(weight.float(), dim=0)
        pair_cosine = float(
            (normed[:, order[0::2]] * normed[:, order[1::2]]).sum(dim=0).mean().item()
        )

    exact_num = (weight.float() @ acts.float().T - coeff.float() @ transformed_acts.float().T)
    exact_den = (weight.float() @ acts.float().T).norm().clamp_min(1e-30)
    exact_rel = float((exact_num.norm() / exact_den).item())
    effective_grid_mode = "shared" if strategy == "identity" else grid_mode
    quantized, ternary, fallback = quantize_coefficients(coeff, transformed_acts, effective_grid_mode)
    residual = coeff.float() - quantized.float()
    output_ref = coeff.float() @ transformed_acts.float().T
    output_err = residual @ transformed_acts.float().T
    output_nmse = float((output_err.square().sum() / output_ref.square().sum().clamp_min(1e-30)).item())
    weight_nmse = float((residual.square().sum() / coeff.float().square().sum().clamp_min(1e-30)).item())
    half = coeff.shape[1] // 2
    return {
        "strategy": strategy,
        "grid_mode": effective_grid_mode,
        "pair_cosine_mean": pair_cosine,
        "hf_energy_ratio": hf_ratio,
        "hf_abs_p50": hf_p50,
        "hf_abs_p90": hf_p90,
        "hf_abs_p99": hf_p99,
        "ternary_zero_rate": float((ternary == 0).float().mean().item()),
        "high_zero_rate": None if strategy == "identity" else float((ternary[:, half:] == 0).float().mean().item()),
        "weight_nmse": weight_nmse,
        "activation_weighted_nmse": output_nmse,
        "haar_exact_rel_error": exact_rel,
        "atq_fallback": fallback,
    }


def summarize(rows):
    strategies = sorted({row["strategy"] for row in rows})
    summary = {}
    metrics = (
        "pair_cosine_mean",
        "hf_energy_ratio",
        "ternary_zero_rate",
        "high_zero_rate",
        "weight_nmse",
        "activation_weighted_nmse",
        "haar_exact_rel_error",
    )
    for strategy in strategies:
        selected = [r for r in rows if r["strategy"] == strategy]
        entry = {"n": len(selected), "fallbacks": sum(bool(r["atq_fallback"]) for r in selected)}
        for metric in metrics:
            values = [r[metric] for r in selected if r[metric] is not None]
            if values:
                entry[f"median_{metric}"] = statistics.median(values)
                entry[f"mean_{metric}"] = statistics.fmean(values)
        summary[strategy] = entry

    paired = {}
    by_key = {}
    for row in rows:
        by_key[(row["module"], row["block_start"], row["strategy"])] = row
    keys = sorted({(r["module"], r["block_start"]) for r in rows})
    for candidate in ("adjacent", "dissimilar", "ssr_order", "similarity"):
        hf_better = []
        err_better = []
        err_improvements = []
        for module, block_start in keys:
            cand = by_key[(module, block_start, candidate)]
            rnd = by_key[(module, block_start, "random")]
            hf_better.append(cand["hf_energy_ratio"] < rnd["hf_energy_ratio"])
            err_better.append(cand["activation_weighted_nmse"] < rnd["activation_weighted_nmse"])
            err_improvements.append(
                (rnd["activation_weighted_nmse"] - cand["activation_weighted_nmse"])
                / max(rnd["activation_weighted_nmse"], 1e-30)
            )
        paired[candidate] = {
            "n": len(keys),
            "hf_lower_fraction_vs_random": statistics.fmean(hf_better),
            "weighted_error_lower_fraction_vs_random": statistics.fmean(err_better),
            "median_weighted_error_improvement_vs_random": statistics.median(err_improvements),
        }
    sim = paired["similarity"]
    gate = {
        "hf_lower_fraction_target": 0.70,
        "weighted_error_median_improvement_target": 0.05,
        "pass": sim["hf_lower_fraction_vs_random"] >= 0.70
        and sim["median_weighted_error_improvement_vs_random"] >= 0.05,
    }
    return {"by_strategy": summary, "paired_vs_random": paired, "gate": gate}


def parse_layers(text: str):
    return [int(x) for x in text.split(",") if x]


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
    parser.add_argument("--grid-mode", choices=("shared", "band"), default="shared")
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
    strategies = ("identity", "adjacent", "random", "dissimilar", "ssr_order", "similarity")
    total = len(weights) * args.blocks_per_module * len(strategies)
    completed = 0
    for module_name in sorted(weights):
        weight_cpu = weights[module_name]
        acts_cpu = activations[module_name]
        in_features = weight_cpu.shape[1]
        max_start = in_features - args.blocksize
        if args.blocks_per_module == 1:
            starts = [0]
        else:
            starts = [int(round(i * max_start / (args.blocks_per_module - 1)) // args.blocksize * args.blocksize)
                      for i in range(args.blocks_per_module)]
        starts = sorted(set(min(max_start, max(0, s)) for s in starts))
        for block_start in starts:
            block_end = block_start + args.blocksize
            weight = weight_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            acts = acts_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            for strategy in strategies:
                row = measure(
                    weight,
                    acts,
                    strategy,
                    stable_seed(f"{module_name}:{block_start}:{strategy}", args.seed),
                    args.grid_mode,
                )
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

    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summarize(rows),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
