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
from transformers import AutoModelForCausalLM

from haar_mechanism_diagnostics import shared_atq
from pt2_llm.data import get_loaders


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "little") + base) % (2**31)


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones((1, 1), device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(n)


@torch.no_grad()
def random_orthogonal(n: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    mat = torch.randn((n, n), generator=gen, device=device, dtype=torch.float32)
    q, r = torch.linalg.qr(mat)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs


@torch.no_grad()
def signed_hadamard(n: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    h = hadamard(n, device)
    left = torch.randint(0, 2, (n,), generator=gen, device=device, dtype=torch.float32) * 2 - 1
    right = torch.randint(0, 2, (n,), generator=gen, device=device, dtype=torch.float32) * 2 - 1
    return left[:, None] * h * right[None, :]


@torch.no_grad()
def ordered_hadamard(weight: torch.Tensor, acts: torch.Tensor, mode: str) -> torch.Tensor:
    n = weight.shape[1]
    if mode == "weight_norm_hadamard":
        score = weight.float().square().sum(dim=0)
    elif mode == "activation_rms_hadamard":
        score = acts.float().square().mean(dim=0)
    elif mode == "reverse_activation_rms_hadamard":
        score = -acts.float().square().mean(dim=0)
    elif mode == "joint_norm_hadamard":
        score = weight.float().square().sum(dim=0) * acts.float().square().mean(dim=0)
    else:
        raise ValueError(mode)
    order = torch.argsort(score, descending=True)
    perm = torch.eye(n, device=weight.device, dtype=torch.float32)[order]
    return perm.T @ hadamard(n, weight.device)


@torch.no_grad()
def rotation_matrix(strategy: str, weight: torch.Tensor, acts: torch.Tensor, seed: int) -> torch.Tensor | None:
    n = weight.shape[1]
    if strategy == "identity":
        return None
    if strategy == "random_orthogonal":
        return random_orthogonal(n, seed, weight.device)
    if strategy == "hadamard":
        return hadamard(n, weight.device)
    if strategy == "signed_hadamard":
        return signed_hadamard(n, seed, weight.device)
    if strategy == "random_perm_hadamard":
        gen = torch.Generator(device=weight.device)
        gen.manual_seed(seed)
        order = torch.randperm(n, generator=gen, device=weight.device)
        perm = torch.eye(n, device=weight.device, dtype=torch.float32)[order]
        return perm.T @ hadamard(n, weight.device)
    if strategy == "activation_rms_permutation":
        score = acts.float().square().mean(dim=0)
        order = torch.argsort(score, descending=True)
        return torch.eye(n, device=weight.device, dtype=torch.float32)[order].T
    if strategy in {"weight_norm_hadamard", "activation_rms_hadamard", "reverse_activation_rms_hadamard", "joint_norm_hadamard"}:
        return ordered_hadamard(weight, acts, strategy)
    raise ValueError(strategy)


@torch.no_grad()
def measure_rotation(weight: torch.Tensor, acts: torch.Tensor, strategy: str, seed: int):
    rot = rotation_matrix(strategy, weight, acts, seed)
    if rot is None:
        coeff = weight.float()
        transformed_acts = acts.float()
        exact_rel = 0.0
        orthogonal_rel = 0.0
    else:
        coeff = weight.float() @ rot
        transformed_acts = acts.float() @ rot
        exact_num = weight.float() @ acts.float().T - coeff @ transformed_acts.T
        exact_den = (weight.float() @ acts.float().T).norm().clamp_min(1e-30)
        exact_rel = float((exact_num.norm() / exact_den).item())
        ident = torch.eye(rot.shape[0], device=rot.device)
        orthogonal_rel = float(((rot.T @ rot - ident).norm() / ident.norm()).item())

    covariance = transformed_acts.float().T @ transformed_acts.float()
    covariance /= max(1, transformed_acts.shape[0])
    quantized, ternary, fallback = shared_atq(coeff, covariance)
    residual = coeff.float() - quantized.float()
    output_ref = coeff.float() @ transformed_acts.float().T
    output_err = residual @ transformed_acts.float().T
    output_nmse = float((output_err.square().sum() / output_ref.square().sum().clamp_min(1e-30)).item())
    coeff_nmse = float((residual.square().sum() / coeff.float().square().sum().clamp_min(1e-30)).item())
    if rot is None:
        reconstructed = quantized
    else:
        reconstructed = quantized @ rot.T
    inverse_weight_nmse = float(
        ((weight.float() - reconstructed.float()).square().sum() / weight.float().square().sum().clamp_min(1e-30)).item()
    )
    return {
        "strategy": strategy,
        "ternary_zero_rate": float((ternary == 0).float().mean().item()),
        "coeff_nmse": coeff_nmse,
        "inverse_weight_nmse": inverse_weight_nmse,
        "activation_weighted_nmse": output_nmse,
        "rotation_exact_rel_error": exact_rel,
        "orthogonal_rel_error": orthogonal_rel,
        "atq_fallback": fallback,
    }


def parse_layers(text: str):
    return [int(x) for x in text.split(",") if x]


def median(values):
    return statistics.median(values)


def mean(values):
    return statistics.fmean(values)


def summarize(rows):
    strategies = sorted({row["strategy"] for row in rows})
    by_strategy = {}
    for strategy in strategies:
        selected = [r for r in rows if r["strategy"] == strategy]
        by_strategy[strategy] = {
            "n": len(selected),
            "fallbacks": sum(bool(r["atq_fallback"]) for r in selected),
            "median_ternary_zero_rate": median([r["ternary_zero_rate"] for r in selected]),
            "median_coeff_nmse": median([r["coeff_nmse"] for r in selected]),
            "median_inverse_weight_nmse": median([r["inverse_weight_nmse"] for r in selected]),
            "median_activation_weighted_nmse": median([r["activation_weighted_nmse"] for r in selected]),
            "mean_activation_weighted_nmse": mean([r["activation_weighted_nmse"] for r in selected]),
            "median_rotation_exact_rel_error": median([r["rotation_exact_rel_error"] for r in selected]),
            "median_orthogonal_rel_error": median([r["orthogonal_rel_error"] for r in selected]),
        }
    by_key = {(r["module"], r["block_start"], r["strategy"]): r for r in rows}
    block_keys = sorted({(r["module"], r["block_start"]) for r in rows})
    paired_vs_identity = {}
    for strategy in strategies:
        if strategy == "identity":
            continue
        improvements = []
        wins = []
        for module, block_start in block_keys:
            cand = by_key[(module, block_start, strategy)]
            ident = by_key[(module, block_start, "identity")]
            improvement = (
                ident["activation_weighted_nmse"] - cand["activation_weighted_nmse"]
            ) / max(ident["activation_weighted_nmse"], 1e-30)
            improvements.append(improvement)
            wins.append(improvement > 0)
        paired_vs_identity[strategy] = {
            "n": len(improvements),
            "median_weighted_error_improvement_vs_identity": median(improvements),
            "mean_weighted_error_improvement_vs_identity": mean(improvements),
            "weighted_error_win_fraction_vs_identity": mean(wins),
        }
    structured = [
        "hadamard",
        "signed_hadamard",
        "random_perm_hadamard",
        "weight_norm_hadamard",
        "activation_rms_hadamard",
        "reverse_activation_rms_hadamard",
        "joint_norm_hadamard",
    ]
    best_name = max(
        structured,
        key=lambda s: paired_vs_identity[s]["median_weighted_error_improvement_vs_identity"],
    )
    gate = {
        "candidate": best_name,
        "target_median_weighted_error_improvement_vs_identity": 0.05,
        "target_win_fraction_vs_identity": 0.70,
        "pass": paired_vs_identity[best_name]["median_weighted_error_improvement_vs_identity"] >= 0.05
        and paired_vs_identity[best_name]["weighted_error_win_fraction_vs_identity"] >= 0.70,
    }
    return {"by_strategy": by_strategy, "paired_vs_identity": paired_vs_identity, "gate": gate}


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

    strategies = (
        "identity",
        "random_orthogonal",
        "hadamard",
        "signed_hadamard",
        "random_perm_hadamard",
        "activation_rms_permutation",
        "weight_norm_hadamard",
        "activation_rms_hadamard",
        "reverse_activation_rms_hadamard",
        "joint_norm_hadamard",
    )
    rows = []
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
            starts = [
                int(round(i * max_start / (args.blocks_per_module - 1)) // args.blocksize * args.blocksize)
                for i in range(args.blocks_per_module)
            ]
        starts = sorted(set(min(max_start, max(0, s)) for s in starts))
        for block_start in starts:
            block_end = block_start + args.blocksize
            weight = weight_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            acts = acts_cpu[:, block_start:block_end].to("cuda:0", dtype=torch.float32)
            for strategy in strategies:
                row = measure_rotation(
                    weight,
                    acts,
                    strategy,
                    stable_seed(f"{module_name}:{block_start}:{strategy}", args.seed),
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
                if completed % 28 == 0 or completed == total:
                    print(f"Measured {completed}/{total}", flush=True)
            del weight, acts
            torch.cuda.empty_cache()

    summary = summarize(rows)
    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "rows": rows,
        "summary": summary,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
