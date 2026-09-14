#!/usr/bin/env python3
"""Publication-faithful ReQuant refinement rule on the frozen ternary Q0.

This runner implements the published fixed-grid coordinate-refinement rule:
four deterministic sweeps, K=2 neighboring grid moves, strict acceptance only
when the exact quadratic reconstruction delta is negative, and incremental
gradient refresh after each accepted update.  It deliberately keeps the
project's frozen affine ternary initializer and Q/K scope, so the result is a
faithful refinement-rule comparison, not a claim that the entire ReQuant PTQ
initializer was reproduced.

Two variants are run sequentially on one RTX 4090:
  * matched: the existing TernRefine calibration protocol (8 x 128, batch 2)
  * paper_default_budget: 512 WikiText-2 sequences of length 2048

Statistics, module states, and per-variant progress are checkpointed so a
harness interruption does not require repeating completed work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_p11_p12_4090 import build_c4_cached_batches
from cegsp_p5a_affine_adapter_feasibility_4090 import (
    AffineCode,
    affine_weight,
    apply_affine_patch,
    audit_all,
    eval_metrics,
    make_affine_code,
    projection_weight,
    restore_qk,
    snapshot_qk,
    target_modules,
    with_ppl,
)
from cegsp_requant_comparison_4090 import reconstruction_gradient, run_coordinate_descent
from tqgsp_support_projection_4090 import build_wikitext_splits, log, parse_csv_ints


ALL_LAYERS = list(range(24))
RESEARCH_LAYERS = [3, 4, 5, 6, 7, 8]


def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--layers", default=",".join(map(str, ALL_LAYERS)))
    p.add_argument("--requant-layers", default=",".join(map(str, RESEARCH_LAYERS)))
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--sweeps", type=int, default=4)
    p.add_argument("--neighbor-k", type=int, default=2)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def finite_metrics(metrics: Dict[str, Dict[str, float]]) -> bool:
    return all(
        math.isfinite(float(v["nll"])) and math.isfinite(float(v["ppl"]))
        for v in metrics.values()
    )


def nll_delta(metrics: Dict[str, Dict[str, float]], baseline: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {k: float(metrics[k]["nll"] - baseline[k]["nll"]) for k in metrics}


def state_hash(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(states):
        for key in ("q", "k"):
            digest.update(str(layer).encode())
            digest.update(key.encode())
            digest.update(states[layer][key].cpu().numpy().tobytes())
    return digest.hexdigest()


def changed_coordinates(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> int:
    return sum(
        int((states[layer][key] != code.T).sum().item())
        for layer, layer_codes in codes.items()
        for key, code in layer_codes.items()
    )


def cardinality_violations(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> int:
    total = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            before = code.T.abs().sum(dim=-1)
            after = states[layer][key].abs().sum(dim=-1)
            total += int((before != after).sum().item())
    return total


def metric_summary(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        k: {"nll": float(v["nll"]), "ppl": float(v["ppl"])}
        for k, v in metrics.items()
    }


def capture_streaming_stats(
    model: torch.nn.Module,
    batches: Sequence[torch.Tensor],
    layers: Sequence[int],
    device: torch.device,
    checkpoint: Path,
    final_path: Path,
) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    """Accumulate H=Xe Xe^T and B=(Xe-X)Xe^T without storing all tokens."""
    if final_path.exists():
        return torch.load(final_path, map_location="cpu", weights_only=False)

    dim = None
    accum_h: Dict[Tuple[int, str], torch.Tensor] = {}
    accum_b: Dict[Tuple[int, str], torch.Tensor] = {}
    token_counts: Dict[Tuple[int, str], int] = {}
    completed = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        completed = int(saved["completed_batches"])
        for k, v in saved["H"].items():
            accum_h[tuple(k)] = v.to(device)
        for k, v in saved["B"].items():
            accum_b[tuple(k)] = v.to(device)
        token_counts = {tuple(k): int(v) for k, v in saved["tokens"].items()}
        log(f"resuming streaming statistics at batch {completed}/{len(batches)}")

    current: Dict[Tuple[int, str], torch.Tensor] = {}
    hooks = []
    for layer in layers:
        refs = target_modules(model, int(layer))
        for key in ("q", "k"):
            def hook(_module, args, _layer=int(layer), _key=str(key)):
                if not args:
                    raise RuntimeError(f"missing input for L{_layer}.{_key}")
                x = args[0].detach().float()
                if x.ndim != 3:
                    raise RuntimeError(f"unexpected input shape for L{_layer}.{_key}: {tuple(x.shape)}")
                current[(_layer, _key)] = x.reshape(-1, x.shape[-1]).contiguous()
            hooks.append(refs[key].module.register_forward_pre_hook(hook))

    try:
        model.eval()
        with torch.no_grad():
            for idx in range(completed, len(batches)):
                current.clear()
                model(input_ids=batches[idx][:, :-1].to(device), use_cache=False)
                full = {k: v for k, v in current.items()}
                current.clear()
                model(input_ids=batches[idx][:, :-1].to(device), use_cache=False)
                quant = {k: v for k, v in current.items()}
                for layer in layers:
                    for key in ("q", "k"):
                        token_key = (int(layer), str(key))
                        x = full[token_key]
                        xe = quant[token_key]
                        if dim is None:
                            dim = int(x.shape[-1])
                        if token_key not in accum_h:
                            accum_h[token_key] = torch.zeros((dim, dim), dtype=torch.float32, device=device)
                            accum_b[token_key] = torch.zeros((dim, dim), dtype=torch.float32, device=device)
                            token_counts[token_key] = 0
                        accum_h[token_key].add_(xe.T @ xe)
                        accum_b[token_key].add_((xe - x).T @ xe)
                        token_counts[token_key] += int(x.shape[0])
                if (idx + 1) % 16 == 0 or idx + 1 == len(batches):
                    torch.save(
                        {
                            "completed_batches": idx + 1,
                            "H": {k: v.detach().cpu() for k, v in accum_h.items()},
                            "B": {k: v.detach().cpu() for k, v in accum_b.items()},
                            "tokens": {k: v for k, v in token_counts.items()},
                        },
                        checkpoint,
                    )
                    log(f"stats progress {idx + 1}/{len(batches)}")
                del full, quant
    finally:
        for hook in hooks:
            hook.remove()

    result: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for layer in layers:
        result[layer] = {}
        for key in ("q", "k"):
            token_key = (int(layer), str(key))
            result[layer][key] = {
                "H": accum_h[token_key].detach().cpu().contiguous(),
                "B": accum_b[token_key].detach().cpu().contiguous(),
                "tokens": torch.tensor(token_counts[token_key]),
            }
    torch.save(result, final_path)
    return result


def load_or_build_data(tokenizer, args: argparse.Namespace):
    eval_seq = 128
    eval_batch = 2
    val, w2, c4, source = None, None, None, None
    # The evaluator protocol is frozen to the existing P12 W2/C4 setup.
    _, val, w2, source = build_wikitext_splits(tokenizer, eval_seq, eval_batch, 8, 8, 8, 0, 0)
    c4 = build_c4_cached_batches(tokenizer, eval_seq, eval_batch, 8, 0)
    return val, w2, c4, source


def variant_spec() -> Dict[str, Dict[str, int]]:
    return {
        "matched": {"fit_seq_len": 128, "fit_batch_size": 2, "fit_batches": 8, "fit_sequences": 16},
        "paper_default_budget": {"fit_seq_len": 2048, "fit_batch_size": 2, "fit_batches": 256, "fit_sequences": 512},
    }


def run_variant(
    name: str,
    spec: Dict[str, int],
    model: torch.nn.Module,
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    codes: Dict[int, Dict[str, AffineCode]],
    affine_states: Dict[int, Dict[str, torch.Tensor]],
    baseline_metrics: Dict[str, Dict[str, float]],
    val,
    w2,
    c4,
    source: str,
    out_dir: Path,
) -> Dict[str, object]:
    final_path = out_dir / f"requant_faithful_{name}.json"
    if final_path.exists():
        log(f"reusing completed variant {name}")
        return json.loads(final_path.read_text())

    fit, _, _, _ = build_wikitext_splits(
        tokenizer,
        spec["fit_seq_len"],
        spec["fit_batch_size"],
        spec["fit_batches"],
        0,
        0,
        0,
        0,
    )
    log(f"variant={name} fit_batches={len(fit)} seq={spec['fit_seq_len']} sequences={spec['fit_sequences']}")
    restore_qk(model, fp_qk)
    stats = capture_streaming_stats(
        model,
        fit,
        RESEARCH_LAYERS,
        device,
        out_dir / f"stats_{name}_partial.pt",
        out_dir / f"stats_{name}.pt",
    )

    states: Dict[int, Dict[str, torch.Tensor]] = {
        layer: {key: code.T.clone() for key, code in cs.items()}
        for layer, cs in codes.items()
    }
    module_order = [(layer, key) for layer in RESEARCH_LAYERS for key in ("q", "k")]
    partial_path = out_dir / f"requant_faithful_{name}_partial.json"
    state_ckpt = out_dir / f"state_{name}.pt"
    infos: Dict[str, object] = {}
    cursor = 0
    if state_ckpt.exists() and partial_path.exists():
        saved = torch.load(state_ckpt, map_location="cpu", weights_only=False)
        for layer in states:
            for key in states[layer]:
                states[layer][key] = saved["states"][f"L{layer}.{key}"].clone()
        partial = json.loads(partial_path.read_text())
        cursor = int(partial["completed_modules"])
        infos = partial["module_infos"]
        log(f"resuming {name} at module {cursor}/{len(module_order)}")

    apply_affine_patch(model, codes, affine_states)
    for index in range(cursor, len(module_order)):
        layer, key = module_order[index]
        info = run_coordinate_descent(
            layer,
            key,
            codes[layer][key],
            fp_qk[layer][key],
            stats[layer][key],
            states[layer][key],
            args.sweeps,
        )
        infos[f"L{layer}.{key}"] = info
        torch.save(
            {"states": {f"L{l}.{k}": states[l][k].cpu() for l in states for k in states[l]}},
            state_ckpt,
        )
        partial_path.write_text(json.dumps({"completed_modules": index + 1, "module_infos": infos}, indent=2))
        log(f"{name} module {index + 1}/{len(module_order)} complete L{layer}.{key} accepted={info['accepted_updates']}")

    apply_affine_patch(model, codes, states)
    metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
    audit = audit_all(codes, states)
    result = {
        "variant": name,
        "calibration": {
            "source": source,
            "fit_batches": spec["fit_batches"],
            "fit_sequences": spec["fit_sequences"],
            "seq_len": spec["fit_seq_len"],
            "batch_size": spec["fit_batch_size"],
        },
        "published_rule": {
            "sweeps_T": args.sweeps,
            "neighbor_K": args.neighbor_k,
            "acceptance": "strict quadratic reconstruction delta < 0",
            "coordinate_order": "row-major coordinate cycle, repeated T sweeps",
            "incremental_gradient_refresh": True,
        },
        "initializer_scope": {
            "same_affine_ternary_Q0": True,
            "fixed_mu_alpha": True,
            "layers": RESEARCH_LAYERS,
            "modules": [f"L{layer}.{key}" for layer in RESEARCH_LAYERS for key in ("q", "k")],
            "original_requant_initializer_reproduced": False,
            "initializer_note": "The paper uses per-channel asymmetric integer quantization; this comparison freezes the project's affine ternary Q0 as requested.",
        },
        "module_infos": infos,
        "metrics": metric_summary(metrics),
        "delta_vs_affine_nll": nll_delta(metrics, baseline_metrics),
        "changed_coordinates": changed_coordinates(codes, states),
        "cardinality_violations": cardinality_violations(codes, states),
        "state_hash": state_hash(codes, states),
        "audit": audit,
        "finite": finite_metrics(metrics),
    }
    final_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    restore_qk(model, fp_qk)
    return result


def main() -> None:
    args = args_parser()
    start = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("faithful ReQuant requires CUDA")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    layers = parse_csv_ints(args.layers)
    requant_layers = parse_csv_ints(args.requant_layers)
    if layers != ALL_LAYERS or requant_layers != RESEARCH_LAYERS:
        raise ValueError("frozen protocol requires all OPT-350M layers and ReQuant layers 3--8")
    if args.neighbor_k != 2 or args.sweeps != 4:
        raise ValueError("publication-faithful run requires T=4 and K=2")

    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"publication-faithful ReQuant rule on frozen ternary Q0; gpu={torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    val, w2, c4, source = load_or_build_data(tokenizer, args)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.eval()
    fp_qk = snapshot_qk(model, layers)
    codes = {
        layer: {key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor) for key in ("q", "k")}
        for layer in layers
    }
    affine_states = {layer: {key: code.T.clone() for key, code in cs.items()} for layer, cs in codes.items()}
    apply_affine_patch(model, codes, affine_states)
    baseline_metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
    log(f"shared affine baseline={baseline_metrics}")

    variants = {}
    for name, spec in variant_spec().items():
        variants[name] = run_variant(
            name,
            spec,
            model,
            tokenizer,
            args,
            device,
            fp_qk,
            codes,
            affine_states,
            baseline_metrics,
            val,
            w2,
            c4,
            source,
            out_dir,
        )

    result = {
        "run_id": args.run_id,
        "status": "complete",
        "experiment": "publication-faithful ReQuant refinement rule on frozen affine ternary Q0",
        "config": vars(args),
        "protocol": {
            "same_affine_initializer": True,
            "same_model_tokenizer_evaluator": True,
            "scope": "OPT-350M Q/K layers 3--8",
            "codebook": "Q=mu+alpha*T, T in {-1,0,+1}",
            "fixed_mu_alpha": True,
            "published_refinement_rule": "Algorithm 1: exact quadratic row loss, K=2, T=4, strict negative-delta acceptance, incremental g refresh",
            "full_paper_initializer_reproduced": False,
            "initializer_difference": "published ReQuant uses per-channel asymmetric integer quantization; this run freezes the requested affine ternary Q0",
        },
        "evaluator": {
            "wikitext_source": source,
            "val_batches": 8,
            "w2_batches": 8,
            "c4_batches": 8,
            "seq_len": 128,
            "batch_size": 2,
            "split": "same as existing P12 evaluator",
        },
        "baseline": {"affine": metric_summary(baseline_metrics), "audit": audit_all(codes, affine_states), "state_hash": state_hash(codes, affine_states)},
        "variants": variants,
        "integrity": {
            "baseline_shared": True,
            "both_variants_finite": all(bool(v["finite"]) for v in variants.values()),
            "both_variants_codebook_legal": all(int(v["audit"]["total_illegal_states"]) == 0 for v in variants.values()),
            "paper_rule_T4_K2": args.sweeps == 4 and args.neighbor_k == 2,
            "not_official_full_pipeline": True,
        },
        "timing": {"total_sec": time.time() - start},
    }
    (out_dir / "requant_faithful_comparison_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    restore_qk(model, fp_qk)
    log(json.dumps(result["integrity"], indent=2))


if __name__ == "__main__":
    main()
