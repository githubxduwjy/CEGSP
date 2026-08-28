#!/usr/bin/env python3
"""CEGSP v2 P1: one-gradient ternary move-space comparison.

This follows CEGSP-V2-P0.  The goal is not to pivot; it tests whether CEGSP's
gap to one-step QAT is caused by an overly narrow discrete move space.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_ce_gradient_4090 import (
    collect_ce_qk_grads,
    projection_weight,
    set_projection_weight,
    target_modules,
)
from tqgsp_support_projection_4090 import (
    apply_direct_ptq_local,
    build_wikitext_splits,
    direct_ternary_weight,
    evaluate_nll,
    gradient_projection_candidates_unique,
    gradient_signflip_candidates,
    log,
    make_code,
    parse_csv_ints,
    run_one_shot,
    run_one_shot_flips,
    weight_from_state,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="CEGSP-V2-P1-MOVE-SPACE-OPT125M")
    p.add_argument("--model", default="facebook/opt-125m")
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=16)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--rho", type=float, default=0.7)
    p.add_argument("--edits", type=int, default=64)
    p.add_argument("--topks", default="3,6")
    p.add_argument("--qat-etas", default="0,0.01,0.03,0.1,0.3,1.0")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def parse_floats(text: str) -> List[float]:
    return [float(x) for x in text.split(",") if x]


def snapshot_qk(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        layer: {
            key: projection_weight(target_modules(model, layer)[key]).detach().float().cpu().clone()
            for key in ("q", "k")
        }
        for layer in layers
    }


def restore_qk(model: torch.nn.Module, layers: List[int], weights: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer in layers:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            set_projection_weight(refs[key], weights[layer][key])


def apply_patch_set(
    model: torch.nn.Module,
    layers: List[int],
    direct: Dict[int, Dict[str, torch.Tensor]],
    edited: Dict[int, Dict[str, torch.Tensor]],
    selected: List[int],
) -> None:
    restore_qk(model, layers, direct)
    for layer in selected:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            set_projection_weight(refs[key], edited[layer][key])


def build_edits(
    fp: Dict[int, Dict[str, torch.Tensor]],
    grads: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    rho: float,
    edits: int,
) -> Tuple[Dict[str, Dict[int, Dict[str, torch.Tensor]]], List[Dict[str, object]]]:
    variants = {"support": {}, "support_refit": {}, "signflip": {}}
    meta = []
    for layer in layers:
        codes = {key: make_code(fp[layer][key], group_size, rho) for key in ("q", "k")}
        base_states = {key: code.state.clone() for key, code in codes.items()}

        support_candidates = gradient_projection_candidates_unique(codes, grads[layer], "qk", edits)
        support_states, support_trace = run_one_shot(base_states, codes, support_candidates, edits)
        variants["support"][layer] = {
            key: weight_from_state(codes[key], support_states[key], refit=False) for key in ("q", "k")
        }
        variants["support_refit"][layer] = {
            key: weight_from_state(codes[key], support_states[key], refit=True) for key in ("q", "k")
        }

        flip_candidates = gradient_signflip_candidates(codes, grads[layer], "qk", edits)
        flip_states, flip_trace = run_one_shot_flips(base_states, flip_candidates, edits)
        variants["signflip"][layer] = {
            key: weight_from_state(codes[key], flip_states[key], refit=False) for key in ("q", "k")
        }
        meta.append(
            {
                "layer": layer,
                "support_edits": len(support_trace),
                "signflip_edits": len(flip_trace),
                "support_top_scores": [float(c.score) for c in support_candidates[:5]],
                "signflip_top_scores": [float(c.score) for c in flip_candidates[:5]],
            }
        )
    return variants, meta


def one_step_qat(
    model: torch.nn.Module,
    fit: List[torch.Tensor],
    val: List[torch.Tensor],
    untouched: List[torch.Tensor],
    device: torch.device,
    fp: Dict[int, Dict[str, torch.Tensor]],
    direct: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    rho: float,
    etas: List[float],
) -> Dict[str, object]:
    rows = []
    for eta in etas:
        latent = {layer: {key: fp[layer][key].clone() for key in ("q", "k")} for layer in layers}
        q0 = {}
        for layer in layers:
            q0[layer] = {}
            for key in ("q", "k"):
                q0[layer][key] = direct_ternary_weight(latent[layer][key], group_size, rho)[0].float().cpu()
        apply_patch_set(model, layers, direct, q0, layers)
        grads = collect_ce_qk_grads(model, fit, layers, device, 1)
        for layer in layers:
            for key in ("q", "k"):
                scale = fp[layer][key].float().std().clamp_min(1e-8) / grads[layer][key].float().std().clamp_min(1e-12)
                latent[layer][key] = latent[layer][key] - float(eta) * scale * grads[layer][key]
        q1 = {}
        for layer in layers:
            q1[layer] = {}
            for key in ("q", "k"):
                q1[layer][key] = direct_ternary_weight(latent[layer][key], group_size, rho)[0].float().cpu()
        apply_patch_set(model, layers, direct, q1, layers)
        rows.append(
            {
                "eta": eta,
                "val": evaluate_nll(model, val, device),
                "untouched_w": evaluate_nll(model, untouched, device),
            }
        )
    restore_qk(model, layers, direct)
    return {"rows": rows, "best": min(rows, key=lambda r: r["val"])}


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    layers = parse_csv_ints(args.layers)
    topks = parse_csv_ints(args.topks)
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading {args.model} for {args.run_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, data_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model.config.use_cache = False

    fp = snapshot_qk(model, layers)
    fp_nll = {"val": evaluate_nll(model, val, device), "untouched_w": evaluate_nll(model, untouched, device)}
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.rho)
    direct = snapshot_qk(model, layers)
    direct_nll = {"val": evaluate_nll(model, val, device), "untouched_w": evaluate_nll(model, untouched, device)}
    grads = collect_ce_qk_grads(model, fit, layers, device, 1)

    edits, edit_meta = build_edits(fp, grads, layers, args.group_size, args.rho, args.edits)
    per_layer = []
    for layer in layers:
        row = {"layer": layer}
        for name, weights in edits.items():
            apply_patch_set(model, layers, direct, weights, [layer])
            val_nll = evaluate_nll(model, val, device)
            row[f"{name}_single_val"] = val_nll
            row[f"{name}_single_delta_val"] = val_nll - direct_nll["val"]
        per_layer.append(row)
    restore_qk(model, layers, direct)

    ranked = {
        name: [int(r["layer"]) for r in sorted(per_layer, key=lambda r, n=name: r[f"{n}_single_delta_val"])]
        for name in edits
    }
    mixed_rank = []
    for row in per_layer:
        best_name = min(edits.keys(), key=lambda n: row[f"{n}_single_delta_val"])
        mixed_rank.append((int(row["layer"]), best_name, row[f"{best_name}_single_delta_val"]))
    mixed_rank.sort(key=lambda x: x[2])

    patch_results = {}
    for k in topks:
        kk = max(0, min(k, len(layers)))
        for name, weights in edits.items():
            selected = ranked[name][:kk]
            apply_patch_set(model, layers, direct, weights, selected)
            scores = {"val": evaluate_nll(model, val, device), "untouched_w": evaluate_nll(model, untouched, device)}
            patch_results[f"{name}-top{kk}"] = {
                "selected_layers": selected,
                "nll": scores,
                "delta_val": scores["val"] - direct_nll["val"],
                "delta_untouched_w": scores["untouched_w"] - direct_nll["untouched_w"],
            }
        restore_qk(model, layers, direct)
        selected_mixed = mixed_rank[:kk]
        restore_qk(model, layers, direct)
        for layer, name, _ in selected_mixed:
            refs = target_modules(model, layer)
            for key in ("q", "k"):
                set_projection_weight(refs[key], edits[name][layer][key])
        scores = {"val": evaluate_nll(model, val, device), "untouched_w": evaluate_nll(model, untouched, device)}
        patch_results[f"mixed-top{kk}"] = {
            "selected": [{"layer": l, "edit": n} for l, n, _ in selected_mixed],
            "nll": scores,
            "delta_val": scores["val"] - direct_nll["val"],
            "delta_untouched_w": scores["untouched_w"] - direct_nll["untouched_w"],
        }
    restore_qk(model, layers, direct)

    qat = one_step_qat(
        model,
        fit,
        val,
        untouched,
        device,
        fp,
        direct,
        layers,
        args.group_size,
        args.rho,
        parse_floats(args.qat_etas),
    )
    best_patch_name = min(patch_results, key=lambda n: patch_results[n]["nll"]["val"])
    one_step_gain = direct_nll["untouched_w"] - qat["best"]["untouched_w"]
    best_patch_gain = direct_nll["untouched_w"] - patch_results[best_patch_name]["nll"]["untouched_w"]
    closure_vs_one_step = best_patch_gain / one_step_gain if abs(one_step_gain) > 1e-12 else float("nan")

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "config": vars(args),
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_batches": len(untouched),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_mem": int(torch.cuda.max_memory_allocated()),
        },
        "clean_room": {
            "cegsp_uses_teacher": False,
            "cegsp_uses_optimizer": False,
            "qat_control_uses_latent_fp": True,
        },
        "quant_counts": quant_counts,
        "nll": {
            "fp": fp_nll,
            "direct": direct_nll,
            "patch_sets": patch_results,
            "one_step_qat": qat,
        },
        "best_patch_by_val": best_patch_name,
        "closure_vs_one_step_qat_untouched": closure_vs_one_step,
        "per_layer": per_layer,
        "edit_meta": edit_meta,
        "elapsed_sec": time.time() - started,
        "status": "complete",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
