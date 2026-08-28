#!/usr/bin/env python3
"""CEGSP v2 P4R-Fairness: strict edit matching and QAT update scope.

This is a small fairness check after P4-R. It does not change CEGSP. It only
checks whether one-step latent QAT can match CEGSP's discrete edit budget more
strictly, and whether restricting QAT updates to CEGSP-selected layers changes
the conclusion.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_v2_p4r_qat_transition_4090 import (
    direct_states_from_fp,
    eval_current,
    metric_pack,
    parse_float_list,
    quantize_latent_qk,
    transition_counts,
)
from cegsp_v2_p4_gap_cost_4090 import (
    apply_qk_patch,
    build_c4_untouched_batches,
    build_cegsp_layer_patches,
    normalized_latent_step,
    snapshot_qk,
    restore_qk,
)
from cegsp_ce_gradient_4090 import collect_ce_qk_grads, target_modules, projection_weight
from tqgsp_support_projection_4090 import (
    apply_direct_ptq_local,
    build_wikitext_splits,
    evaluate_nll,
    log,
    parse_csv_ints,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=64)
    p.add_argument("--fit-token-offset", type=int, default=8192)
    p.add_argument("--val-token-offset", type=int, default=8192)
    p.add_argument("--c4-untouched-batches", type=int, default=32)
    p.add_argument("--c4-token-offset", type=int, default=16384)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--max-edits", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--layer-topk", type=int, default=6)
    p.add_argument("--etas", default="3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,1e-4,1.5e-4,2e-4,3e-4,5e-4,7e-4,1e-3")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def run_one_step_scope(
    model,
    fit,
    val,
    untouched_w,
    untouched_c4,
    device,
    fp_qk,
    direct_qk,
    direct_states,
    all_layers: List[int],
    update_layers: List[int],
    group_size: int,
    threshold_factor: float,
    etas: List[float],
    target_changes: int,
    scope_name: str,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    restore_qk(model, all_layers, direct_qk)
    grads = collect_ce_qk_grads(model, fit, update_layers, device, grad_batches=1)
    for eta in etas:
        latent = {layer: {key: fp_qk[layer][key].clone() for key in ("q", "k")} for layer in update_layers}
        latent = normalized_latent_step(latent, fp_qk, grads, update_layers, float(eta))
        quant, states = quantize_latent_qk(latent, update_layers, group_size, threshold_factor)
        apply_qk_patch(model, all_layers, direct_qk, quant, update_layers)
        nll = eval_current(model, val, untouched_w, untouched_c4, device)
        scoped_before = {layer: direct_states[layer] for layer in update_layers}
        counts = transition_counts(scoped_before, states, update_layers)
        rows.append(
            {
                "scope": scope_name,
                "eta": float(eta),
                "nll": nll,
                "with_ppl": metric_pack(nll),
                "transition_counts": counts,
                "abs_change_error_vs_cegsp": abs(int(counts["changed_total"]) - int(target_changes)),
            }
        )
    restore_qk(model, all_layers, direct_qk)
    return {
        "scope": scope_name,
        "rows": rows,
        "validation_best": min(rows, key=lambda r: float(r["nll"]["val"])),
        "edit_matched_to_cegsp": min(rows, key=lambda r: int(r["abs_change_error_vs_cegsp"])),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_csv_ints(args.layers)
    etas = parse_float_list(args.etas)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    timing: Dict[str, float] = {}

    log(f"loading {args.model} run_id={args.run_id} dtype={args.dtype}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched_w, data_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.fit_token_offset,
        args.val_token_offset,
    )
    untouched_c4 = build_c4_untouched_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )
    timing["load_tokenizer_and_data_sec"] = time.time() - t0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    timing["load_model_sec"] = time.time() - t0

    t0 = time.time()
    fp_qk = snapshot_qk(model, layers)
    fp_nll = eval_current(model, val, untouched_w, untouched_c4, device)
    timing["snapshot_fp_eval_sec"] = time.time() - t0

    t0 = time.time()
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
    direct_qk = snapshot_qk(model, layers)
    direct_states = direct_states_from_fp(fp_qk, layers, args.group_size, args.threshold_factor)
    direct_nll = eval_current(model, val, untouched_w, untouched_c4, device)
    timing["direct_ptq_eval_sec"] = time.time() - t0

    t0 = time.time()
    grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    cegsp_patches, traces = build_cegsp_layer_patches(
        fp_qk, grads, layers, args.group_size, args.threshold_factor, args.max_edits
    )
    per_layer = []
    for layer in layers:
        apply_qk_patch(model, layers, direct_qk, cegsp_patches, [layer])
        val_nll = evaluate_nll(model, val, device)
        per_layer.append(
            {
                "layer": int(layer),
                "single_layer_delta_val_nll": float(val_nll - direct_nll["val"]),
                **traces[layer],
            }
        )
    ranked = sorted(per_layer, key=lambda r: float(r["single_layer_delta_val_nll"]))
    selected_layers = [int(r["layer"]) for r in ranked[: args.layer_topk]]
    apply_qk_patch(model, layers, direct_qk, cegsp_patches, selected_layers)
    cegsp_nll = eval_current(model, val, untouched_w, untouched_c4, device)
    changed = 0
    for layer in selected_layers:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            changed += int(projection_weight(refs[key]).detach().float().cpu().ne(direct_qk[layer][key].float()).sum().item())
    restore_qk(model, layers, direct_qk)
    timing["cegsp_fixed_sec"] = time.time() - t0

    t0 = time.time()
    all_qk = run_one_step_scope(
        model, fit, val, untouched_w, untouched_c4, device, fp_qk, direct_qk, direct_states,
        layers, layers, args.group_size, args.threshold_factor, etas, changed, "all_qk_layers"
    )
    timing["one_step_all_qk_grid_sec"] = time.time() - t0

    t0 = time.time()
    selected_qk = run_one_step_scope(
        model, fit, val, untouched_w, untouched_c4, device, fp_qk, direct_qk, direct_states,
        layers, selected_layers, args.group_size, args.threshold_factor, etas, changed, "cegsp_selected_layers_only"
    )
    timing["one_step_selected_qk_grid_sec"] = time.time() - t0

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "status": "complete",
        "config": vars(args),
        "validation_version": {
            "name": "CEGSP-v2-P4R-fairness-strict-edit-match-and-update-scope",
            "primary_question": "Does stricter edit matching or QAT update scope change the P4-R conclusion?",
        },
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched_w),
            "untouched_c4_batches": len(untouched_c4),
            "fit_token_offset": args.fit_token_offset,
            "val_token_offset": args.val_token_offset,
            "c4_token_offset": args.c4_token_offset,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "quant_counts": quant_counts,
        "nll": {
            "fp": fp_nll,
            "fp_with_ppl": metric_pack(fp_nll),
            "direct_ternary": direct_nll,
            "direct_ternary_with_ppl": metric_pack(direct_nll),
            "cegsp": {
                "selected_layers": selected_layers,
                "changed_coordinates": int(changed),
                "nll": cegsp_nll,
                "with_ppl": metric_pack(cegsp_nll),
                "delta_vs_direct": {k: float(cegsp_nll[k] - direct_nll[k]) for k in cegsp_nll},
            },
            "one_step_qat_scopes": {
                "all_qk_layers": all_qk,
                "cegsp_selected_layers_only": selected_qk,
            },
        },
        "timing": timing,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
