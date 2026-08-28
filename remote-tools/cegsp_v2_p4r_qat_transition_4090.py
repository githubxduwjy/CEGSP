#!/usr/bin/env python3
"""CEGSP v2 P4-R: QAT transition audit and edit-matched one-step baseline.

This script strengthens the QAT comparison after P4.  It does not introduce a
new CEGSP module.  It keeps canonical CEGSP fixed, then audits whether one-step
latent QAT actually crosses ternary decision boundaries, and compares CEGSP to
an edit-matched one-step QAT point.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_ce_gradient_4090 import collect_ce_qk_grads, target_modules, projection_weight, set_projection_weight
from cegsp_v2_p4_gap_cost_4090 import (
    apply_qk_patch,
    build_c4_untouched_batches,
    build_cegsp_layer_patches,
    normalized_latent_step,
    safe_ppl,
    snapshot_qk,
    restore_qk,
)
from tqgsp_support_projection_4090 import (
    apply_direct_ptq_local,
    build_wikitext_splits,
    evaluate_nll,
    log,
    make_code,
    parse_csv_ints,
    direct_ternary_weight,
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
    p.add_argument("--one-step-etas", default="1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
    p.add_argument("--multi-step-etas", default="0.001,0.003,0.01")
    p.add_argument("--multi-steps", default="5,10,20,50")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def metric_pack(nll: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {k: {"nll": float(v), "ppl": safe_ppl(float(v))} for k, v in nll.items()}


def quantize_latent_qk(
    latent_qk: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    weights: Dict[int, Dict[str, torch.Tensor]] = {}
    states: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        weights[layer] = {}
        states[layer] = {}
        for key in ("q", "k"):
            code = make_code(latent_qk[layer][key], group_size, threshold_factor)
            q, _ = direct_ternary_weight(latent_qk[layer][key], group_size, threshold_factor)
            weights[layer][key] = q.float().cpu()
            states[layer][key] = code.state.clone().cpu()
    return weights, states


def direct_states_from_fp(
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
    group_size: int,
    threshold_factor: float,
) -> Dict[int, Dict[str, torch.Tensor]]:
    states: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        states[layer] = {}
        for key in ("q", "k"):
            states[layer][key] = make_code(fp_qk[layer][key], group_size, threshold_factor).state.clone().cpu()
    return states


def transition_counts(
    before: Dict[int, Dict[str, torch.Tensor]],
    after: Dict[int, Dict[str, torch.Tensor]],
    layers: List[int],
) -> Dict[str, int]:
    counts = {
        "changed_total": 0,
        "zero_to_pos": 0,
        "zero_to_neg": 0,
        "pos_to_zero": 0,
        "neg_to_zero": 0,
        "pos_to_neg": 0,
        "neg_to_pos": 0,
        "other": 0,
    }
    for layer in layers:
        for key in ("q", "k"):
            b = before[layer][key].reshape(-1)
            a = after[layer][key].reshape(-1)
            valid = b.ne(a)
            counts["changed_total"] += int(valid.sum().item())
            pairs = [
                ("zero_to_pos", b.eq(0) & a.gt(0)),
                ("zero_to_neg", b.eq(0) & a.lt(0)),
                ("pos_to_zero", b.gt(0) & a.eq(0)),
                ("neg_to_zero", b.lt(0) & a.eq(0)),
                ("pos_to_neg", b.gt(0) & a.lt(0)),
                ("neg_to_pos", b.lt(0) & a.gt(0)),
            ]
            accounted = torch.zeros_like(valid)
            for name, mask in pairs:
                mask = mask & valid
                counts[name] += int(mask.sum().item())
                accounted |= mask
            counts["other"] += int((valid & ~accounted).sum().item())
    return counts


def eval_current(model, val, untouched_w, untouched_c4, device) -> Dict[str, float]:
    out = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched_w, device),
    }
    if untouched_c4:
        out["untouched_c4"] = evaluate_nll(model, untouched_c4, device)
    return out


def run_one_step_grid(
    model,
    fit,
    val,
    untouched_w,
    untouched_c4,
    device,
    fp_qk,
    direct_qk,
    direct_states,
    layers,
    group_size,
    threshold_factor,
    etas,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    # Use the same quantized-point gradient source for all one-step eta values.
    restore_qk(model, layers, direct_qk)
    grads = collect_ce_qk_grads(model, fit, layers, device, grad_batches=1)
    for eta in etas:
        latent = {layer: {key: fp_qk[layer][key].clone() for key in ("q", "k")} for layer in layers}
        latent = normalized_latent_step(latent, fp_qk, grads, layers, float(eta))
        quant, states = quantize_latent_qk(latent, layers, group_size, threshold_factor)
        apply_qk_patch(model, layers, direct_qk, quant, layers)
        nll = eval_current(model, val, untouched_w, untouched_c4, device)
        rows.append(
            {
                "eta": float(eta),
                "steps": 1,
                "nll": nll,
                "with_ppl": metric_pack(nll),
                "transition_counts": transition_counts(direct_states, states, layers),
            }
        )
    restore_qk(model, layers, direct_qk)
    return rows


def run_multi_step_curve(
    model,
    fit,
    val,
    untouched_w,
    untouched_c4,
    device,
    fp_qk,
    direct_qk,
    direct_states,
    layers,
    group_size,
    threshold_factor,
    etas,
    step_counts,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for steps in step_counts:
        for eta in etas:
            latent = {layer: {key: fp_qk[layer][key].clone() for key in ("q", "k")} for layer in layers}
            for _ in range(int(steps)):
                quant, _states = quantize_latent_qk(latent, layers, group_size, threshold_factor)
                apply_qk_patch(model, layers, direct_qk, quant, layers)
                grads = collect_ce_qk_grads(model, fit, layers, device, grad_batches=1)
                latent = normalized_latent_step(latent, fp_qk, grads, layers, float(eta))
            quant, states = quantize_latent_qk(latent, layers, group_size, threshold_factor)
            apply_qk_patch(model, layers, direct_qk, quant, layers)
            nll = eval_current(model, val, untouched_w, untouched_c4, device)
            rows.append(
                {
                    "eta": float(eta),
                    "steps": int(steps),
                    "nll": nll,
                    "with_ppl": metric_pack(nll),
                    "transition_counts": transition_counts(direct_states, states, layers),
                }
            )
    restore_qk(model, layers, direct_qk)
    return rows


def select_best_by_val(rows: List[Dict[str, object]]) -> Dict[str, object]:
    return min(rows, key=lambda r: float(r["nll"]["val"]))


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
    one_step_etas = parse_float_list(args.one_step_etas)
    multi_step_etas = parse_float_list(args.multi_step_etas)
    multi_steps = parse_csv_ints(args.multi_steps)
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
    timing["snapshot_fp_and_eval_sec"] = time.time() - t0

    t0 = time.time()
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
    direct_qk = snapshot_qk(model, layers)
    direct_states = direct_states_from_fp(fp_qk, layers, args.group_size, args.threshold_factor)
    direct_nll = eval_current(model, val, untouched_w, untouched_c4, device)
    timing["direct_ptq_and_eval_sec"] = time.time() - t0

    t0 = time.time()
    ce_grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    timing["ce_gradient_collection_sec"] = time.time() - t0

    t0 = time.time()
    cegsp_patches, cegsp_traces = build_cegsp_layer_patches(
        fp_qk,
        ce_grads,
        layers,
        args.group_size,
        args.threshold_factor,
        args.max_edits,
    )
    per_layer = []
    for layer in layers:
        apply_qk_patch(model, layers, direct_qk, cegsp_patches, [layer])
        val_nll = evaluate_nll(model, val, device)
        per_layer.append(
            {
                "layer": int(layer),
                "single_layer_val_nll": float(val_nll),
                "single_layer_delta_val_nll": float(val_nll - direct_nll["val"]),
                **cegsp_traces[layer],
            }
        )
    ranked = sorted(per_layer, key=lambda r: float(r["single_layer_delta_val_nll"]))
    selected_layers = [int(r["layer"]) for r in ranked[: max(0, min(args.layer_topk, len(ranked)))]]
    apply_qk_patch(model, layers, direct_qk, cegsp_patches, selected_layers)
    cegsp_nll = eval_current(model, val, untouched_w, untouched_c4, device)
    cegsp_changed_coordinates = 0
    # CEGSP is support relocation, so count actual weight-value changes in selected Q/K matrices.
    for layer in selected_layers:
        refs = target_modules(model, layer)
        for key in ("q", "k"):
            patched = projection_weight(refs[key]).detach().float().cpu()
            base = direct_qk[layer][key].detach().float().cpu()
            cegsp_changed_coordinates += int(patched.ne(base).sum().item())
    restore_qk(model, layers, direct_qk)
    timing["cegsp_edit_select_eval_sec"] = time.time() - t0

    t0 = time.time()
    one_step_rows = run_one_step_grid(
        model,
        fit,
        val,
        untouched_w,
        untouched_c4,
        device,
        fp_qk,
        direct_qk,
        direct_states,
        layers,
        args.group_size,
        args.threshold_factor,
        one_step_etas,
    )
    timing["one_step_qat_transition_grid_sec"] = time.time() - t0
    one_step_val_best = select_best_by_val(one_step_rows)
    one_step_edit_matched = min(
        one_step_rows,
        key=lambda r: abs(int(r["transition_counts"]["changed_total"]) - int(cegsp_changed_coordinates)),
    )

    t0 = time.time()
    multi_rows = run_multi_step_curve(
        model,
        fit,
        val,
        untouched_w,
        untouched_c4,
        device,
        fp_qk,
        direct_qk,
        direct_states,
        layers,
        args.group_size,
        args.threshold_factor,
        multi_step_etas,
        multi_steps,
    )
    timing["multi_step_qat_curve_sec"] = time.time() - t0
    best_multi_by_step = {
        str(step): select_best_by_val([r for r in multi_rows if int(r["steps"]) == int(step)])
        for step in multi_steps
    }
    best_multi = select_best_by_val(multi_rows)

    denom_w = float(direct_nll["untouched_w"] - best_multi["nll"]["untouched_w"])
    gap_w = (
        float((direct_nll["untouched_w"] - cegsp_nll["untouched_w"]) / denom_w)
        if abs(denom_w) > 1e-12
        else float("nan")
    )
    if "untouched_c4" in direct_nll and "untouched_c4" in best_multi["nll"]:
        denom_c4 = float(direct_nll["untouched_c4"] - best_multi["nll"]["untouched_c4"])
        gap_c4 = (
            float((direct_nll["untouched_c4"] - cegsp_nll["untouched_c4"]) / denom_c4)
            if abs(denom_c4) > 1e-12
            else float("nan")
        )
    else:
        gap_c4 = float("nan")

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "status": "complete",
        "config": vars(args),
        "validation_version": {
            "name": "CEGSP-v2-P4R-QAT-transition-and-edit-matched-baseline",
            "primary_question": "Does a properly swept one-step latent QAT cross ternary boundaries, and how does edit-matched one-step QAT compare with canonical CEGSP?",
            "not_a_new_cegsp_module": True,
        },
        "clean_room_invariants": {
            "cegsp_uses_qat_teacher": False,
            "cegsp_uses_optimizer_steps": False,
            "cegsp_uses_latent_fp_update": False,
            "qat_controls_use_latent_fp_update": True,
            "qat_controls_are_baselines_only": True,
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
                "changed_coordinates": int(cegsp_changed_coordinates),
                "nll": cegsp_nll,
                "with_ppl": metric_pack(cegsp_nll),
                "delta_vs_direct": {k: float(cegsp_nll[k] - direct_nll[k]) for k in cegsp_nll},
            },
            "one_step_qat": {
                "rows": one_step_rows,
                "validation_best": one_step_val_best,
                "edit_matched_to_cegsp": one_step_edit_matched,
            },
            "multi_step_qat": {
                "rows": multi_rows,
                "best_by_step_validation": best_multi_by_step,
                "best_overall_validation": best_multi,
            },
        },
        "gap_closure_vs_best_multistep": {
            "untouched_w": gap_w,
            "untouched_c4": gap_c4,
        },
        "timing": timing,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
