#!/usr/bin/env python3
"""P5-C: official PT² state parity followed by affine-index CEGSP.

The official PT² quantizer returns a placeholder T tensor.  This harness
captures the real ternary state inside PT²'s quantizer call, verifies that the
returned block is exactly affine ternary, and only then runs CEGSP on the
official PT² model.  No state is fitted from the final weight alone.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_11a_audit_adapter import (
    make_pt2_args,
    make_pt2_calib_loader,
    patch_opt_position_embeddings_compat,
)
from cegsp_ce_gradient_4090 import collect_ce_qk_grads
from cegsp_p5a_affine_adapter_feasibility_4090 import (
    AffineCode,
    AffineEdit,
    apply_affine_patch,
    apply_edits,
    audit_all,
    build_group_candidates,
    cardinality_violations,
    eval_metrics,
    restore_qk,
    snapshot_qk,
    with_ppl,
)
from cegsp_p5b_overall_affine_4090 import (
    changed_coordinates,
    metric_delta,
    per_layer_edit_counts,
    random_layerwise_edits,
    select_layerwise_edits,
)
from cegsp_v2_p4_gap_cost_4090 import build_c4_untouched_batches
from tqgsp_support_projection_4090 import build_wikitext_splits, log, parse_csv_ints


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=8)
    p.add_argument("--c4-untouched-batches", type=int, default=8)
    p.add_argument("--fit-token-offset", type=int, default=0)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--pt2-calib-samples", type=int, default=16)
    p.add_argument("--pt2-calib-seq-len", type=int, default=128)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--layer-probe-edits", type=int, default=8)
    p.add_argument("--edits-per-layer", type=int, default=64)
    p.add_argument("--primary-layer-budget", type=int, default=6)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def finite_metrics(metrics: Dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in metrics.values())


def fingerprint_batches(batches: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch[:, :-1].contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def module_by_name(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    current = layer
    for part in name.split("."):
        current = getattr(current, part)
    return current


def opt_module_specs(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    names = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.out_proj",
        "fc1",
        "fc2",
    ]
    specs: List[Tuple[int, str, torch.nn.Module]] = []
    layers = model.model.decoder.layers
    for layer_idx, layer in enumerate(layers):
        for name in names:
            specs.append((layer_idx, name, module_by_name(layer, name)))
    return specs


def affine_from_q_and_t(q: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    q = q.detach().float()
    t = t.detach().float()
    t_mean = t.mean(dim=1, keepdim=True)
    q_mean = q.mean(dim=1, keepdim=True)
    tc = t - t_mean
    qc = q - q_mean
    denom = (tc * tc).sum(dim=1, keepdim=True)
    alpha = torch.where(denom > 1e-12, (tc * qc).sum(dim=1, keepdim=True) / denom, torch.zeros_like(denom))
    mu = q_mean - alpha * t_mean
    residual = float((q - (mu + alpha * t)).abs().max().item())
    return mu, alpha, residual


class PT2StateCapture:
    def __init__(self) -> None:
        self.active = False
        self.initial_t: torch.Tensor | None = None
        self.final_t: torch.Tensor | None = None
        self.records: List[Dict[str, object]] = []
        self.post_fasterquant: List[Dict[str, object]] = []

    def reset(self) -> None:
        self.initial_t = None
        self.final_t = None


def install_state_capture(qmod: object, gptqmod: object, capture: PT2StateCapture):
    original_init = qmod.ternary_init
    original_update = qmod.update_ternary
    original_quantize = qmod.TernaryQuantizer.quantize
    original_fasterquant = gptqmod.GPTQ.fasterquant

    def ternary_init_capture(x, *args, **kwargs):
        result = original_init(x, *args, **kwargs)
        if capture.active:
            capture.initial_t = result[2].detach().float().cpu().clone()
        return result

    def update_capture(x, alpha, mean):
        result = original_update(x, alpha, mean)
        if capture.active:
            capture.final_t = result.detach().float().cpu().clone()
        return result

    def quantize_capture(self, w, *args, **kwargs):
        capture.reset()
        capture.active = True
        try:
            q, placeholder_t = original_quantize(self, w, *args, **kwargs)
        finally:
            capture.active = False
        t = capture.final_t if capture.final_t is not None else capture.initial_t
        if t is None:
            raise RuntimeError("PT2 quantizer call did not expose ternary state")
        q_cpu = q.detach().float().cpu().clone()
        t_cpu = t.detach().float().cpu().clone()
        if q_cpu.shape != t_cpu.shape:
            raise RuntimeError(f"state shape mismatch q={tuple(q_cpu.shape)} T={tuple(t_cpu.shape)}")
        mu, alpha, residual = affine_from_q_and_t(q_cpu, t_cpu)
        capture.records.append(
            {
                "q": q_cpu,
                "T": t_cpu,
                "mu": mu,
                "alpha": alpha,
                "residual": residual,
                "placeholder_T_nonzero": int(placeholder_t.detach().abs().sum().item()),
            }
        )
        return q, placeholder_t

    def fasterquant_capture(self, *args, **kwargs):
        start = len(capture.records)
        result = original_fasterquant(self, *args, **kwargs)
        end = len(capture.records)
        if end > start:
            q = torch.cat([item["q"] for item in capture.records[start:end]], dim=1)
            w = self.layer.weight.detach().float().cpu()
            capture.post_fasterquant.append(
                {
                    "module": str(getattr(self.layer, "global_name", "")),
                    "shape": [int(x) for x in w.shape],
                    "num_blocks": end - start,
                    "max_post_vs_capture_q": float((w - q).abs().max().item()),
                    "capture_start": start,
                    "capture_end": end,
                }
            )
        return result

    qmod.ternary_init = ternary_init_capture
    qmod.update_ternary = update_capture
    qmod.TernaryQuantizer.quantize = quantize_capture
    gptqmod.GPTQ.fasterquant = fasterquant_capture

    def restore() -> None:
        qmod.ternary_init = original_init
        qmod.update_ternary = original_update
        qmod.TernaryQuantizer.quantize = original_quantize
        gptqmod.GPTQ.fasterquant = original_fasterquant

    return restore


def make_pt2_codes(
    model: torch.nn.Module,
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    records: Sequence[Dict[str, object]],
    group_size: int,
) -> Tuple[Dict[int, Dict[str, AffineCode]], Dict[str, object]]:
    specs = opt_module_specs(model)
    cursor = 0
    codes: Dict[int, Dict[str, AffineCode]] = {}
    parity_rows: List[Dict[str, object]] = []
    for layer, name, module in specs:
        rows, cols = module.weight.shape
        n_blocks = (int(cols) + group_size - 1) // group_size
        blocks = [records[cursor + i] for i in range(n_blocks)]
        cursor += n_blocks
        if name not in {"self_attn.q_proj", "self_attn.k_proj"}:
            continue
        key = "q" if name.endswith("q_proj") else "k"
        q_blocks = [item["q"] for item in blocks]
        t_blocks = [item["T"] for item in blocks]
        mu_blocks = [item["mu"] for item in blocks]
        alpha_blocks = [item["alpha"] for item in blocks]
        q_cat = torch.cat(q_blocks, dim=1)
        t_float = torch.cat(t_blocks, dim=1)
        t_round = t_float.round().to(torch.int8)
        mu = torch.stack([x.squeeze(1) for x in mu_blocks], dim=1).unsqueeze(-1)
        alpha = torch.stack([x.squeeze(1) for x in alpha_blocks], dim=1).unsqueeze(-1)
        padded_cols = n_blocks * group_size
        fp_pad = torch.zeros((rows, padded_cols), dtype=torch.float32)
        fp_pad[:, :cols] = fp_qk[layer][key]
        valid = torch.zeros((rows, padded_cols), dtype=torch.bool)
        valid[:, :cols] = True
        code = AffineCode(
            mu=mu,
            alpha=alpha,
            T=t_round.view(rows, n_blocks, group_size),
            valid=valid.view(rows, n_blocks, group_size),
            original_shape=(int(rows), int(cols)),
            group_size=group_size,
            fp_padded=fp_pad.view(rows, n_blocks, group_size),
        )
        codes.setdefault(layer, {})[key] = code
        final_w = module.weight.detach().float().cpu()
        q_final = q_cat[:, :cols]
        q_deployed = q_final.to(module.weight.dtype).float()
        _, _, deployed_codebook_residual = affine_from_q_and_t(q_deployed, t_float[:, :cols])
        parity_rows.append(
            {
                "layer": layer,
                "module": name,
                "shape": [int(rows), int(cols)],
                "blocks": n_blocks,
                "group_size": group_size,
                "capture_codebook_residual": max(float(item["residual"]) for item in blocks),
                "deployed_codebook_residual": float(deployed_codebook_residual),
                "capture_T_illegal": int(((t_float != -1) & (t_float != 0) & (t_float != 1)).sum().item()),
                "capture_T_nonfinite": int((~torch.isfinite(t_float)).sum().item()),
                "final_vs_capture_q_max_abs": float((final_w - q_final).abs().max().item()),
                "final_vs_deployed_capture_q_max_abs": float((final_w - q_deployed).abs().max().item()),
                "capture_q_abs_max": float(q_final.abs().max().item()),
                "deployed_capture_q_abs_max": float(q_deployed.abs().max().item()),
                "final_dtype": str(module.weight.dtype),
                "placeholder_T_nonzero_total": int(sum(int(item["placeholder_T_nonzero"]) for item in blocks)),
            }
        )
    if cursor != len(records):
        raise RuntimeError(f"PT2 state record mapping mismatch consumed={cursor} records={len(records)}")
    return codes, {"rows": parity_rows}


def parity_summary(
    codes: Dict[int, Dict[str, AffineCode]],
    parity_detail: Dict[str, object],
    group_size: int,
) -> Dict[str, object]:
    rows = parity_detail["rows"]
    max_capture_residual = max((float(row["capture_codebook_residual"]) for row in rows), default=float("inf"))
    max_deployed_codebook_residual = max((float(row["deployed_codebook_residual"]) for row in rows), default=float("inf"))
    max_final_residual = max((float(row["final_vs_capture_q_max_abs"]) for row in rows), default=float("inf"))
    max_final_deployed_residual = max((float(row["final_vs_deployed_capture_q_max_abs"]) for row in rows), default=float("inf"))
    illegal_t = sum(int(row["capture_T_illegal"]) for row in rows)
    nonfinite_t = sum(int(row["capture_T_nonfinite"]) for row in rows)
    code_audit = audit_all(codes)
    pass_state = (
        len(rows) == 48
        and group_size == 128
        and illegal_t == 0
        and nonfinite_t == 0
        # The captured PT2 state is FP32 affine (q = mu + alpha*T) before the
        # official model-weight cast.  The deployed FP16 tensor need not be
        # exactly affine in FP32 because its three levels are rounded
        # independently.  Therefore the gate is applied to the captured
        # state, while deployment parity is checked against cast(q).
        and max_capture_residual < 1e-3
        and max_final_deployed_residual < 1e-3
        and code_audit["total_illegal_states"] == 0
    )
    return {
        "pass": pass_state,
        "qk_module_count": len(rows),
        "expected_qk_module_count": 48,
        "group_size": group_size,
        "scale_granularity": "per-row per-group",
        "permutation": "disabled (ssr=False)",
        "illegal_T_count": illegal_t,
        "nonfinite_T_count": nonfinite_t,
        "max_capture_codebook_residual": max_capture_residual,
        "max_deployed_codebook_residual": max_deployed_codebook_residual,
        "max_final_vs_capture_q_residual": max_final_residual,
        "max_final_vs_deployed_capture_q_residual": max_final_deployed_residual,
        "code_audit": code_audit,
        "module_rows": rows,
    }


def main() -> None:
    args = parse_args()
    if args.model != "facebook/opt-350m":
        raise ValueError("P5-C pre-registration is fixed to facebook/opt-350m")
    if args.group_size != 128 or args.pt2_calib_samples != 16 or args.pt2_calib_seq_len != 128:
        raise ValueError("P5-C pre-registration requires group=128, calibration=16x128")
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("P5-C requires CUDA")
    device = torch.device("cuda")
    layers = list(range(24))
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, w2, wikitext_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.fit_token_offset,
        args.val_token_offset,
    )
    c4 = build_c4_untouched_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )
    fit_fingerprint = fingerprint_batches(fit)
    pt2_loader = make_pt2_calib_loader(fit)
    if len(pt2_loader) != args.pt2_calib_samples:
        raise RuntimeError(f"calibration sample mismatch {len(pt2_loader)} != {args.pt2_calib_samples}")

    sys.path.insert(0, str(Path(args.pt2_root)))
    pt2_quantize = importlib.import_module("quantize")
    qmod = importlib.import_module("pt2_llm.quantizer")
    gptqmod = importlib.import_module("pt2_llm.gptq")
    pt2_quantize.args = make_pt2_args(
        args,
        "atq",
        False,
        "cuda:0",
    )
    pt2_quantize.args.nsamples = args.pt2_calib_samples
    pt2_quantize.args.calib_seqlen = args.pt2_calib_seq_len
    pt2_quantize.args.blocksize = args.group_size
    pt2_quantize.args.percdamp = args.percdamp
    pt2_quantize.groupsize = args.group_size

    log(f"P5-C loading official PT2 model={args.model} method=atq ssr=False")
    model = pt2_quantize.get_model(args.model, args.pt2_calib_seq_len)
    model.seqlen = args.pt2_calib_seq_len
    model.eval()
    compat_patched = patch_opt_position_embeddings_compat(model)
    fp_qk = snapshot_qk(model, layers)
    capture = PT2StateCapture()
    restore_capture = install_state_capture(qmod, gptqmod, capture)
    try:
        with torch.no_grad():
            pt2_quantize.quant_sequential(model, pt2_loader, "cuda:0")
    finally:
        restore_capture()
    model.to(device)
    model.config.use_cache = False
    pt2_qk = snapshot_qk(model, layers)

    expected_blocks = sum(
        (int(module.weight.shape[1]) + args.group_size - 1) // args.group_size
        for _, _, module in opt_module_specs(model)
    )
    if expected_blocks != len(capture.records):
        raise RuntimeError(f"PT2 block capture mismatch expected={expected_blocks} got={len(capture.records)}")
    codes, parity_detail = make_pt2_codes(model, fp_qk, capture.records, args.group_size)
    parity = parity_summary(codes, parity_detail, args.group_size)

    result: Dict[str, object] = {
        "run_id": args.run_id,
        "experiment": "CEGSP-P5-C official PT2 -> affine CEGSP compatibility",
        "status": "parity_passed" if parity["pass"] else "parity_failed",
        "config": vars(args),
        "protocol": {
            "pt2_method": "atq",
            "ssr": False,
            "pt2_full_model_quantization": True,
            "cegsp_targets": "all 24 layers Q/K only",
            "selection_uses_validation": False,
            "selection_uses_untouched": False,
            "mu_alpha_refit": False,
            "teacher_or_qat": False,
            "calibration_fingerprint": fit_fingerprint,
            "evaluator": "same compact CEGSP evaluator for val/W2/C4",
        },
        "data": {
            "wikitext_source": wikitext_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "wikitext2_untouched_batches": len(w2),
            "c4_untouched_batches": len(c4),
            "pt2_calib_samples": len(pt2_loader),
            "pt2_calib_seq_len": args.pt2_calib_seq_len,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        "compat": {
            "opt_position_embeddings_kwarg_dropped": compat_patched,
            "captured_quantizer_calls": len(capture.records),
            "expected_quantizer_calls": expected_blocks,
            "pt2_placeholder_T_is_nonzero": any(
                int(item["placeholder_T_nonzero"]) > 0 for item in capture.records
            ),
        },
        "state_parity": parity,
        "post_fasterquant_diagnostic": {
            "count": len(capture.post_fasterquant),
            "max_post_vs_capture_q": max(
                (float(x["max_post_vs_capture_q"]) for x in capture.post_fasterquant),
                default=float("nan"),
            ),
            "first_rows": capture.post_fasterquant[:12],
        },
        "elapsed_until_parity_sec": time.time() - started,
    }

    if not parity["pass"]:
        result["performance_gate"] = "NOT_RUN_STATE_PARITY_FAILED"
        result["elapsed_sec"] = time.time() - started
        path = out_dir / "p5c_pt2_affine_result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"state parity failed; wrote {path}")
        return

    pt2_metrics = eval_metrics(model, device, val, w2, c4)
    grads = collect_ce_qk_grads(model, fit, layers, device, 1)
    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_rows: List[Dict[str, object]] = []
    for layer in layers:
        candidates: List[AffineEdit] = []
        for key in ("q", "k"):
            candidates.extend(
                build_group_candidates(
                    layer, key, codes[layer][key], grads[layer][key], "affine_fp"
                )
            )
        candidates.sort(key=lambda e: (-e.score_formula, e.key, e.row, e.block))
        candidates_by_layer[layer] = candidates
        probe = candidates[: args.layer_probe_edits]
        layer_rows.append(
            {
                "layer": layer,
                "num_candidates": len(candidates),
                "probe_score_sum": float(sum(e.score_formula for e in probe)),
                "top_scores": [float(e.score_formula) for e in probe[:5]],
            }
        )
    layer_rows.sort(key=lambda row: (-float(row["probe_score_sum"]), int(row["layer"])))
    selected_layers = [int(row["layer"]) for row in layer_rows[: args.primary_layer_budget]]
    ce_edits = select_layerwise_edits(candidates_by_layer, selected_layers, args.edits_per_layer)
    ce_states = apply_edits(codes, ce_edits)
    apply_affine_patch(model, codes, ce_states)
    ce_metrics = eval_metrics(model, device, val, w2, c4)
    ce_audit = audit_all(codes, ce_states)
    ce_counts = per_layer_edit_counts(ce_edits)

    restore_qk(model, pt2_qk)
    random_edits = random_layerwise_edits(codes, selected_layers, ce_counts, args.seed + 6)
    random_states = apply_edits(codes, random_edits)
    apply_affine_patch(model, codes, random_states)
    random_metrics = eval_metrics(model, device, val, w2, c4)
    random_audit = audit_all(codes, random_states)
    restore_qk(model, pt2_qk)

    ce_name = "pt2_plus_affine_cegsp_top6"
    random_name = "pt2_plus_matched_random_top6"
    variants = {
        "pt2": {
            "metrics": with_ppl(pt2_metrics),
            "num_edits": 0,
            "changed_coordinates": 0,
        },
        ce_name: {
            "selected_layers": selected_layers,
            "metrics": with_ppl(ce_metrics),
            "delta_vs_pt2_nll": metric_delta(ce_metrics, pt2_metrics),
            "num_edits": len(ce_edits),
            "changed_coordinates": changed_coordinates(codes, ce_states),
            "edits_per_layer": ce_counts,
            "audit": ce_audit,
            "cardinality_violations": cardinality_violations(codes, ce_states),
        },
        random_name: {
            "selected_layers": selected_layers,
            "metrics": with_ppl(random_metrics),
            "delta_vs_pt2_nll": metric_delta(random_metrics, pt2_metrics),
            "num_edits": len(random_edits),
            "changed_coordinates": changed_coordinates(codes, random_states),
            "edits_per_layer": per_layer_edit_counts(random_edits),
            "audit": random_audit,
            "cardinality_violations": cardinality_violations(codes, random_states),
        },
    }
    def nll(name: str, split: str) -> float:
        return float(variants[name]["metrics"][split]["nll"])

    legality_pass = all(
        v["audit"]["total_illegal_states"] == 0 and v["cardinality_violations"] == 0
        for v in variants.values()
        if "audit" in v
    )
    finite_pass = finite_metrics(pt2_metrics) and finite_metrics(ce_metrics) and finite_metrics(random_metrics)
    ce_vs_pt2_val = nll(ce_name, "val") < nll("pt2", "val")
    ce_vs_pt2_w2 = nll(ce_name, "wikitext2_untouched") < nll("pt2", "wikitext2_untouched")
    ce_vs_random_w2 = nll(ce_name, "wikitext2_untouched") < nll(random_name, "wikitext2_untouched")
    performance_pass = parity["pass"] and legality_pass and finite_pass and ce_vs_pt2_val and ce_vs_pt2_w2 and ce_vs_random_w2
    result["layer_ranking"] = layer_rows
    result["selected_layers"] = selected_layers
    result["variants"] = variants
    result["diagnostic_reference"] = {
        "source": "CEGSP-P5-B local result; not an official PT2 system",
        "run_id": "cegsp_p5b_overall_affine_opt350m_20260828",
        "metrics": {
            "val": {"nll": 4.178844571113586, "ppl": 65.290371},
            "wikitext2_untouched": {"nll": 4.309960246086121, "ppl": 74.437530},
            "c4_untouched": {"nll": 3.6735510528087616, "ppl": 39.391539},
        },
    }
    result["performance_gate"] = {
        "legality_pass": legality_pass,
        "finite_pass": finite_pass,
        "ce_improves_pt2_val": ce_vs_pt2_val,
        "ce_improves_pt2_w2": ce_vs_pt2_w2,
        "ce_beats_matched_random_w2": ce_vs_random_w2,
        "strong_compatibility_pass": performance_pass,
    }
    restore_qk(model, fp_qk)
    result["elapsed_sec"] = time.time() - started
    path = out_dir / "p5c_pt2_affine_result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {path}")
    log(json.dumps(result["performance_gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
