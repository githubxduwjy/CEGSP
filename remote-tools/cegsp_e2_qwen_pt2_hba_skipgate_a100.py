#!/usr/bin/env python3
"""E2-B: Qwen3-8B PT2 + HBA with strict parity gate explicitly skipped.

This runner is intentionally a conditional experiment.  E2-A0 showed that the
official PT2 Qwen path completes and exports a legal ternary state, but strict
absolute deployed-vs-sidecar parity is still pending BF16-aware forensics.  The
user explicitly requested continuing beyond that gate; therefore this script
records the skipped gate in the artifact and does not promote the result to a
strict state-export PASS.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import torch
from transformers import set_seed

from cegsp_e2_qwen_pt2_health_a100 import apply_qwen_pt2_layer_adapter
from cegsp_p7_a100_scaling import (
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    get_decoder_layers,
    metric_delta,
    target_qk,
)
from cegsp_p9s2_detached_pt2_plugin import (
    Capture,
    apply_ssr_codes,
    affine_from_q_and_t,
    cardinality_violations,
    finite_metrics,
    install_capture,
    load_detached_artifacts,
    module_specs,
    official_metrics,
    restore_qk,
)
from cegsp_pt2_hba_detached_a100 import (
    batches_hash,
    candidate_manifest,
    row_for_patch,
    select_prefix,
    state_hash,
)


GRID = (1, 2, 4, 8, 16, 32, 64)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B")
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--calib-nsamples", type=int, default=2)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--val-start", type=int, default=1)
    p.add_argument("--val-samples", type=int, default=1)
    p.add_argument("--grad-samples", type=int, default=1)
    p.add_argument("--candidate-top-k", type=int, default=256)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--num-p", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--pt2-checkpoint", default="", help="compact full-module PT2 state to resume without quantization")
    p.add_argument("--skip-strict-parity-gate", action="store_true")
    return p.parse_args()


def make_pt2_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.calib_nsamples,
        percdamp=args.percdamp,
        blocksize=args.group_size,
        num_p=args.num_p,
        salient_metric="hessian",
        device="cuda:0",
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=args.calib_seq_len,
        ppl_seqlen=args.calib_seq_len,
        quant_only="",
        invert=False,
        ssr=True,
        log_wandb=False,
        tasks="",
        experiment=args.run_id,
        num_fewshot=0,
        limit=-1,
    )


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def restore_model_device(model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
    """Restore the full model after PT2's layerwise evaluator offloads it."""
    model.to(device)
    for name in ("model", "lm_head"):
        module = getattr(model, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
    # PT2/accelerate may leave custom quantized parameters outside the normal
    # recursive .to() path.  Move their storage explicitly before auditing.
    for parameter in model.parameters():
        if parameter.device != device:
            parameter.data = parameter.data.to(device)
    for module in model.modules():
        for name, buffer in list(module._buffers.items()):
            if buffer is not None and buffer.device != device:
                module._buffers[name] = buffer.to(device)
    bad_parameters = [name for name, p in model.named_parameters() if p.device != device]
    bad_buffers = [name for name, b in model.named_buffers() if b.device != device]
    if bad_parameters or bad_buffers:
        raise RuntimeError(
            "device restore failed before CE backward: "
            f"parameters={bad_parameters[:8]} buffers={bad_buffers[:8]}"
        )
    return {
        "device": str(device),
        "parameter_count": sum(1 for _ in model.parameters()),
        "buffer_count": sum(1 for _ in model.buffers()),
        "all_parameters_on_device": True,
        "all_buffers_on_device": True,
    }


def module_by_name(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    current = layer
    for part in name.split("."):
        current = getattr(current, part)
    return current


def build_full_pt2_state(model: torch.nn.Module, captured_modules: Sequence[Dict[str, object]], group_size: int) -> Dict[str, Any]:
    """Build a resumable state for all PT2-quantized linear modules.

    PT2's capture stream is not guaranteed to expose ternary metadata for every
    quantized module on every architecture.  Qwen3, for example, can complete
    official ATQ+SSR while some non-target modules do not carry a captured
    ``T`` tensor.  Those modules are still needed to reconstruct the deployed
    baseline, so we store them as raw deployed weights instead of aborting the
    checkpoint.  Modules with ``T`` keep the compact affine ternary state used
    by TernRefine diagnostics.
    """
    specs = module_specs(model)
    if len(specs) != len(captured_modules):
        raise RuntimeError(f"full PT2 capture mismatch specs={len(specs)} captures={len(captured_modules)}")
    payload: Dict[str, Any] = {
        "format": "CEGSP_PT2_FULL_STATE_V2_MIXED",
        "model_type": str(getattr(getattr(model, "config", None), "model_type", "")),
        "group_size": int(group_size),
        "modules": {},
        "ternary_module_count": 0,
        "raw_module_count": 0,
    }
    for index, (layer, name, module) in enumerate(specs):
        capture = captured_modules[index]
        t_perm = capture.get("T")
        module_id = f"{layer}.{name}"
        deployed = module.weight.detach().cpu().contiguous()
        if t_perm is None:
            payload["modules"][module_id] = {
                "encoding": "raw_weight",
                "layer": int(layer),
                "name": name,
                "weight": deployed.to(torch.float16),
                "original_shape": [int(deployed.shape[0]), int(deployed.shape[1])],
                "reason": "PT2 capture did not expose ternary T for this module",
            }
            payload["raw_module_count"] += 1
            continue
        t_perm = t_perm.to(torch.int8).cpu().contiguous()
        perm = capture["perm"].to(torch.long).cpu().contiguous()
        q_perm = deployed.float()[:, perm]
        rows, columns = q_perm.shape
        if tuple(t_perm.shape) != (rows, columns):
            raise RuntimeError(f"full PT2 shape mismatch module={layer}.{name} q={tuple(q_perm.shape)} T={tuple(t_perm.shape)}")
        blocks = (columns + group_size - 1) // group_size
        q_padded = torch.zeros((rows, blocks * group_size), dtype=torch.float32)
        t_padded = torch.zeros((rows, blocks * group_size), dtype=torch.float32)
        valid = torch.zeros_like(q_padded, dtype=torch.bool)
        q_padded[:, :columns] = q_perm
        t_padded[:, :columns] = t_perm.float()
        valid[:, :columns] = True
        q_group = q_padded.view(rows, blocks, group_size)
        t_group = t_padded.view(rows, blocks, group_size).round().to(torch.int8)
        mu_rows, alpha_rows = [], []
        for block in range(blocks):
            width = min(group_size, columns - block * group_size)
            mu, alpha, _ = affine_from_q_and_t(q_group[:, block, :width], t_group[:, block, :width])
            mu_rows.append(mu.squeeze(1))
            alpha_rows.append(alpha.squeeze(1))
        mu = torch.stack(mu_rows, dim=1).unsqueeze(-1).contiguous()
        alpha = torch.stack(alpha_rows, dim=1).unsqueeze(-1).contiguous()
        payload["modules"][module_id] = {
            "encoding": "affine_ternary",
            "layer": int(layer),
            "name": name,
            "T": t_group,
            "mu": mu,
            "alpha": alpha,
            "valid": valid.view(rows, blocks, group_size),
            "perm": perm,
            "original_shape": [int(rows), int(columns)],
            "group_size": int(group_size),
            "ssr": bool(capture["ssr"]),
        }
        payload["ternary_module_count"] += 1
    payload["module_count"] = len(payload["modules"])
    return payload


def save_full_pt2_state(payload: Dict[str, Any], out: Path) -> Dict[str, Any]:
    partial = out / "pt2_full_state.pt.partial"
    final = out / "pt2_full_state.pt"
    torch.save(payload, partial)
    os.replace(partial, final)
    manifest = {
        "format": payload["format"],
        "path": str(final),
        "module_count": int(payload["module_count"]),
        "ternary_module_count": int(payload.get("ternary_module_count", 0)),
        "raw_module_count": int(payload.get("raw_module_count", 0)),
        "bytes": int(final.stat().st_size),
        "group_size": int(payload["group_size"]),
    }
    write_json(out / "pt2_full_state_manifest.json", manifest)
    return manifest


def load_full_pt2_state(model: torch.nn.Module, checkpoint: Path) -> Dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("format") not in {"CEGSP_PT2_FULL_STATE_V1", "CEGSP_PT2_FULL_STATE_V2_MIXED"}:
        raise RuntimeError(f"unsupported PT2 checkpoint format: {payload.get('format')}")
    layers = get_decoder_layers(model)
    for row in payload["modules"].values():
        layer = layers[int(row["layer"])]
        module = module_by_name(layer, str(row["name"]))
        if row.get("encoding") == "raw_weight":
            weight = row["weight"]
            module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
            continue
        t = row["T"].to(torch.float32)
        q_perm = (row["mu"].float() + row["alpha"].float() * t).view(int(row["original_shape"][0]), -1)
        q_perm = q_perm[:, : int(row["original_shape"][1])]
        inverse = torch.argsort(row["perm"].long())
        q_original = q_perm[:, inverse]
        module.weight.data.copy_(q_original.to(device=module.weight.device, dtype=module.weight.dtype))
    return {
        "format": payload["format"],
        "module_count": int(payload["module_count"]),
        "ternary_module_count": int(payload.get("ternary_module_count", payload.get("module_count", 0))),
        "raw_module_count": int(payload.get("raw_module_count", 0)),
        "checkpoint": str(checkpoint),
    }


def reload_clean_pt2_model(
    pt2_quantize: Any,
    model_path: str,
    seqlen: int,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Create a clean model after evaluator offload and restore compact PT2 state.

    The official evaluator intentionally moves the model between devices.  On
    Qwen3 this can leave PT2/accelerate-managed parameters and buffers outside
    the normal recursive ``Module.to`` path.  Reusing that instance is unsafe;
    a fresh instance plus the saved compact state is deterministic and avoids
    re-running PT2 quantization.
    """
    fresh = pt2_quantize.get_model(model_path, seqlen)
    fresh_adapter = apply_qwen_pt2_layer_adapter(fresh, model_path)
    fresh.seqlen = seqlen
    fresh.to(device)
    load_info = load_full_pt2_state(fresh, checkpoint_path)
    return fresh, {"adapter": fresh_adapter, "checkpoint": load_info}


def sidecar_diagnostic(codes: Dict[int, Dict[str, Any]], perms: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    layers = sorted(codes)
    audit = audit_all(codes)
    qk_count = sum(len(v) for v in codes.values())
    bijection = all(
        sorted(perms[layer][key].tolist()) == list(range(codes[layer][key].original_shape[1]))
        for layer in layers
        for key in codes[layer]
    )
    complete = all(sorted(codes[layer]) == ["k", "q"] for layer in layers)
    return {
        "strict_parity_gate_skipped": True,
        "reason": "user requested continuing after E2-A0 strict absolute parity failed at BF16-scale residual",
        "qk_module_count": qk_count,
        "expected_qk_module_count": len(layers) * 2,
        "decoder_layers_from_sidecar": len(layers),
        "layer_keys_complete": complete,
        "group_size": next(iter(next(iter(codes.values())).values())).group_size if codes else None,
        "ternary_code_legal": audit["total_illegal_states"] == 0,
        "ternary_code_finite": all(
            bool(torch.isfinite(code.T.float()).all().item())
            for layer_codes in codes.values()
            for code in layer_codes.values()
        ),
        "ssr_bijection": bool(bijection),
        "code_audit": audit,
        "not_a_strict_health_pass": True,
    }


def apply_edit_list(codes: Dict[int, Dict[str, Any]], edits: Sequence[AffineEdit]) -> Dict[int, Dict[str, torch.Tensor]]:
    states = {layer: {key: code.T.clone() for key, code in layer_codes.items()} for layer, layer_codes in codes.items()}
    used = set()
    for edit in edits:
        donor_key = (edit.layer, edit.key, edit.row, edit.block, edit.donor)
        receiver_key = (edit.layer, edit.key, edit.row, edit.block, edit.receiver)
        if donor_key in used or receiver_key in used:
            raise RuntimeError(f"candidate collision: {donor_key} {receiver_key}")
        state = states[edit.layer][edit.key]
        if int(state[edit.row, edit.block, edit.donor].item()) == 0:
            raise RuntimeError(f"invalid donor state: {donor_key}")
        if int(state[edit.row, edit.block, edit.receiver].item()) != 0:
            raise RuntimeError(f"invalid receiver state: {receiver_key}")
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = int(edit.receiver_sign)
        used.add(donor_key)
        used.add(receiver_key)
    return states


def main() -> None:
    args = parse_args()
    started = time.time()
    if not args.skip_strict_parity_gate:
        raise RuntimeError("This conditional runner requires --skip-strict-parity-gate.")
    if args.group_size != 128 or args.calib_seq_len != 2048:
        raise ValueError("frozen E2-B protocol requires group_size=128 and seq_len=2048")
    if args.grad_samples != 1:
        raise ValueError("frozen E2-B protocol uses exactly one gradient batch")
    if args.val_start + args.val_samples > args.calib_nsamples:
        raise ValueError("validation slice exceeds calibration samples")
    if not torch.cuda.is_available():
        raise RuntimeError("E2-B requires CUDA")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "e2_qwen_pt2_hba_partial.json"
    result_path = out / "e2_qwen_pt2_hba_result.json"

    write_json(partial_path, {"run_id": args.run_id, "status": "started", "config": vars(args)})

    sys_path = str(Path(__file__).resolve().parent)
    if sys_path not in os.sys.path:
        os.sys.path.insert(0, sys_path)
    if args.pt2_root not in os.sys.path:
        os.sys.path.insert(0, args.pt2_root)
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root

    data_mod = importlib.import_module("pt2_llm.data")
    if hasattr(data_mod, "DATA_ROOT"):
        data_mod.DATA_ROOT = args.pt2_data_root
    pt2_quantize = importlib.import_module("quantize")
    pt2_quantize.args = make_pt2_args(args)
    pt2_quantize.groupsize = args.group_size

    log(f"loading Qwen sidecar {args.sidecar_dir}")
    codes, perms, qk_checkpoint = load_detached_artifacts(Path(args.sidecar_dir))
    layers = sorted(codes)
    diag = sidecar_diagnostic(codes, perms)
    if not (diag["layer_keys_complete"] and diag["ternary_code_legal"] and diag["ternary_code_finite"] and diag["ssr_bijection"]):
        raise RuntimeError(f"sidecar legality diagnostic failed: {diag}")
    base_hash = state_hash(codes)

    log(f"loading PT2 calibration nsamples={args.calib_nsamples}")
    calib_loader, _ = data_mod.get_loaders(
        "wikitext2",
        nsamples=args.calib_nsamples,
        seed=0,
        seqlen=args.calib_seq_len,
        model=args.model,
    )
    if len(calib_loader) != args.calib_nsamples:
        raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.calib_nsamples}")

    log("rebuilding official PT2 Qwen deployment; strict parity gate is recorded but skipped")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    adapter = apply_qwen_pt2_layer_adapter(model, args.model)
    model.seqlen = args.calib_seq_len
    checkpoint_manifest = None
    if args.pt2_checkpoint:
        log(f"loading compact full PT2 checkpoint {args.pt2_checkpoint}; skipping quant_sequential")
        model.to(device)
        checkpoint_info = load_full_pt2_state(model, Path(args.pt2_checkpoint))
        checkpoint_manifest = {**checkpoint_info, "resumed_without_quantization": True}
    else:
        qmod = importlib.import_module("pt2_llm.quantizer")
        gptqmod = importlib.import_module("pt2_llm.gptq")
        ssrmod = importlib.import_module("pt2_llm.gptq_ssr")
        capture = Capture()
        capture.store_all_t = True
        restore_capture = install_capture(qmod, gptqmod, ssrmod, capture, args.group_size)
        try:
            pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
        finally:
            restore_capture()
        model.to(device)
        full_state = build_full_pt2_state(model, capture.modules, args.group_size)
        checkpoint_manifest = save_full_pt2_state(full_state, out)
        write_json(partial_path, {
            "run_id": args.run_id,
            "status": "pt2_checkpoint_saved",
            "pt2_checkpoint": checkpoint_manifest,
            "elapsed_sec": time.time() - started,
        })
    checkpoint_path = Path(args.pt2_checkpoint) if args.pt2_checkpoint else Path(str(checkpoint_manifest["path"]))
    model.config.use_cache = False
    model.eval()
    if len(get_decoder_layers(model)) != len(layers):
        raise RuntimeError(f"decoder depth mismatch: model={len(get_decoder_layers(model))} sidecar={len(layers)}")

    restore_qk(model, qk_checkpoint)
    grad_batches = [calib_loader[0][0]]
    val_batches = [calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)]
    data_manifest = {
        "dataset": "wikitext2 via official PT2 loader",
        "calib_nsamples": len(calib_loader),
        "gradient_batches": len(grad_batches),
        "validation_start": args.val_start,
        "validation_samples": len(val_batches),
        "seq_len": args.calib_seq_len,
        "fit_sample_hash": batches_hash(grad_batches),
        "validation_sample_hash": batches_hash(val_batches),
        "selection_uses_w2_c4": False,
    }
    write_json(out / "calibration_manifest.json", data_manifest)

    log("evaluating rebuilt PT2 baseline W2/C4 once for Qwen reference")
    baseline_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.calib_seq_len)
    baseline_ppl = {
        "wikitext2_ppl": float(baseline_metrics["wikitext2_ppl"]),
        "c4_ppl": float(baseline_metrics["c4_ppl"]),
    }
    baseline_nll = {k: float(math.log(v)) for k, v in baseline_ppl.items()}

    # Do not attempt to restore the evaluator-mutated Qwen instance.  The
    # official qwen_eval path deliberately offloads modules and may install
    # device hooks that make an in-place restore incomplete.  Release it and
    # reload a clean model from the compact full PT2 state saved above.
    log("reloading a clean Qwen model from the saved PT2 checkpoint for CE/HBA")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model, clean_restore = reload_clean_pt2_model(
        pt2_quantize,
        args.model,
        args.calib_seq_len,
        checkpoint_path,
        device,
    )
    device_restore = {
        "strategy": "fresh_model_reload_from_compact_pt2_state",
        "in_place_restore_attempted": False,
        "all_parameters_on_device_after_reload": all(p.device == device for p in model.parameters()),
        "all_buffers_on_device_after_reload": all(b.device == device for b in model.buffers()),
        "checkpoint": str(checkpoint_path),
    }
    if not device_restore["all_parameters_on_device_after_reload"] or not device_restore["all_buffers_on_device_after_reload"]:
        raise RuntimeError(f"fresh PT2 model is not fully on {device}: {device_restore}")
    model.config.use_cache = False
    model.eval()
    # The fresh model has the same compact PT2 state, but re-apply the Q/K
    # sidecar state explicitly before the one CE backward pass.
    restore_qk(model, qk_checkpoint)
    write_json(partial_path, {
        "run_id": args.run_id,
        "status": "baseline_evaluated",
        "baseline_ppl": baseline_ppl,
        "baseline_nll": baseline_nll,
        "state_diagnostic": diag,
        "device_restore": device_restore,
        "pt2_checkpoint": checkpoint_manifest,
        "elapsed_sec": time.time() - started,
    })

    log("collecting exactly one quantized-point CE gradient")
    grads_original = collect_grads(model, grad_batches, layers, device, args.grad_samples)
    grads_ssr = {layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")} for layer in layers}

    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_summary: List[Dict[str, Any]] = []
    for layer in layers:
        candidates = build_top_candidates(codes, grads_ssr, layer, args.candidate_top_k)
        if len(candidates) < max(GRID):
            raise RuntimeError(f"layer {layer} has only {len(candidates)} candidates")
        candidates_by_layer[layer] = candidates
        layer_summary.append({
            "layer": int(layer),
            "candidate_count": len(candidates),
            "top_score": float(candidates[0].score),
            "top8_score_sum": float(sum(e.score for e in candidates[:8])),
        })
    layer_summary.sort(key=lambda row: (-row["top8_score_sum"], row["layer"]))
    manifest_hash = candidate_manifest(out / "candidate_manifest.jsonl", candidates_by_layer, layers)

    hba_rows: List[Dict[str, Any]] = []
    previous_val = None
    selected_k = None
    stop_reason = "reached_grid_end"
    for k in GRID:
        selected = select_prefix(candidates_by_layer, k, layers)
        row = row_for_patch(model, device, codes, perms, base_hash, selected, val_batches, k)
        hba_rows.append(row)
        write_json(out / "hba_curve_partial.json", {"grid": list(GRID), "rows": hba_rows, "selected_k_so_far": selected_k})
        log(f"HBA K={k}: val_nll={row['validation_nll']:.8f} edits={row['num_relocations']}")
        if previous_val is not None and not (row["validation_nll"] < previous_val):
            stop_reason = f"first_non_improvement_at_K={k}"
            break
        selected_k = k
        previous_val = row["validation_nll"]
        restore_qk(model, qk_checkpoint)
    if selected_k is None:
        selected_k = hba_rows[0]["k_per_layer"]
        stop_reason = "first_grid_point_only"

    selected_edits = select_prefix(candidates_by_layer, selected_k, layers)
    selected_states = apply_edit_list(codes, selected_edits)
    selected_audit = audit_all(codes, selected_states)
    selected_card = cardinality_violations(codes, selected_states)
    selected_patch = {
        "k_per_layer": int(selected_k),
        "num_relocations": len(selected_edits),
        "changed_coordinates": changed_coordinates(codes, selected_states),
        "expected_relocations": len(layers) * int(selected_k),
        "expected_changed_coordinates": 2 * len(layers) * int(selected_k),
        "selected_layers": layers,
        "module_counts": {key: sum(1 for e in selected_edits if e.key == key) for key in ("q", "k")},
        "qgp_score_sum": float(sum(e.score for e in selected_edits)),
        "audit": selected_audit,
        "cardinality_violations": int(selected_card),
        "state_hash_before": base_hash,
        "edit_ids": [
            [int(e.layer), str(e.key), int(e.row), int(e.block), int(e.donor), int(e.receiver), int(e.receiver_sign)]
            for e in selected_edits
        ],
    }
    write_json(out / "hba_curve.json", {"grid": list(GRID), "rows": hba_rows, "selected_k": selected_k, "stop_reason": stop_reason})
    write_json(out / "selected_patch.json", selected_patch)

    log(f"evaluating Qwen PT2+HBA selected patch on W2/C4; K*={selected_k}")
    apply_ssr_codes(model, codes, perms, selected_states)
    final_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.calib_seq_len)
    final_ppl = {
        "wikitext2_ppl": float(final_metrics["wikitext2_ppl"]),
        "c4_ppl": float(final_metrics["c4_ppl"]),
    }
    final_nll = {k: float(math.log(v)) for k, v in final_ppl.items()}
    delta = metric_delta(final_nll, baseline_nll)

    result = {
        "run_id": args.run_id,
        "experiment": "E2-B Qwen3-8B PT2 + HBA with strict parity skipped",
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "conditional_result": True,
            "strict_state_export_parity_gate": "skipped_by_user_request",
            "not_a_strict_E2_health_pass": True,
            "pt2_rebuilt": not bool(args.pt2_checkpoint),
            "pt2_checkpoint_saved_or_loaded": checkpoint_manifest,
            "baseline_evaluated_once": True,
            "hba_scope": f"all {len(layers)} decoder layers, Q/K only",
            "one_quantized_point_backward": True,
            "fixed_qgp_ranking": True,
            "hba_grid": list(GRID),
            "hba_stop": "first strict validation non-improvement",
            "no_rerank": True,
            "teacher_or_qat": False,
            "mu_alpha_frozen": True,
            "validation_does_not_use_w2_c4": True,
        },
        "state_diagnostic": diag,
        "architecture_adapter": adapter,
        "pt2_checkpoint": checkpoint_manifest,
        "device_restore": device_restore,
        "data": data_manifest,
        "baseline_pt2_metrics": {"absolute_ppl": baseline_ppl, "absolute_nll": baseline_nll, "finite": finite_metrics(baseline_ppl)},
        "candidate_summary": {
            "all_layers": layers,
            "candidate_top_k_per_layer": args.candidate_top_k,
            "total_candidates": sum(len(x) for x in candidates_by_layer.values()),
            "candidate_manifest_hash": manifest_hash,
            "layer_rows": layer_summary,
        },
        "hba": {
            "curve": hba_rows,
            "selected_k": int(selected_k),
            "stop_reason": stop_reason,
            "selected_patch": selected_patch,
        },
        "pt2_plus_hba_metrics": {
            "absolute_ppl": final_ppl,
            "absolute_nll": final_nll,
            "delta_vs_rebuilt_pt2_nll": delta,
            "finite": finite_metrics(final_ppl),
        },
        "gate": {
            "conditional_run_complete": True,
            "strict_parity_was_skipped": True,
            "hba_validation_curve_finite": all(row["finite"] for row in hba_rows),
            "hba_validation_curve_legal": all(row["legal"] for row in hba_rows),
            "selected_patch_exact_relocations": len(selected_edits) == len(layers) * selected_k,
            "selected_patch_exact_changed_coordinates": changed_coordinates(codes, selected_states) == 2 * len(layers) * selected_k,
            "selected_patch_finite": finite_metrics(final_ppl),
            "selected_patch_legal": selected_audit["total_illegal_states"] == 0 and selected_card == 0,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "timing": {"total_sec": time.time() - started},
    }
    write_json(out / "config.json", vars(args))
    write_json(result_path, result)
    log(json.dumps(result["gate"], ensure_ascii=False))
    log(f"wrote {result_path}")


if __name__ == "__main__":
    main()
