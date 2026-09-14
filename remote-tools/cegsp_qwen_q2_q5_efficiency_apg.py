#!/usr/bin/env python3
"""Qwen Q2/Q5: efficiency profile and APG first-stop audit.

This runner reuses the frozen Qwen3-8B PT2 checkpoint and the detached
TernRefine sidecar. It does not re-run PT2, does not recompute the baseline,
and does not save any full model checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import set_seed

from cegsp_e2_qwen_pt2_health_a100 import apply_qwen_pt2_layer_adapter
from cegsp_e2_qwen_pt2_hba_skipgate_a100 import load_full_pt2_state, sidecar_diagnostic
from cegsp_p7_a100_scaling import (
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    get_decoder_layers,
)
from cegsp_p9s2_detached_pt2_plugin import (
    apply_ssr_codes,
    cardinality_violations,
    load_detached_artifacts,
    restore_qk,
)
from cegsp_pt2_hba_detached_a100 import (
    GRID,
    apply_edit_list,
    batches_hash,
    row_for_patch,
    select_prefix,
    state_hash,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B")
    p.add_argument("--sidecar-dir", default="/CEGSP/model/tqgsp-runs/CEGSP-E2-QWEN3-8B-PT2-DIRECT-EVAL-NS128-A100-20260909-42151-r1/pt2_qwen3_8b_sidecar")
    p.add_argument("--pt2-checkpoint", default="/CEGSP/model/tqgsp-runs/CEGSP-E2-QWEN3-8B-PT2-DIRECT-EVAL-NS128-A100-20260909-42151-r1/pt2_full_state.pt")
    p.add_argument("--replication-dir", default="/CEGSP/model/tqgsp-runs/CEGSP-E2-QWEN3-8B-PT2-HBA-REPLICATION-A100-20260909-42131-r3")
    p.add_argument("--run-id", default="CEGSP-Q2Q5-QWEN3-8B-EFFICIENCY-APG-A100-20260911-42115-r1")
    p.add_argument("--out-dir", default="/CEGSP/model/tqgsp-runs")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--calib-nsamples", type=int, default=128)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--fit-index", type=int, default=0)
    p.add_argument("--val-start", type=int, default=1)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--grad-samples", type=int, default=1)
    p.add_argument("--candidate-top-k", type=int, default=256)
    p.add_argument("--profile-repeats", type=int, default=3)
    p.add_argument("--warmup-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260911)
    return p.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed(start: float) -> float:
    cuda_sync()
    return time.perf_counter() - start


def install_cache_only_datasets_shim() -> None:
    if "datasets" in sys.modules:
        return
    module = types.ModuleType("datasets")

    def _cache_miss(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("datasets shim is cache-only; required PT2 torch cache is missing")

    module.load_dataset = _cache_miss
    module.load_from_disk = _cache_miss
    sys.modules["datasets"] = module


def summarize(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_trials(trials: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "gradient_sec",
        "candidate_qgp_scoring_sec",
        "compatibility_selection_sec",
        "apg_validation_forward_sec",
        "profiled_pipeline_sec",
        "max_memory_allocated_gb",
        "max_memory_reserved_gb",
    ]
    return {key: summarize([float(t[key]) for t in trials]) for key in keys}


def audit_apg_curves(replication_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(replication_dir.glob("replicate_*/apg_curve.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        curve = data.get("rows", [])
        if not curve:
            continue
        vals = [(int(r["k_per_layer"]), float(r["validation_nll"])) for r in curve]
        observed_argmin_k, observed_argmin_val = min(vals, key=lambda item: item[1])
        first_stop_selected_k = int(data.get("selected_k", observed_argmin_k))
        first_non_improve_k = None
        prev = None
        selected_by_recomputed_rule = None
        for k, val in vals:
            if prev is not None and not (val < prev[1]):
                first_non_improve_k = k
                break
            selected_by_recomputed_rule = k
            prev = (k, val)
        rows.append({
            "replicate": path.parent.name,
            "curve_path": str(path),
            "evaluated_grid": [k for k, _ in vals],
            "validation_nll_by_k": {str(k): val for k, val in vals},
            "stored_selected_k": first_stop_selected_k,
            "recomputed_first_stop_selected_k": int(selected_by_recomputed_rule),
            "first_non_improvement_k": first_non_improve_k,
            "observed_argmin_k": int(observed_argmin_k),
            "observed_argmin_validation_nll": float(observed_argmin_val),
            "first_stop_equals_observed_argmin": bool(first_stop_selected_k == observed_argmin_k),
            "prefix_only_caveat": max(k for k, _ in vals) < max(GRID),
        })
    return {
        "source_replication_dir": str(replication_dir),
        "num_replicates": len(rows),
        "rows": rows,
        "all_first_stop_equal_observed_argmin": bool(rows) and all(r["first_stop_equals_observed_argmin"] for r in rows),
        "note": "Q5 uses the APG prefix actually evaluated by the frozen E2 Qwen replication; it does not rerun model inference.",
    }


def profile_once(
    model: torch.nn.Module,
    device: torch.device,
    calib_loader: Sequence[Any],
    codes: Dict[int, Dict[str, Any]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    qk_checkpoint: Dict[int, Dict[str, torch.Tensor]],
    layers: Sequence[int],
    args: argparse.Namespace,
    base_hash: str,
    write_partials_to: Path | None,
    trial_name: str,
) -> Dict[str, Any]:
    restore_qk(model, qk_checkpoint)
    model.to(device)
    model.config.use_cache = False
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cuda_sync()
    total_start = time.perf_counter()

    fit_batches = [calib_loader[args.fit_index][0]]
    val_batches = [calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)]

    start = time.perf_counter()
    grads_original = collect_grads(model, fit_batches, layers, device, args.grad_samples)
    grads_ssr = {layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")} for layer in layers}
    gradient_sec = elapsed(start)

    start = time.perf_counter()
    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    for layer in layers:
        candidates = build_top_candidates(codes, grads_ssr, layer, args.candidate_top_k)
        if len(candidates) < max(GRID):
            raise RuntimeError(f"layer {layer} has only {len(candidates)} candidates")
        candidates_by_layer[layer] = candidates
    candidate_qgp_scoring_sec = elapsed(start)

    start = time.perf_counter()
    selected_by_k = {int(k): select_prefix(candidates_by_layer, int(k), layers) for k in GRID}
    compatibility_selection_sec = elapsed(start)

    hba_rows: List[Dict[str, Any]] = []
    selected_k = None
    prev_val = None
    stop_reason = "reached_grid_end"
    start = time.perf_counter()
    for k in GRID:
        row = row_for_patch(model, device, codes, perms, base_hash, selected_by_k[int(k)], val_batches, int(k))
        hba_rows.append(row)
        if write_partials_to is not None:
            write_json(write_partials_to, {
                "trial": trial_name,
                "latest_k": int(k),
                "completed_rows": hba_rows,
                "selected_k_so_far": selected_k,
            })
        log(f"{trial_name}: K={k} val_nll={row['validation_nll']:.8f}")
        if prev_val is not None and not (row["validation_nll"] < prev_val):
            stop_reason = f"first_non_improvement_at_K={k}"
            break
        selected_k = int(k)
        prev_val = float(row["validation_nll"])
        restore_qk(model, qk_checkpoint)
        model.to(device)
        model.eval()
    apg_validation_forward_sec = elapsed(start)
    profiled_pipeline_sec = elapsed(total_start)

    selected_edits = selected_by_k[int(selected_k)]
    selected_states = apply_edit_list(codes, selected_edits)
    selected_audit = audit_all(codes, selected_states)
    selected_cardinality_violations = cardinality_violations(codes, selected_states)
    apply_ssr_codes(model, codes, perms, selected_states)

    return {
        "trial": trial_name,
        "gradient_sec": float(gradient_sec),
        "candidate_qgp_scoring_sec": float(candidate_qgp_scoring_sec),
        "compatibility_selection_sec": float(compatibility_selection_sec),
        "apg_validation_forward_sec": float(apg_validation_forward_sec),
        "profiled_pipeline_sec": float(profiled_pipeline_sec),
        "candidate_moves_total": int(sum(len(v) for v in candidates_by_layer.values())),
        "candidate_top_k_per_layer": int(args.candidate_top_k),
        "num_layers": int(len(layers)),
        "qk_modules": int(sum(len(v) for v in codes.values())),
        "apg_curve": hba_rows,
        "selected_k": int(selected_k),
        "stop_reason": stop_reason,
        "selected_relocations": int(len(selected_edits)),
        "selected_changed_coordinates": int(changed_coordinates(codes, selected_states)),
        "selected_cardinality_violations": int(selected_cardinality_violations),
        "selected_audit": selected_audit,
        "selected_legal": bool(selected_audit["total_illegal_states"] == 0 and selected_cardinality_violations == 0),
        "selected_finite": bool(all(row["finite"] for row in hba_rows)),
        "max_memory_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        "max_memory_reserved_gb": float(torch.cuda.max_memory_reserved() / (1024 ** 3)),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "q2q5_partial.json", {"status": "started", "run_id": args.run_id, "config": vars(args)})

    if args.pt2_root not in sys.path:
        sys.path.insert(0, args.pt2_root)
    tool_dir = str(Path(__file__).resolve().parent)
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root
    install_cache_only_datasets_shim()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    q5 = audit_apg_curves(Path(args.replication_dir))
    write_json(out / "q5_apg_first_stop_audit.json", q5)

    setup_start = time.perf_counter()
    data_mod = importlib.import_module("pt2_llm.data")
    if hasattr(data_mod, "DATA_ROOT"):
        data_mod.DATA_ROOT = args.pt2_data_root
    pt2_quantize = importlib.import_module("quantize")

    log("loading detached Qwen PT2 sidecar")
    codes, perms, qk_checkpoint = load_detached_artifacts(Path(args.sidecar_dir))
    layers = sorted(codes)
    diag = sidecar_diagnostic(codes, perms)
    if len(layers) != 36 or sum(len(v) for v in codes.values()) != 72:
        raise RuntimeError(f"unexpected Qwen scope layers={len(layers)} qk={sum(len(v) for v in codes.values())}")
    if not (diag["layer_keys_complete"] and diag["ternary_code_legal"] and diag["ternary_code_finite"] and diag["ssr_bijection"]):
        raise RuntimeError(f"sidecar diagnostic failed: {diag}")
    base_hash = state_hash(codes)

    log("loading PT2 calibration cache")
    calib_loader, _ = data_mod.get_loaders("wikitext2", nsamples=args.calib_nsamples, seed=0, seqlen=args.calib_seq_len, model=args.model)
    if args.fit_index in range(args.val_start, args.val_start + args.val_samples):
        raise RuntimeError("fit index overlaps APG selection slice")
    fit_hash = batches_hash([calib_loader[args.fit_index][0]])
    val_hash = batches_hash([calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)])

    log("loading Qwen model and frozen PT2 checkpoint")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    adapter = apply_qwen_pt2_layer_adapter(model, args.model)
    model.seqlen = args.calib_seq_len
    model.to(device)
    checkpoint_info = load_full_pt2_state(model, Path(args.pt2_checkpoint))
    model.config.use_cache = False
    model.eval()
    if len(get_decoder_layers(model)) != len(layers):
        raise RuntimeError("decoder depth mismatch")
    restore_qk(model, qk_checkpoint)
    setup_load_sec = elapsed(setup_start)

    warmup_trials = []
    for i in range(args.warmup_repeats):
        log(f"warmup trial {i + 1}/{args.warmup_repeats}")
        warmup_trials.append(profile_once(
            model, device, calib_loader, codes, perms, qk_checkpoint, layers, args, base_hash,
            None, f"warmup_{i + 1}",
        ))

    profiled_trials = []
    for i in range(args.profile_repeats):
        log(f"profile trial {i + 1}/{args.profile_repeats}")
        profiled_trials.append(profile_once(
            model, device, calib_loader, codes, perms, qk_checkpoint, layers, args, base_hash,
            out / "q2_efficiency_profile_partial.json", f"profile_{i + 1}",
        ))
        write_json(out / "q2_efficiency_profile_partial.json", {
            "status": "running",
            "completed_profile_trials": len(profiled_trials),
            "profile_trials": profiled_trials,
        })

    result = {
        "status": "complete",
        "run_id": args.run_id,
        "experiment": "Q2/Q5 Qwen3-8B efficiency profile and APG first-stop audit",
        "config": vars(args),
        "protocol": {
            "model": "Qwen3-8B",
            "initializer": "frozen PT2 checkpoint reused",
            "strict_qwen_parity_gate": "skipped_conditional_endpoint",
            "no_pt2_rerun": True,
            "baseline_recomputed": False,
            "full_model_checkpoint_saved": False,
            "scope": "36 decoder layers, Q/K only",
            "one_backward_per_profile_trial": True,
            "fixed_qgp_ranking_within_trial": True,
            "apg_grid": list(GRID),
            "selection_uses_w2_c4": False,
        },
        "state": {
            "checkpoint": checkpoint_info,
            "sidecar_dir": str(args.sidecar_dir),
            "sidecar_diagnostic": diag,
            "architecture_adapter": adapter,
            "state_hash": base_hash,
        },
        "data": {
            "fit_index": int(args.fit_index),
            "fit_hash": fit_hash,
            "selection_indices": list(range(args.val_start, args.val_start + args.val_samples)),
            "selection_hash": val_hash,
            "fit_disjoint_from_selection": True,
            "calib_nsample": int(args.calib_nsamples),
            "seq_len": int(args.calib_seq_len),
        },
        "q2_efficiency_profile": {
            "setup_load_sec": float(setup_load_sec),
            "warmup_trials": warmup_trials,
            "profile_trials": profiled_trials,
            "summary": summarize_trials(profiled_trials),
        },
        "q5_apg_first_stop_vs_val_argmin": q5,
        "gate": {
            "status_complete": True,
            "q2_reused_existing_pt2_checkpoint": True,
            "q2_no_baseline_recompute": True,
            "q2_no_full_model_save": True,
            "q2_all_trials_finite": all(t["selected_finite"] for t in profiled_trials),
            "q2_all_trials_legal": all(t["selected_legal"] for t in profiled_trials),
            "q2_all_trials_exact_576_relocations": all(t["selected_relocations"] == 576 for t in profiled_trials),
            "q2_all_trials_exact_1152_changed_coordinates": all(t["selected_changed_coordinates"] == 1152 for t in profiled_trials),
            "q5_three_replicates_available": q5["num_replicates"] == 3,
            "q5_first_stop_equals_observed_argmin_all": q5["all_first_stop_equal_observed_argmin"],
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
        "timing": {"wall_sec": float(time.time() - started)},
    }
    write_json(out / "q2q5_result.json", result)
    log(json.dumps(result["gate"], ensure_ascii=False))
    log(f"wrote {out / 'q2q5_result.json'}")


if __name__ == "__main__":
    main()
