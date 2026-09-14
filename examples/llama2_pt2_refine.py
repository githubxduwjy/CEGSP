#!/usr/bin/env python3
"""PT2 + HBA replication across fitting slices on one frozen PT2 state.

This runner reuses the audited detached PT2 sidecar and saved PT2 reference
metrics.  It rebuilds the official PT2 deployment once, verifies state parity
once, and then runs several TernRefine/HBA replicates that differ only in the
single fitting calibration sample used for the quantized-point CE gradient.

The selection slice is fixed across replicates, W2/C4 are untouched until each
replicate has selected K*, and the PT2 baseline is never recomputed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import statistics
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import set_seed

from ternrefine.large_model_affine import (
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    evaluate_nll,
    get_decoder_layers,
    metric_delta,
)
from ternrefine.pt2_sidecar import (
    apply_ssr_codes,
    cardinality_violations,
    finite_metrics,
    load_detached_artifacts,
    official_metrics,
    restore_qk,
)
from ternrefine.pt2_hba import (
    GRID,
    apply_edit_list,
    candidate_manifest,
    edit_key,
    load_reference,
    make_full_pt2_args,
    row_for_patch,
    select_prefix,
    state_hash,
    state_parity_gate,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_model_on_device(model: torch.nn.Module, device: torch.device) -> None:
    """Restore a consistent CUDA/eval state after helpers that may offload modules."""
    model.to(device)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.eval()


def completed_replicate_result(rep_dir: Path) -> Dict[str, Any] | None:
    """Reuse a fully completed replicate when retrying the same registered run."""
    summary_path = rep_dir / "run_summary.json"
    if not summary_path.exists():
        return None
    result = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = result.get("gate", {})
    required_gate_keys = (
        "fit_disjoint_from_selection",
        "gradient_finite",
        "backward_count_is_one",
        "rerank_count_is_zero",
        "apg_curve_finite",
        "apg_curve_legal",
        "selected_patch_exact_relocations",
        "selected_patch_exact_changed_coordinates",
        "selected_patch_legal",
        "selected_patch_finite",
    )
    if result.get("replicate") != rep_dir.name:
        return None
    if not all(bool(gate.get(key, False)) for key in required_gate_keys):
        return None
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/Llama-2-7b-hf")
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--pt2-checkpoint", default="", help="Load an existing CEGSP full PT2 checkpoint and skip official quant_sequential rebuild.")
    p.add_argument("--reference-json", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--calib-nsamples", type=int, default=128)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--val-start", type=int, default=1)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--fit-indices", default="0,17,33")
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--candidate-top-k", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def parse_indices(spec: str) -> List[int]:
    values = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"fit indices must be unique: {values}")
    return values


def tensor_hash(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    x = tensor.detach().cpu().contiguous()
    h.update(str(x.dtype).encode())
    h.update(str(tuple(x.shape)).encode())
    h.update(x.numpy().tobytes())
    return h.hexdigest()


def batches_hash(batches: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for batch in batches:
        h.update(tensor_hash(batch).encode())
    return h.hexdigest()


def rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def spearman_from_scores(a: Dict[Any, float], b: Dict[Any, float]) -> Dict[str, Any]:
    keys = sorted(set(a).intersection(b))
    if len(keys) < 2:
        return {"common": len(keys), "rho": None}
    ar = rank([a[k] for k in keys])
    br = rank([b[k] for k in keys])
    return {"common": len(keys), "rho": pearson(ar, br)}


def summarize(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def make_data_manifest(
    calib_loader: Sequence[Any],
    fit_index: int,
    val_start: int,
    val_samples: int,
    seq_len: int,
) -> Dict[str, Any]:
    fit_batches = [calib_loader[fit_index][0]]
    val_batches = [calib_loader[i][0] for i in range(val_start, val_start + val_samples)]
    return {
        "dataset": "wikitext2 via official PT2 loader",
        "calib_nsamples": len(calib_loader),
        "fit_index": int(fit_index),
        "nominal_fit_token_offset": int(fit_index * seq_len),
        "gradient_batches": 1,
        "validation_start": int(val_start),
        "validation_samples": int(val_samples),
        "validation_indices": list(range(val_start, val_start + val_samples)),
        "seq_len": int(seq_len),
        "fit_sample_hash": batches_hash(fit_batches),
        "validation_sample_hash": batches_hash(val_batches),
        "fit_disjoint_from_selection": fit_index not in set(range(val_start, val_start + val_samples)),
        "note": "Only the fit_index changes across replicates. The selection slice is fixed, and W2/C4 are not used for stopping.",
    }


def install_cache_only_datasets_shim() -> None:
    """Let PT2's cached loader import on environments with broken datasets.

    PT2's data module imports ``datasets`` at module import time even when
    ``get_loaders`` immediately returns a torch-saved cache file.  This shim is
    intentionally cache-only: if a cache miss tries to call load_from_disk, the
    run fails loudly instead of silently changing the data protocol.
    """
    if "datasets" in sys.modules:
        return
    module = types.ModuleType("datasets")

    def _cache_miss(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("datasets shim is cache-only; required PT2 torch cache is missing")

    module.load_dataset = _cache_miss
    module.load_from_disk = _cache_miss
    sys.modules["datasets"] = module


def run_one_replicate(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    out: Path,
    rep_name: str,
    fit_index: int,
    calib_loader: Sequence[Any],
    codes: Dict[int, Dict[str, Any]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    qk_checkpoint: Dict[int, Dict[str, torch.Tensor]],
    layers: Sequence[int],
    base_hash: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    rep_dir = out / rep_name
    rep_dir.mkdir(parents=True, exist_ok=True)
    restore_qk(model, qk_checkpoint)
    ensure_model_on_device(model, device)
    model.zero_grad(set_to_none=True)

    fit_batches = [calib_loader[fit_index][0]]
    val_batches = [calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)]
    data_manifest = make_data_manifest(calib_loader, fit_index, args.val_start, args.val_samples, args.calib_seq_len)
    if not data_manifest["fit_disjoint_from_selection"]:
        raise RuntimeError(f"{rep_name} fit index overlaps fixed selection split")
    (rep_dir / "fit_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    (rep_dir / "selection_manifest.json").write_text(
        json.dumps({
            "validation_start": args.val_start,
            "validation_samples": args.val_samples,
            "validation_indices": data_manifest["validation_indices"],
            "validation_sample_hash": data_manifest["validation_sample_hash"],
        }, indent=2),
        encoding="utf-8",
    )

    log(f"{rep_name}: collecting one quantized-point CE gradient at fit_index={fit_index}")
    grad_started = time.time()
    grads_original = collect_grads(model, fit_batches, layers, device, 1)
    grads_ssr = {
        layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")}
        for layer in layers
    }
    grad_audit = {
        "backward_count": 1,
        "elapsed_sec": time.time() - grad_started,
        "all_gradient_tensors_finite": all(
            bool(torch.isfinite(grads_ssr[layer][key]).all().item())
            for layer in layers for key in ("q", "k")
        ),
        "gradient_tensor_count": sum(len(grads_ssr[layer]) for layer in layers),
    }
    (rep_dir / "gradient_audit.json").write_text(json.dumps(grad_audit, indent=2), encoding="utf-8")

    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_summary: List[Dict[str, Any]] = []
    score_map: Dict[Tuple[Any, ...], float] = {}
    for layer in layers:
        candidates = build_top_candidates(codes, grads_ssr, layer, args.candidate_top_k)
        if len(candidates) < max(GRID):
            raise RuntimeError(f"{rep_name}: layer {layer} has only {len(candidates)} candidates")
        candidates_by_layer[layer] = candidates
        for edit in candidates:
            score_map[edit_key(edit)] = float(edit.score)
        layer_summary.append({
            "layer": int(layer),
            "candidate_count": len(candidates),
            "top_score": float(candidates[0].score),
            "top8_score_sum": float(sum(e.score for e in candidates[:8])),
        })
    layer_summary.sort(key=lambda row: (-row["top8_score_sum"], row["layer"]))
    manifest_hash = candidate_manifest(rep_dir / "candidate_manifest.jsonl", candidates_by_layer, layers)
    ranking_manifest = {
        "candidate_manifest_hash": manifest_hash,
        "candidate_top_k_per_layer": args.candidate_top_k,
        "total_candidates": sum(len(x) for x in candidates_by_layer.values()),
        "ranking": "one fixed QGP ranking for this replicate",
        "rerank_count": 0,
        "layer_rows": layer_summary,
    }
    (rep_dir / "ranking_manifest.json").write_text(json.dumps(ranking_manifest, indent=2), encoding="utf-8")

    hba_rows: List[Dict[str, Any]] = []
    baseline_row = row_for_patch(model, device, codes, perms, base_hash, [], val_batches, 0)
    hba_rows.append(baseline_row)
    restore_qk(model, qk_checkpoint)
    ensure_model_on_device(model, device)
    previous_val = baseline_row["validation_nll"]
    selected_k = 0
    stop_reason = "reached_grid_end"
    for k in GRID:
        selected = select_prefix(candidates_by_layer, k, layers)
        row = row_for_patch(model, device, codes, perms, base_hash, selected, val_batches, k)
        hba_rows.append(row)
        (rep_dir / "apg_curve_partial.json").write_text(
            json.dumps(
                {
                    "grid": [0, *list(GRID)],
                    "completed_rows": hba_rows,
                    "latest_k": k,
                    "selected_k_so_far": selected_k,
                    "stop_reason_so_far": stop_reason,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log(f"{rep_name}: HBA K={k} val_nll={row['validation_nll']:.8f} edits={row['num_relocations']}")
        if not (row["validation_nll"] < previous_val):
            stop_reason = f"first_non_improvement_at_K={k}"
            (rep_dir / "apg_curve_partial.json").write_text(
                json.dumps(
                    {
                        "grid": [0, *list(GRID)],
                        "completed_rows": hba_rows,
                        "latest_k": k,
                        "selected_k_so_far": selected_k,
                        "stop_reason_so_far": stop_reason,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            break
        selected_k = k
        previous_val = row["validation_nll"]
        restore_qk(model, qk_checkpoint)
        ensure_model_on_device(model, device)

    selected_edits = select_prefix(candidates_by_layer, selected_k, layers)
    selected_states = apply_edit_list(codes, selected_edits)
    selected_audit = audit_all(codes, selected_states)
    selected_card = cardinality_violations(codes, selected_states)
    selected_row = next(row for row in hba_rows if row["k_per_layer"] == selected_k)
    if not selected_row["legal"] or not selected_row["finite"]:
        raise RuntimeError(f"{rep_name}: selected HBA patch failed legal/finite validation")

    selected_patch = {
        "k_per_layer": int(selected_k),
        "num_relocations": len(selected_edits),
        "changed_coordinates": changed_coordinates(codes, selected_states),
        "selected_layers": sorted(int(x) for x in {e.layer for e in selected_edits}),
        "module_counts": {key: sum(1 for e in selected_edits if e.key == key) for key in ("q", "k")},
        "qgp_score_sum": float(sum(e.score for e in selected_edits)),
        "state_hash_before": base_hash,
        "state_hash_after": state_hash({
            layer: {
                key: type("State", (), {"T": selected_states[layer][key]})()
                for key in selected_states[layer]
            }
            for layer in selected_states
        }),
        "audit": selected_audit,
        "cardinality_violations": selected_card,
        "edit_ids": [edit_key(e) for e in selected_edits],
    }
    selected_payload = json.dumps(selected_patch, sort_keys=True, separators=(",", ":"))
    (rep_dir / "selected_patch_pre_eval.json").write_text(json.dumps(selected_patch, indent=2), encoding="utf-8")
    (rep_dir / "selected_patch_pre_eval_hash.txt").write_text(
        hashlib.sha256(selected_payload.encode()).hexdigest() + "\n",
        encoding="utf-8",
    )

    log(f"{rep_name}: evaluating selected patch on untouched W2/C4; K*={selected_k}")
    apply_ssr_codes(model, codes, perms, selected_states)
    final_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.calib_seq_len)
    ensure_model_on_device(model, device)
    if not finite_metrics(final_metrics):
        raise RuntimeError(f"{rep_name}: selected PT2+HBA metrics are nonfinite: {final_metrics}")
    final_ppl = {
        "wikitext2_ppl": float(final_metrics["wikitext2_ppl"]),
        "c4_ppl": float(final_metrics["c4_ppl"]),
    }
    final_nll = {key: float(math.log(value)) for key, value in final_ppl.items()}
    reference_nll = {
        "wikitext2_ppl": reference["wikitext2_nll"],
        "c4_ppl": reference["c4_nll"],
    }
    delta_nll = metric_delta(final_nll, reference_nll)

    (rep_dir / "apg_curve.json").write_text(
        json.dumps({"grid": [0, *list(GRID)], "rows": hba_rows, "selected_k": selected_k, "stop_reason": stop_reason}, indent=2),
        encoding="utf-8",
    )
    (rep_dir / "selected_patch.json").write_text(json.dumps(selected_patch, indent=2), encoding="utf-8")
    (rep_dir / "selected_patch_hash.txt").write_text(hashlib.sha256(selected_payload.encode()).hexdigest() + "\n", encoding="utf-8")

    result = {
        "replicate": rep_name,
        "fit_index": int(fit_index),
        "nominal_fit_token_offset": int(fit_index * args.calib_seq_len),
        "data": data_manifest,
        "gradient_audit": grad_audit,
        "ranking_manifest": ranking_manifest,
        "apg": {"baseline_row": baseline_row, "curve": hba_rows, "selected_k": int(selected_k), "stop_reason": stop_reason},
        "selected_patch": selected_patch,
        "metrics": {
            "absolute_ppl": final_ppl,
            "absolute_nll": final_nll,
            "delta_vs_saved_pt2_nll": delta_nll,
        },
        "gate": {
            "fit_disjoint_from_selection": bool(data_manifest["fit_disjoint_from_selection"]),
            "gradient_finite": bool(grad_audit["all_gradient_tensors_finite"]),
            "backward_count_is_one": True,
            "rerank_count_is_zero": True,
            "apg_curve_finite": all(row["finite"] for row in hba_rows),
            "apg_curve_legal": all(row["legal"] for row in hba_rows),
            "selected_patch_exact_relocations": len(selected_edits) == len(layers) * selected_k,
            "selected_patch_exact_changed_coordinates": changed_coordinates(codes, selected_states) == 2 * len(layers) * selected_k,
            "selected_patch_legal": selected_audit["total_illegal_states"] == 0 and selected_card == 0,
            "selected_patch_finite": finite_metrics(final_ppl),
        },
        "score_map": {str(k): v for k, v in score_map.items()},
    }
    (rep_dir / "run_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    restore_qk(model, qk_checkpoint)
    ensure_model_on_device(model, device)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.group_size != 128 or args.calib_seq_len != 2048:
        raise ValueError("PT2+HBA replication requires group_size=128 and calib_seq_len=2048")
    if not torch.cuda.is_available():
        raise RuntimeError("PT2+HBA replication requires CUDA")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    fit_indices = parse_indices(args.fit_indices)
    if args.val_start + args.val_samples > args.calib_nsamples:
        raise RuntimeError("validation slice exceeds calibration loader")
    if any(i < 0 or i >= args.calib_nsamples for i in fit_indices):
        raise RuntimeError(f"fit index outside calibration loader: {fit_indices}")
    if any(args.val_start <= i < args.val_start + args.val_samples for i in fit_indices):
        raise RuntimeError("fit indices must be disjoint from the fixed selection slice")

    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    reference = load_reference(Path(args.reference_json))

    if args.pt2_root not in sys.path:
        sys.path.insert(0, args.pt2_root)
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root
    install_cache_only_datasets_shim()
    data_mod = __import__("pt2_llm.data", fromlist=["get_loaders"])
    pt2_quantize = importlib.import_module("quantize")
    pt2_quantize.args = make_full_pt2_args(SimpleNamespace(**vars(args), grad_samples=1))
    pt2_quantize.groupsize = args.group_size

    log(f"loading detached PT2 sidecar from {args.sidecar_dir}")
    codes, perms, qk_checkpoint = load_detached_artifacts(Path(args.sidecar_dir))
    layers = sorted(codes)
    if len(layers) != 32 or any(sorted(layer_codes) != ["k", "q"] for layer_codes in codes.values()):
        raise RuntimeError(f"unexpected detached scope: layers={len(layers)}")
    base_hash = state_hash(codes)

    calib_loader, _ = data_mod.get_loaders(
        "wikitext2", nsamples=args.calib_nsamples, seed=0,
        seqlen=args.calib_seq_len, model=args.model
    )
    if len(calib_loader) != args.calib_nsamples:
        raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.calib_nsamples}")

    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    model.seqlen = args.calib_seq_len
    if args.pt2_checkpoint:
        from ternrefine.artifact_state import load_full_pt2_state

        ckpt_path = Path(args.pt2_checkpoint)
        if not ckpt_path.exists():
            raise RuntimeError(f"PT2 checkpoint does not exist: {ckpt_path}")
        log(f"loading frozen PT2 deployment from {ckpt_path}; quant_sequential skipped")
        model.to(device)
        load_info = load_full_pt2_state(model, ckpt_path)
        log(json.dumps({"pt2_checkpoint_load": load_info}, ensure_ascii=False))
    else:
        log(f"rebuilding official PT2 deployment once on {torch.cuda.get_device_name(0)}; baseline evaluation skipped")
        pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
    ensure_model_on_device(model, device)
    if len(get_decoder_layers(model)) != 32:
        raise RuntimeError(f"model decoder depth mismatch: {len(get_decoder_layers(model))}")

    parity = state_parity_gate(model, codes, perms, qk_checkpoint)
    state_fingerprint = {
        "state_hash": base_hash,
        "state_parity": parity,
        "reference_pt2_metrics": reference,
        "qk_scope_layers": layers,
        "qk_module_count": 64,
        "group_size": args.group_size,
        "ssr_permutation_required": True,
    }
    (out / "pt2_state_fingerprint.json").write_text(json.dumps(state_fingerprint, indent=2), encoding="utf-8")
    if not parity.get("pass", False):
        result = {
            "status": "NOT_RUN_STATE_PARITY_FAILED",
            "run_id": args.run_id,
            "config": vars(args),
            "pt2_state_fingerprint": state_fingerprint,
            "elapsed_sec": time.time() - started,
        }
        (out / "pt2_hba_replication_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log("state parity failed; wrote failure summary")
        return

    selection_hash = batches_hash([calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)])
    prereg = {
        "experiment": "PT2 + HBA fitting-split replication",
        "question": "Same frozen official PT2 state, different fitting calibration sample, same selection split.",
        "fit_indices": fit_indices,
        "nominal_fit_token_offsets": [i * args.calib_seq_len for i in fit_indices],
        "selection_indices": list(range(args.val_start, args.val_start + args.val_samples)),
        "selection_hash": selection_hash,
        "apg_grid": list(GRID),
        "apg_grid_with_baseline": [0, *list(GRID)],
        "stopping": "first strict validation non-improvement",
        "first_prefix_compared_to_q0": True,
        "scope": "32 decoder layers, Q/K only",
        "baseline_recomputed": False,
        "baseline_evaluation_skipped": True,
        "teacher_or_qat": False,
        "one_backward_per_replicate": True,
        "no_rerank": True,
        "mu_alpha_frozen": True,
    }
    (out / "PT2_APG_REPLICATION_PREREGISTRATION.json").write_text(json.dumps(prereg, indent=2), encoding="utf-8")

    replicate_results: List[Dict[str, Any]] = []
    for idx, fit_index in enumerate(fit_indices, start=1):
        rep_name = f"replicate_{idx:02d}_fitindex{fit_index}"
        cached_result = completed_replicate_result(out / rep_name)
        if cached_result is not None:
            log(f"{rep_name}: found complete prior replicate; reusing run_summary.json")
            restore_qk(model, qk_checkpoint)
            ensure_model_on_device(model, device)
            replicate_results.append(cached_result)
            continue
        replicate_results.append(
            run_one_replicate(
                model, device, args, out, rep_name, fit_index, calib_loader,
                codes, perms, qk_checkpoint, layers, base_hash, reference,
            )
        )

    w2_deltas = [r["metrics"]["delta_vs_saved_pt2_nll"]["wikitext2_ppl"] for r in replicate_results]
    c4_deltas = [r["metrics"]["delta_vs_saved_pt2_nll"]["c4_ppl"] for r in replicate_results]
    pairwise = []
    for i in range(len(replicate_results)):
        for j in range(i + 1, len(replicate_results)):
            a = replicate_results[i]
            b = replicate_results[j]
            aset = {tuple(x) for x in a["selected_patch"]["edit_ids"]}
            bset = {tuple(x) for x in b["selected_patch"]["edit_ids"]}
            score_a = a["score_map"]
            score_b = b["score_map"]
            pairwise.append({
                "a": a["replicate"],
                "b": b["replicate"],
                "selected_jaccard": len(aset & bset) / max(len(aset | bset), 1),
                "selected_overlap_count": len(aset & bset),
                "score_spearman_on_common_top_candidates": spearman_from_scores(score_a, score_b),
            })

    result = {
        "status": "complete",
        "run_id": args.run_id,
        "experiment": "PT2 + HBA fitting-split replication on frozen official PT2 state",
        "config": vars(args),
        "protocol": prereg,
        "pt2_state_fingerprint": state_fingerprint,
        "replicates": replicate_results,
        "replicate_table": [
            {
                "replicate": r["replicate"],
                "fit_index": r["fit_index"],
                "nominal_fit_token_offset": r["nominal_fit_token_offset"],
                "K_star": r["apg"]["selected_k"],
                "relocations": r["selected_patch"]["num_relocations"],
                "changed_coordinates": r["selected_patch"]["changed_coordinates"],
                "q_relocations": r["selected_patch"]["module_counts"]["q"],
                "k_relocations": r["selected_patch"]["module_counts"]["k"],
                "val_selected_nll": next(
                    row["validation_nll"]
                    for row in r["apg"]["curve"]
                    if row["k_per_layer"] == r["apg"]["selected_k"]
                ),
                "w2_ppl": r["metrics"]["absolute_ppl"]["wikitext2_ppl"],
                "c4_ppl": r["metrics"]["absolute_ppl"]["c4_ppl"],
                "delta_w2_nll": r["metrics"]["delta_vs_saved_pt2_nll"]["wikitext2_ppl"],
                "delta_c4_nll": r["metrics"]["delta_vs_saved_pt2_nll"]["c4_ppl"],
            }
            for r in replicate_results
        ],
        "summary_stats": {
            "w2_delta_nll": summarize(w2_deltas),
            "c4_delta_nll": summarize(c4_deltas),
            "w2_improved_count": sum(1 for x in w2_deltas if x < 0),
            "c4_improved_count": sum(1 for x in c4_deltas if x < 0),
            "both_domains_improved_count": sum(1 for a, b in zip(w2_deltas, c4_deltas) if a < 0 and b < 0),
            "num_replicates": len(replicate_results),
        },
        "ranking_stability": pairwise,
        "gate": {
            "status_complete": True,
            "state_parity_pass": bool(parity.get("pass", False)),
            "same_frozen_pt2_state": True,
            "same_selection_hash_all_replicates": all(r["data"]["validation_sample_hash"] == selection_hash for r in replicate_results),
            "fit_hashes_unique": len({r["data"]["fit_sample_hash"] for r in replicate_results}) == len(replicate_results),
            "fit_disjoint_from_selection": all(r["gate"]["fit_disjoint_from_selection"] for r in replicate_results),
            "baseline_was_not_recomputed": True,
            "all_gradients_finite": all(r["gate"]["gradient_finite"] for r in replicate_results),
            "one_backward_each": all(r["gate"]["backward_count_is_one"] for r in replicate_results),
            "no_rerank": all(r["gate"]["rerank_count_is_zero"] for r in replicate_results),
            "all_selected_patches_legal": all(r["gate"]["selected_patch_legal"] for r in replicate_results),
            "all_selected_patches_finite": all(r["gate"]["selected_patch_finite"] for r in replicate_results),
            "all_exact_relocations": all(r["gate"]["selected_patch_exact_relocations"] for r in replicate_results),
            "all_exact_changed_coordinates": all(r["gate"]["selected_patch_exact_changed_coordinates"] for r in replicate_results),
        },
        "outcome": {
            "strong_replication": all(a < 0 and b < 0 for a, b in zip(w2_deltas, c4_deltas)),
            "partial_replication": (
                sum(1 for a, b in zip(w2_deltas, c4_deltas) if a < 0 and b < 0) >= 2
                and statistics.mean(w2_deltas) < 0
                and statistics.mean(c4_deltas) < 0
            ),
            "non_replication": statistics.mean(w2_deltas) >= 0 or statistics.mean(c4_deltas) >= 0,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "timing": {
            "total_sec": time.time() - started,
            "note": "The PT2 deployment is loaded from a frozen checkpoint when --pt2-checkpoint is provided; otherwise it is rebuilt once. Replicate HBA curves are sequential because each has its own first-non-improvement stopping rule.",
        },
    }
    (out / "pt2_hba_replication_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(result["gate"], ensure_ascii=False))
    log(json.dumps(result["outcome"], ensure_ascii=False))
    log(f"wrote {out / 'pt2_hba_replication_result.json'}")


if __name__ == "__main__":
    main()
