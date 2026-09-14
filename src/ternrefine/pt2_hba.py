#!/usr/bin/env python3
"""PT2 + updated HBA on an already exported, auditable PT2 ternary state.

This runner deliberately does not re-run PT2 quantization or evaluate a new
baseline.  It reloads the existing P9-S2 detached sidecar, computes exactly
one quantized-point CE gradient, builds one fixed QGP ranking over all Llama
Q/K layers, and applies the updated HBA geometric-prefix stopping rule.
The previously recorded PT2 metrics are read from --reference-json only for
reporting deltas; they are never recomputed in this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, set_seed

from ternrefine.large_model_affine import (
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    evaluate_nll,
    get_decoder_layers,
    metric_delta,
    target_qk,
)
from ternrefine.pt2_sidecar import (
    apply_ssr_codes,
    cardinality_violations,
    detached_reload_gate,
    finite_metrics,
    load_detached_artifacts,
    official_metrics,
    restore_qk,
)


GRID = (1, 2, 4, 8, 16, 32, 64)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/Llama-2-7b-hf")
    p.add_argument("--sidecar-dir", required=True)
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
    p.add_argument("--grad-samples", type=int, default=1)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--candidate-top-k", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260907)
    return p.parse_args()


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
        x = batch.detach().cpu().contiguous()
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def state_hash(codes: Dict[int, Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for layer in sorted(codes):
        for key in sorted(codes[layer]):
            h.update(f"{layer}:{key}".encode())
            h.update(tensor_hash(codes[layer][key].T).encode())
    return h.hexdigest()


def edit_key(edit: AffineEdit) -> tuple:
    return (
        int(edit.layer), str(edit.key), int(edit.row), int(edit.block),
        int(edit.donor), int(edit.receiver), int(edit.receiver_sign),
    )


def apply_edit_list(codes: Dict[int, Dict[str, Any]], edits: Sequence[AffineEdit]) -> Dict[int, Dict[str, torch.Tensor]]:
    states = {
        layer: {key: code.T.clone() for key, code in layer_codes.items()}
        for layer, layer_codes in codes.items()
    }
    used = set()
    for edit in edits:
        donor_key = (edit.layer, edit.key, edit.row, edit.block, edit.donor)
        receiver_key = (edit.layer, edit.key, edit.row, edit.block, edit.receiver)
        if donor_key in used or receiver_key in used:
            raise RuntimeError(f"candidate collision in selected HBA prefix: {edit_key(edit)}")
        state = states[edit.layer][edit.key]
        if int(state[edit.row, edit.block, edit.donor].item()) == 0:
            raise RuntimeError(f"invalid donor state for {edit_key(edit)}")
        if int(state[edit.row, edit.block, edit.receiver].item()) != 0:
            raise RuntimeError(f"invalid receiver state for {edit_key(edit)}")
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = int(edit.receiver_sign)
        used.add(donor_key)
        used.add(receiver_key)
    return states


def select_prefix(candidates_by_layer: Dict[int, List[AffineEdit]], k: int, layers: Sequence[int]) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    for layer in layers:
        selected.extend(candidates_by_layer[layer][:k])
    return selected


def candidate_manifest(path: Path, candidates_by_layer: Dict[int, List[AffineEdit]], layers: Sequence[int]) -> str:
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        idx = 0
        for layer in layers:
            for rank, edit in enumerate(candidates_by_layer[layer]):
                row = {
                    "candidate_id": idx,
                    "layer": int(edit.layer),
                    "module": str(edit.key),
                    "rank_in_layer": int(rank),
                    "row": int(edit.row),
                    "group": int(edit.block),
                    "donor": int(edit.donor),
                    "receiver": int(edit.receiver),
                    "donor_sign": int(edit.donor_sign),
                    "receiver_sign": int(edit.receiver_sign),
                    "qgp_score": float(edit.score),
                }
                payload = json.dumps(row, sort_keys=True, separators=(",", ":"))
                handle.write(payload + "\n")
                h.update(payload.encode())
                h.update(b"\n")
                idx += 1
    return h.hexdigest()


def row_for_patch(
    model: torch.nn.Module,
    device: torch.device,
    codes: Dict[int, Dict[str, Any]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    baseline_state_hash: str,
    selected: Sequence[AffineEdit],
    val_batches: Sequence[torch.Tensor],
    k: int,
) -> Dict[str, Any]:
    states = apply_edit_list(codes, selected)
    apply_ssr_codes(model, codes, perms, states)
    val_nll = evaluate_nll(model, device, val_batches)
    audit = audit_all(codes, states)
    card = cardinality_violations(codes, states)
    return {
        "k_per_layer": int(k),
        "num_layers": len(codes),
        "num_relocations": len(selected),
        "expected_relocations": len(codes) * int(k),
        "changed_coordinates": changed_coordinates(codes, states),
        "expected_changed_coordinates": 2 * len(codes) * int(k),
        "selected_layers": sorted(int(x) for x in {e.layer for e in selected}),
        "module_counts": {
            key: sum(1 for e in selected if e.key == key)
            for key in ("q", "k")
        },
        "qgp_score_sum": float(sum(e.score for e in selected)),
        "validation_nll": float(val_nll),
        "validation_ppl": float(math.exp(val_nll)),
        "audit": audit,
        "cardinality_violations": int(card),
        "finite": bool(math.isfinite(val_nll)),
        "legal": bool(audit["total_illegal_states"] == 0 and card == 0),
        "base_state_hash": baseline_state_hash,
    }


def load_reference(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pt2 = data.get("variants", {}).get("pt2", {}).get("metrics", {})
    if not pt2:
        pt2 = data.get("instrumented_baseline_metrics", {})
    if "wikitext2_ppl" not in pt2 or "c4_ppl" not in pt2:
        raise RuntimeError("reference JSON has no saved PT2 W2/C4 metrics")
    return {
        "path": str(path),
        "run_id": data.get("run_id"),
        "wikitext2_ppl": float(pt2["wikitext2_ppl"]),
        "c4_ppl": float(pt2["c4_ppl"]),
        "wikitext2_nll": float(math.log(float(pt2["wikitext2_ppl"]))),
        "c4_nll": float(math.log(float(pt2["c4_ppl"]))),
    }


def make_full_pt2_args(args: argparse.Namespace) -> SimpleNamespace:
    """Reconstruct the official PT2 quantized deployment without eval."""
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.calib_nsamples,
        percdamp=0.01,
        blocksize=args.group_size,
        num_p=1,
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


def state_parity_gate(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, Any]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    qk_checkpoint: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """Validate the detached state without confusing BF16 assignment rounding.

    The sidecar reconstruction is checked against the saved PT2 checkpoint as
    a diagnostic.  The deployed-state gate itself restores the actual saved
    Q/K checkpoint and then checks that reload, so BF16 casting during a
    codebook reconstruction cannot masquerade as a corrupt PT2 export.
    """
    reconstruction = detached_reload_gate(model, codes, perms, qk_checkpoint)
    restore_qk(model, qk_checkpoint)
    reload_rows: List[Dict[str, Any]] = []
    for layer, layer_codes in codes.items():
        refs = target_qk(model, layer)
        for key, code in layer_codes.items():
            saved = qk_checkpoint[layer][key]
            deployed_tensor = refs[key].module.weight.detach().cpu()
            # qk_checkpoint.pt is FP16 while the evaluation model is BF16;
            # compare after the exact deployment cast, not in mixed dtypes.
            expected = saved.to(dtype=deployed_tensor.dtype).float()
            deployed = deployed_tensor.float()
            reload_rows.append({
                "layer": int(layer),
                "key": str(key),
                "saved_checkpoint_reload_max_abs": float((deployed - expected).abs().max().item()),
                "saved_checkpoint_dtype": str(saved.dtype),
                "deployed_dtype": str(deployed_tensor.dtype),
                "ssr_bijection": sorted(perms[layer][key].tolist()) == list(range(code.original_shape[1])),
                "shape_match": tuple(deployed.shape) == tuple(expected.shape),
            })
    max_reload = max(
        (float(row["saved_checkpoint_reload_max_abs"]) for row in reload_rows),
        default=float("inf"),
    )
    code_audit = audit_all(codes)
    legal_t = all(bool(torch.isin(code.T, torch.tensor([-1, 0, 1], dtype=code.T.dtype)).all().item())
                   for layer_codes in codes.values() for code in layer_codes.values())
    finite_t = all(bool(torch.isfinite(code.T.float()).all().item())
                   for layer_codes in codes.values() for code in layer_codes.values())
    all_bijection = all(bool(row["ssr_bijection"]) for row in reload_rows)
    all_shapes = all(bool(row["shape_match"]) for row in reload_rows)
    passed = (
        len(reload_rows) == 64
        and reconstruction["max_sidecar_vs_saved_q_residual"] < 1e-3
        and max_reload < 1e-6
        and legal_t
        and finite_t
        and all_bijection
        and all_shapes
        and code_audit["total_illegal_states"] == 0
    )
    return {
        "pass": bool(passed),
        "qk_module_count": len(reload_rows),
        "expected_qk_module_count": 64,
        "saved_checkpoint_loaded": True,
        "saved_checkpoint_reload_max_abs": max_reload,
        "sidecar_reconstruction_diagnostic": reconstruction,
        "sidecar_reconstruction_gate_not_used_for_deployed_reload": True,
        "code_audit": code_audit,
        "ternary_code_legal": legal_t,
        "ternary_code_finite": finite_t,
        "ssr_bijection": all_bijection,
        "shape_match": all_shapes,
        "reload_rows": reload_rows,
        "note": "The saved FP16 PT2 checkpoint is compared after the exact BF16 deployment cast; codebook reconstruction residual remains diagnostic only.",
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.group_size != 128 or args.calib_seq_len != 2048:
        raise ValueError("frozen PT2+HBA protocol requires group_size=128 and calib_seq_len=2048")
    if args.grad_samples != 1:
        raise ValueError("frozen PT2+HBA protocol requires exactly one gradient sample/batch")
    if not torch.cuda.is_available():
        raise RuntimeError("PT2+HBA requires CUDA")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    sidecar = Path(args.sidecar_dir)
    reference = load_reference(Path(args.reference_json))

    import sys
    if args.pt2_root not in sys.path:
        sys.path.insert(0, args.pt2_root)
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root
    data_mod = __import__("pt2_llm.data", fromlist=["get_loaders"])
    pt2_quantize = importlib.import_module("quantize")
    pt2_quantize.args = make_full_pt2_args(args)
    pt2_quantize.groupsize = args.group_size

    log(f"loading PT2 detached sidecar from {sidecar}")
    codes, perms, qk_checkpoint = load_detached_artifacts(sidecar)
    layers = sorted(codes)
    if len(layers) != 32 or any(sorted(layer_codes) != ["k", "q"] for layer_codes in codes.values()):
        raise RuntimeError(f"unexpected detached scope: layers={len(layers)} keys={set(k for x in codes.values() for k in x)}")
    base_hash = state_hash(codes)

    calib_loader, _ = data_mod.get_loaders(
        "wikitext2", nsamples=args.calib_nsamples, seed=0,
        seqlen=args.calib_seq_len, model=args.model
    )
    if len(calib_loader) != args.calib_nsamples:
        raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.calib_nsamples}")
    log(f"rebuilding official PT2 deployment for HBA on {torch.cuda.get_device_name(0)}; baseline evaluation skipped")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    model.seqlen = args.calib_seq_len
    pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
    model.to(device)
    model.config.use_cache = False
    model.eval()
    if len(get_decoder_layers(model)) != 32:
        raise RuntimeError(f"model decoder depth mismatch: {len(get_decoder_layers(model))}")

    parity = state_parity_gate(model, codes, perms, qk_checkpoint)
    if not parity.get("pass", False):
        result = {
            "status": "NOT_RUN_STATE_PARITY_FAILED",
            "run_id": args.run_id,
            "experiment": "PT2 + updated HBA detached-state compatibility",
            "config": vars(args),
            "state_parity": parity,
            "reference_pt2_metrics_not_recomputed": reference,
            "elapsed_sec": time.time() - started,
        }
        (out / "pt2_hba_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        log(f"state parity failed; wrote {out / 'pt2_hba_result.json'}")
        return

    if args.val_start + args.val_samples > len(calib_loader):
        raise RuntimeError("validation slice exceeds calibration loader")
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
        "note": "validation is a reserved slice of the PT2 calibration stream; W2/C4 are not used for stopping",
    }
    (out / "calibration_manifest.json").write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")

    log("collecting exactly one quantized-point CE gradient")
    grads_original = collect_grads(model, grad_batches, layers, device, args.grad_samples)
    grads_ssr = {
        layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")}
        for layer in layers
    }

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

    # HBA is intentionally sequential: the first validation non-improvement
    # is the stopping event. Candidate generation above is vectorized/one-pass;
    # evaluating all future K values in parallel would change this protocol.
    hba_rows: List[Dict[str, Any]] = []
    baseline_row = row_for_patch(model, device, codes, perms, base_hash, [], val_batches, 0)
    hba_rows.append(baseline_row)
    log(f"HBA K=0: val_nll={baseline_row['validation_nll']:.8f} edits=0 baseline")
    restore_qk(model, qk_checkpoint)
    previous_val = baseline_row["validation_nll"]
    selected_k = 0
    stop_reason = "reached_grid_end"
    for k in GRID:
        selected = select_prefix(candidates_by_layer, k, layers)
        row = row_for_patch(model, device, codes, perms, base_hash, selected, val_batches, k)
        hba_rows.append(row)
        log(f"HBA K={k}: val_nll={row['validation_nll']:.8f} edits={row['num_relocations']}")
        if not (row["validation_nll"] < previous_val):
            stop_reason = f"first_non_improvement_at_K={k}"
            break
        selected_k = k
        previous_val = row["validation_nll"]
        # Restore the exact saved PT2 deployment state before the next prefix
        # evaluation; do not introduce a BF16 codebook-reconstruction drift.
        restore_qk(model, qk_checkpoint)
    selected_row = next(row for row in hba_rows if row["k_per_layer"] == selected_k)
    selected_edits = select_prefix(candidates_by_layer, selected_k, layers)
    selected_states = apply_edit_list(codes, selected_edits)
    selected_audit = audit_all(codes, selected_states)
    selected_card = cardinality_violations(codes, selected_states)
    if not selected_row["legal"] or not selected_row["finite"]:
        raise RuntimeError("selected HBA patch failed legal/finite validation")

    log(f"evaluating selected PT2+HBA patch on official W2/C4 only; K*={selected_k}")
    apply_ssr_codes(model, codes, perms, selected_states)
    final_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.calib_seq_len)
    if not finite_metrics(final_metrics):
        raise RuntimeError(f"selected PT2+HBA metrics are nonfinite: {final_metrics}")
    final_ppl = {
        "wikitext2_ppl": float(final_metrics["wikitext2_ppl"]),
        "c4_ppl": float(final_metrics["c4_ppl"]),
    }
    final_nll = {key: float(math.log(value)) for key, value in final_ppl.items()}
    reference_nll = {
        "wikitext2_ppl": reference["wikitext2_nll"],
        "c4_ppl": reference["c4_nll"],
    }

    selected_patch = {
        "k_per_layer": int(selected_k),
        "num_relocations": len(selected_edits),
        "changed_coordinates": changed_coordinates(codes, selected_states),
        "selected_layers": sorted(int(x) for x in {e.layer for e in selected_edits}),
        "module_counts": {key: sum(1 for e in selected_edits if e.key == key) for key in ("q", "k")},
        "qgp_score_sum": float(sum(e.score for e in selected_edits)),
        "state_hash_before": base_hash,
        "state_hash_after": state_hash({layer: {key: type("State", (), {"T": selected_states[layer][key]})() for key in selected_states[layer]} for layer in selected_states}),
        "audit": selected_audit,
        "cardinality_violations": selected_card,
        "edit_ids": [edit_key(e) for e in selected_edits],
    }
    (out / "hba_curve.json").write_text(json.dumps({"grid": [0, *list(GRID)], "rows": hba_rows, "selected_k": selected_k, "stop_reason": stop_reason}, indent=2), encoding="utf-8")
    (out / "selected_patch.json").write_text(json.dumps(selected_patch, indent=2), encoding="utf-8")

    elapsed = time.time() - started
    result = {
        "status": "complete",
        "run_id": args.run_id,
        "experiment": "PT2 + updated HBA detached-state compatibility",
        "config": vars(args),
        "protocol": {
            "pt2_state_source": "existing P9-S2 detached sidecar",
            "full_pt2_deployment_rebuilt": True,
            "baseline_recomputed": False,
            "baseline_evaluation_skipped": True,
            "hba_scope": "all 32 decoder layers, Q/K only",
            "one_quantized_point_backward": True,
            "fixed_qgp_ranking": True,
            "hba_grid": [0, *list(GRID)],
            "hba_stop": "first strict validation non-improvement",
            "hba_compares_first_prefix_to_q0": True,
            "no_rerank": True,
            "no_additional_backward_after_ranking": 0,
            "mu_alpha_frozen": True,
            "group_size": 128,
            "teacher_or_qat": False,
            "validation_does_not_use_w2_c4": True,
        },
        "state_source": {
            "sidecar_dir": str(sidecar),
            "reference_json": reference,
            "state_hash_before": base_hash,
            "candidate_manifest_hash": manifest_hash,
            "qk_checkpoint_loaded": True,
            "full_pt2_deployment_rebuilt": True,
        },
        "state_parity": parity,
        "data": data_manifest,
        "candidate_summary": {
            "all_layers": layers,
            "candidate_top_k_per_layer": args.candidate_top_k,
            "total_candidates": sum(len(x) for x in candidates_by_layer.values()),
            "layer_rows": layer_summary,
        },
        "hba": {
            "baseline_row": baseline_row,
            "curve": hba_rows,
            "selected_k": int(selected_k),
            "stop_reason": stop_reason,
            "selected_patch": selected_patch,
        },
        "pt2_plus_hba_metrics": {
            "absolute_ppl": final_ppl,
            "absolute_nll": final_nll,
            "delta_vs_saved_pt2_nll": metric_delta(final_nll, reference_nll),
            "finite": finite_metrics(final_ppl),
        },
        "gate": {
            "state_parity_pass": bool(parity.get("pass", False)),
            "hba_validation_curve_finite": all(row["finite"] for row in hba_rows),
            "hba_validation_curve_legal": all(row["legal"] for row in hba_rows),
            "selected_patch_exact_relocations": len(selected_edits) == len(layers) * selected_k,
            "selected_patch_exact_changed_coordinates": changed_coordinates(codes, selected_states) == 2 * len(layers) * selected_k,
            "selected_patch_finite": finite_metrics(final_ppl),
            "selected_patch_legal": selected_audit["total_illegal_states"] == 0 and selected_card == 0,
            "pt2_baseline_was_not_recomputed": True,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "timing": {
            "total_sec": elapsed,
            "note": "HBA K evaluations are sequential by stopping definition; candidate construction is one vectorized pass.",
        },
    }
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (out / "pt2_hba_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(result["gate"], ensure_ascii=False))
    log(f"wrote {out / 'pt2_hba_result.json'}")


if __name__ == "__main__":
    main()
