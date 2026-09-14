#!/usr/bin/env python3
"""E2: Qwen3-8B PT2 + HBA fitting-split replication.

This is the Qwen analogue of the successful Llama PT2+HBA replication.  It
uses one frozen, previously exported PT2 state, runs three fitting splits, and
keeps the APG/HBA protocol fixed.  The strict absolute state-parity gate is
recorded as skipped because Qwen3 still has the known BF16-scale sidecar
residual; this runner is therefore a conditional strong-initializer result.
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
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import set_seed

from cegsp_e2_qwen_pt2_health_a100 import apply_qwen_pt2_layer_adapter
from cegsp_e2_qwen_pt2_hba_skipgate_a100 import (
    load_full_pt2_state,
    reload_clean_pt2_model,
    sidecar_diagnostic,
)
from cegsp_p7_a100_scaling import (
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    get_decoder_layers,
    metric_delta,
)
from cegsp_p9s2_detached_pt2_plugin import (
    apply_ssr_codes,
    cardinality_violations,
    finite_metrics,
    load_detached_artifacts,
    official_metrics,
    restore_qk,
)
from cegsp_pt2_hba_detached_a100 import (
    GRID,
    apply_edit_list,
    batches_hash,
    candidate_manifest,
    edit_key,
    row_for_patch,
    select_prefix,
    state_hash,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B")
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--pt2-checkpoint", required=True)
    p.add_argument("--reference-json", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/CEGSP/model/tqgsp-runs")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--calib-nsamples", type=int, default=128)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--val-start", type=int, default=1)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--fit-indices", default="0,17,33")
    p.add_argument("--grad-samples", type=int, default=1)
    p.add_argument("--candidate-top-k", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--skip-strict-parity-gate", action="store_true")
    return p.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def hash_batches(batches: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for batch in batches:
        h.update(tensor_hash(batch).encode())
    return h.hexdigest()


def load_reference(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data.get("official_metrics") or data.get("baseline_pt2_metrics", {}).get("absolute_ppl")
    if not metrics or "wikitext2_ppl" not in metrics or "c4_ppl" not in metrics:
        raise RuntimeError(f"reference JSON has no W2/C4 PT2 metrics: {path}")
    return {
        "wikitext2_ppl": float(metrics["wikitext2_ppl"]),
        "c4_ppl": float(metrics["c4_ppl"]),
        "wikitext2_nll": float(math.log(float(metrics["wikitext2_ppl"]))),
        "c4_nll": float(math.log(float(metrics["c4_ppl"]))),
    }


def completed_replicate(rep_dir: Path) -> Dict[str, Any] | None:
    path = rep_dir / "run_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    gate = data.get("gate", {})
    required = (
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
    if all(bool(gate.get(k)) for k in required):
        return data
    return None


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


def spearman(a: Dict[Tuple[Any, ...], float], b: Dict[Tuple[Any, ...], float]) -> Dict[str, Any]:
    keys = sorted(set(a).intersection(b))
    if len(keys) < 2:
        return {"common": len(keys), "rho": None}
    return {"common": len(keys), "rho": pearson(rank([a[k] for k in keys]), rank([b[k] for k in keys]))}


def summarize(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def install_cache_only_datasets_shim() -> None:
    """Bypass broken datasets/pandas imports when PT2 torch caches exist.

    The remote PT2 data module imports datasets at module load time even when
    get_loaders immediately returns the prebuilt torch cache.  This shim keeps
    the protocol cache-only: any cache miss fails loudly instead of downloading
    or rebuilding data.
    """
    if "datasets" in sys.modules:
        return
    module = types.ModuleType("datasets")

    def _cache_miss(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("datasets shim is cache-only; required PT2 torch cache is missing")

    module.load_dataset = _cache_miss
    module.load_from_disk = _cache_miss
    sys.modules["datasets"] = module


def selection_manifest(calib_loader: Sequence[Any], fit_index: int, val_start: int, val_samples: int, seq_len: int) -> Dict[str, Any]:
    fit_batches = [calib_loader[fit_index][0]]
    val_batches = [calib_loader[i][0] for i in range(val_start, val_start + val_samples)]
    val_indices = list(range(val_start, val_start + val_samples))
    return {
        "dataset": "wikitext2 via official PT2 loader",
        "calib_nsamples": len(calib_loader),
        "fit_index": int(fit_index),
        "nominal_fit_token_offset": int(fit_index * seq_len),
        "gradient_batches": 1,
        "validation_start": int(val_start),
        "validation_samples": int(val_samples),
        "validation_indices": val_indices,
        "seq_len": int(seq_len),
        "fit_sample_hash": hash_batches(fit_batches),
        "validation_sample_hash": hash_batches(val_batches),
        "fit_disjoint_from_selection": fit_index not in set(val_indices),
        "selection_uses_w2_c4": False,
    }


def run_replicate(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    out: Path,
    name: str,
    fit_index: int,
    calib_loader: Sequence[Any],
    codes: Dict[int, Dict[str, Any]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    qk_checkpoint: Dict[int, Dict[str, torch.Tensor]],
    layers: Sequence[int],
    base_hash: str,
    reference: Dict[str, float],
) -> Dict[str, Any]:
    rep_dir = out / name
    rep_dir.mkdir(parents=True, exist_ok=True)
    cached = completed_replicate(rep_dir)
    if cached is not None:
        log(f"{name}: reusing completed run_summary.json")
        return cached

    restore_qk(model, qk_checkpoint)
    model.to(device)
    model.config.use_cache = False
    model.eval()
    model.zero_grad(set_to_none=True)

    manifest = selection_manifest(calib_loader, fit_index, args.val_start, args.val_samples, args.calib_seq_len)
    if not manifest["fit_disjoint_from_selection"]:
        raise RuntimeError(f"{name}: fit index overlaps selection split")
    write_json(rep_dir / "fit_manifest.json", manifest)
    write_json(rep_dir / "selection_manifest.json", {
        "validation_indices": manifest["validation_indices"],
        "validation_sample_hash": manifest["validation_sample_hash"],
    })

    fit_batches = [calib_loader[fit_index][0]]
    val_batches = [calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)]

    log(f"{name}: one CE backward at fit_index={fit_index}")
    grad_start = time.time()
    grads_original = collect_grads(model, fit_batches, layers, device, args.grad_samples)
    grads_ssr = {layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")} for layer in layers}
    grad_audit = {
        "backward_count": 1,
        "elapsed_sec": time.time() - grad_start,
        "all_gradient_tensors_finite": all(
            bool(torch.isfinite(grads_ssr[layer][key]).all().item())
            for layer in layers
            for key in ("q", "k")
        ),
        "gradient_tensor_count": sum(len(grads_ssr[layer]) for layer in layers),
    }
    write_json(rep_dir / "gradient_audit.json", grad_audit)

    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    score_map: Dict[Tuple[Any, ...], float] = {}
    layer_rows: List[Dict[str, Any]] = []
    for layer in layers:
        candidates = build_top_candidates(codes, grads_ssr, layer, args.candidate_top_k)
        if len(candidates) < max(GRID):
            raise RuntimeError(f"{name}: layer {layer} has only {len(candidates)} candidates")
        candidates_by_layer[layer] = candidates
        for edit in candidates:
            score_map[edit_key(edit)] = float(edit.score)
        layer_rows.append({
            "layer": int(layer),
            "candidate_count": len(candidates),
            "top_score": float(candidates[0].score),
            "top8_score_sum": float(sum(e.score for e in candidates[:8])),
        })
    layer_rows.sort(key=lambda row: (-row["top8_score_sum"], row["layer"]))
    candidate_hash = candidate_manifest(rep_dir / "candidate_manifest.jsonl", candidates_by_layer, layers)
    ranking_manifest = {
        "ranking": "one fixed QGP ranking for this replicate",
        "rerank_count": 0,
        "candidate_top_k_per_layer": args.candidate_top_k,
        "total_candidates": sum(len(v) for v in candidates_by_layer.values()),
        "candidate_manifest_hash": candidate_hash,
        "layer_rows": layer_rows,
    }
    write_json(rep_dir / "ranking_manifest.json", ranking_manifest)

    hba_rows: List[Dict[str, Any]] = []
    selected_k = None
    prev_val = None
    stop_reason = "reached_grid_end"
    for k in GRID:
        selected = select_prefix(candidates_by_layer, k, layers)
        row = row_for_patch(model, device, codes, perms, base_hash, selected, val_batches, k)
        hba_rows.append(row)
        write_json(rep_dir / "apg_curve_partial.json", {
            "grid": list(GRID),
            "completed_rows": hba_rows,
            "latest_k": k,
            "selected_k_so_far": selected_k,
            "stop_reason_so_far": stop_reason,
        })
        log(f"{name}: HBA K={k} val_nll={row['validation_nll']:.8f} edits={row['num_relocations']}")
        if prev_val is not None and not (row["validation_nll"] < prev_val):
            stop_reason = f"first_non_improvement_at_K={k}"
            break
        selected_k = k
        prev_val = row["validation_nll"]
        restore_qk(model, qk_checkpoint)
        model.to(device)
        model.eval()

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
        "selected_layers": list(layers),
        "module_counts": {key: sum(1 for e in selected_edits if e.key == key) for key in ("q", "k")},
        "qgp_score_sum": float(sum(e.score for e in selected_edits)),
        "audit": selected_audit,
        "cardinality_violations": int(selected_card),
        "state_hash_before": base_hash,
        "edit_ids": [edit_key(e) for e in selected_edits],
    }
    selected_payload = json.dumps(selected_patch, sort_keys=True, separators=(",", ":"))
    write_json(rep_dir / "selected_patch_pre_eval.json", selected_patch)
    (rep_dir / "selected_patch_pre_eval_hash.txt").write_text(hashlib.sha256(selected_payload.encode()).hexdigest() + "\n")
    write_json(rep_dir / "apg_curve.json", {"grid": list(GRID), "rows": hba_rows, "selected_k": selected_k, "stop_reason": stop_reason})

    log(f"{name}: evaluating selected patch on W2/C4; K*={selected_k}")
    apply_ssr_codes(model, codes, perms, selected_states)
    final_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.calib_seq_len)
    if not finite_metrics(final_metrics):
        raise RuntimeError(f"{name}: nonfinite final metrics {final_metrics}")
    final_ppl = {
        "wikitext2_ppl": float(final_metrics["wikitext2_ppl"]),
        "c4_ppl": float(final_metrics["c4_ppl"]),
    }
    final_nll = {k: float(math.log(v)) for k, v in final_ppl.items()}
    ref_nll = {"wikitext2_ppl": reference["wikitext2_nll"], "c4_ppl": reference["c4_nll"]}
    delta_nll = metric_delta(final_nll, ref_nll)

    write_json(rep_dir / "selected_patch.json", selected_patch)
    (rep_dir / "selected_patch_hash.txt").write_text(hashlib.sha256(selected_payload.encode()).hexdigest() + "\n")
    result = {
        "replicate": name,
        "fit_index": int(fit_index),
        "nominal_fit_token_offset": int(fit_index * args.calib_seq_len),
        "data": manifest,
        "gradient_audit": grad_audit,
        "ranking_manifest": ranking_manifest,
        "apg": {"curve": hba_rows, "selected_k": int(selected_k), "stop_reason": stop_reason},
        "selected_patch": selected_patch,
        "metrics": {
            "absolute_ppl": final_ppl,
            "absolute_nll": final_nll,
            "delta_vs_saved_pt2_nll": delta_nll,
        },
        "mechanism": {
            "p_beneficial_strong": None,
            "rho_qgp_actual": None,
            "note": "Candidate-level beneficial density is not recomputed here; use the paired E2 density diagnostic for p_beneficial/rho if required.",
        },
        "gate": {
            "fit_disjoint_from_selection": bool(manifest["fit_disjoint_from_selection"]),
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
    write_json(rep_dir / "run_summary.json", result)
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    if not args.skip_strict_parity_gate:
        raise RuntimeError("Qwen E2 replication must explicitly mark --skip-strict-parity-gate.")
    if args.group_size != 128 or args.calib_seq_len != 2048:
        raise ValueError("frozen protocol requires group_size=128 and seq_len=2048")
    if args.grad_samples != 1:
        raise ValueError("frozen protocol requires exactly one backward per replicate")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "e2_qwen_pt2_hba_replication_partial.json", {"status": "started", "run_id": args.run_id, "config": vars(args)})

    if args.pt2_root not in sys.path:
        sys.path.insert(0, args.pt2_root)
    tool_dir = str(Path(__file__).resolve().parent)
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root
    install_cache_only_datasets_shim()
    data_mod = importlib.import_module("pt2_llm.data")
    if hasattr(data_mod, "DATA_ROOT"):
        data_mod.DATA_ROOT = args.pt2_data_root
    pt2_quantize = importlib.import_module("quantize")

    fit_indices = parse_indices(args.fit_indices)
    if args.val_start + args.val_samples > args.calib_nsamples:
        raise RuntimeError("selection slice exceeds calibration loader")
    if any(i < 0 or i >= args.calib_nsamples for i in fit_indices):
        raise RuntimeError(f"fit index outside calibration loader: {fit_indices}")
    if any(args.val_start <= i < args.val_start + args.val_samples for i in fit_indices):
        raise RuntimeError("fit indices must be disjoint from selection slice")

    log(f"loading Qwen sidecar {args.sidecar_dir}")
    codes, perms, qk_checkpoint = load_detached_artifacts(Path(args.sidecar_dir))
    layers = sorted(codes)
    diag = sidecar_diagnostic(codes, perms)
    if not (diag["layer_keys_complete"] and diag["ternary_code_legal"] and diag["ternary_code_finite"] and diag["ssr_bijection"]):
        raise RuntimeError(f"sidecar diagnostic failed: {diag}")
    if len(layers) != 36 or sum(len(v) for v in codes.values()) != 72:
        raise RuntimeError(f"unexpected Qwen Q/K scope: layers={len(layers)} qk={sum(len(v) for v in codes.values())}")
    base_hash = state_hash(codes)
    reference = load_reference(Path(args.reference_json))

    log(f"loading PT2 calibration stream nsamples={args.calib_nsamples}")
    calib_loader, _ = data_mod.get_loaders("wikitext2", nsamples=args.calib_nsamples, seed=0, seqlen=args.calib_seq_len, model=args.model)
    if len(calib_loader) != args.calib_nsamples:
        raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.calib_nsamples}")

    log(f"loading frozen compact PT2 checkpoint {args.pt2_checkpoint}")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    adapter = apply_qwen_pt2_layer_adapter(model, args.model)
    model.seqlen = args.calib_seq_len
    model.to(device)
    checkpoint_info = load_full_pt2_state(model, Path(args.pt2_checkpoint))
    model.config.use_cache = False
    model.eval()
    if len(get_decoder_layers(model)) != len(layers):
        raise RuntimeError(f"decoder depth mismatch model={len(get_decoder_layers(model))} sidecar={len(layers)}")
    restore_qk(model, qk_checkpoint)
    write_json(out / "pt2_state_fingerprint.json", {
        "checkpoint": checkpoint_info,
        "sidecar_dir": str(args.sidecar_dir),
        "state_hash": base_hash,
        "state_diagnostic": diag,
        "strict_state_parity_gate": "skipped_by_user_request_after_known_qwen_bf16_residual",
        "reference_pt2_metrics": reference,
    })

    selection_hash = hash_batches([calib_loader[i][0] for i in range(args.val_start, args.val_start + args.val_samples)])
    prereg = {
        "experiment": "E2 Strong-PTQ breadth: Qwen3-8B PT2 + HBA fitting-split replication",
        "model": "Qwen3-8B",
        "initializer": "official PT2 checkpoint, strict parity skipped conditional run",
        "fit_indices": fit_indices,
        "nominal_fit_token_offsets": [i * args.calib_seq_len for i in fit_indices],
        "selection_indices": list(range(args.val_start, args.val_start + args.val_samples)),
        "selection_hash": selection_hash,
        "apg_grid": list(GRID),
        "stopping": "first strict validation non-improvement",
        "scope": "36 decoder layers, Q/K only",
        "baseline_recomputed": False,
        "one_backward_per_replicate": True,
        "fixed_qgp_ranking": True,
        "no_rerank": True,
        "mu_alpha_frozen": True,
        "w2_c4_not_used_for_selection": True,
    }
    write_json(out / "E2_QWEN_PT2_HBA_REPLICATION_PREREGISTRATION.json", prereg)

    replicate_results: List[Dict[str, Any]] = []
    for idx, fit_index in enumerate(fit_indices, start=1):
        rep_name = f"replicate_{idx:02d}_fitindex{fit_index}"
        replicate_results.append(
            run_replicate(model, device, args, out, rep_name, fit_index, calib_loader, codes, perms, qk_checkpoint, layers, base_hash, reference)
        )
        if idx < len(fit_indices):
            log("reloading clean PT2 model for next replicate after official evaluator")
            del model
            torch.cuda.empty_cache()
            model, _ = reload_clean_pt2_model(pt2_quantize, args.model, args.calib_seq_len, Path(args.pt2_checkpoint), device)
            restore_qk(model, qk_checkpoint)

    w2 = [r["metrics"]["delta_vs_saved_pt2_nll"]["wikitext2_ppl"] for r in replicate_results]
    c4 = [r["metrics"]["delta_vs_saved_pt2_nll"]["c4_ppl"] for r in replicate_results]
    pairwise = []
    for i in range(len(replicate_results)):
        for j in range(i + 1, len(replicate_results)):
            a = replicate_results[i]
            b = replicate_results[j]
            aset = {tuple(x) for x in a["selected_patch"]["edit_ids"]}
            bset = {tuple(x) for x in b["selected_patch"]["edit_ids"]}
            score_a = {eval(k): v for k, v in a["score_map"].items()}
            score_b = {eval(k): v for k, v in b["score_map"].items()}
            pairwise.append({
                "a": a["replicate"],
                "b": b["replicate"],
                "selected_overlap_count": len(aset & bset),
                "selected_jaccard": len(aset & bset) / max(len(aset | bset), 1),
                "score_spearman_on_common_top_candidates": spearman(score_a, score_b),
            })

    table = [
        {
            "replicate": r["replicate"],
            "fit_index": r["fit_index"],
            "nominal_fit_token_offset": r["nominal_fit_token_offset"],
            "K_star": r["apg"]["selected_k"],
            "relocations": r["selected_patch"]["num_relocations"],
            "changed_coordinates": r["selected_patch"]["changed_coordinates"],
            "delta_w2_nll": r["metrics"]["delta_vs_saved_pt2_nll"]["wikitext2_ppl"],
            "delta_c4_nll": r["metrics"]["delta_vs_saved_pt2_nll"]["c4_ppl"],
            "w2_ppl": r["metrics"]["absolute_ppl"]["wikitext2_ppl"],
            "c4_ppl": r["metrics"]["absolute_ppl"]["c4_ppl"],
        }
        for r in replicate_results
    ]
    both = sum(1 for a, b in zip(w2, c4) if a < 0 and b < 0)
    result = {
        "status": "complete",
        "run_id": args.run_id,
        "experiment": "E2 Strong-PTQ breadth: Qwen3-8B PT2 + HBA fitting-split replication",
        "config": vars(args),
        "protocol": prereg,
        "conditionality": {
            "strict_qwen_state_parity_gate": "skipped",
            "reason": "requested by user after direct PT2 baseline health passed; known Qwen BF16-scale sidecar residual remains a caveat",
            "not_a_strict_E2_health_pass": True,
        },
        "state": {
            "checkpoint": checkpoint_info,
            "architecture_adapter": adapter,
            "sidecar_diagnostic": diag,
            "state_hash": base_hash,
        },
        "baseline_pt2_metrics": {
            "absolute_ppl": {"wikitext2_ppl": reference["wikitext2_ppl"], "c4_ppl": reference["c4_ppl"]},
            "absolute_nll": {"wikitext2_ppl": reference["wikitext2_nll"], "c4_ppl": reference["c4_nll"]},
            "baseline_recomputed": False,
        },
        "replicates": replicate_results,
        "replicate_table": table,
        "summary_stats": {
            "w2_delta_nll": summarize(w2),
            "c4_delta_nll": summarize(c4),
            "w2_improved_count": sum(1 for x in w2 if x < 0),
            "c4_improved_count": sum(1 for x in c4 if x < 0),
            "both_domains_improved_count": both,
            "num_replicates": len(replicate_results),
        },
        "ranking_stability": pairwise,
        "gate": {
            "status_complete": True,
            "conditional_strict_parity_skipped": True,
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
            "strong_success": both == len(replicate_results),
            "acceptable_success": both >= 2 and statistics.mean(w2) < 0 and statistics.mean(c4) < 0,
            "danger_result": both == 0 or statistics.mean(w2) >= 0 or statistics.mean(c4) >= 0,
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
    write_json(out / "e2_qwen_pt2_hba_replication_result.json", result)
    log(json.dumps(result["gate"], ensure_ascii=False))
    log(json.dumps(result["outcome"], ensure_ascii=False))
    log(f"wrote {out / 'e2_qwen_pt2_hba_replication_result.json'}")


if __name__ == "__main__":
    main()
