#!/usr/bin/env python3
"""P11/P12: matched-budget allocation and signal-isolation experiments.

This is a frozen, PTQ-only experiment for the TernRefine paper.  It starts from an
ordinary affine ternary state, computes one CE gradient on the fit split, and
compares legal active-to-zero/zero-to-sign relocations under a fixed budget.
There is no QAT teacher, latent weight, optimizer update, or evaluation-driven
selection.

P11 changes only model-level allocation (HBA, flat-global, uniform and fixed
random-layer controls).  P12-A fixes the HBA layer allocation and changes only
the ranking signal from quantized-point CE to weight reconstruction error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ternrefine.ce_gradient import collect_ce_qk_grads
from ternrefine.affine_adapter import (
    AffineCode,
    AffineEdit,
    apply_affine_patch,
    audit_all,
    build_group_candidates,
    cardinality_violations,
    eval_metrics,
    make_affine_code,
    restore_qk,
    select_unique_edits,
    snapshot_qk,
    with_ppl,
)
from ternrefine.data_eval import build_wikitext_splits, log, parse_csv_ints


LAYERS = list(range(24))
PRIMARY_RANDOM_SEEDS = (20260901, 20260902, 20260903)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
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
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--edits-total", type=int, default=384)
    p.add_argument("--hba-layers", type=int, default=6)
    p.add_argument("--edits-per-hba-layer", type=int, default=64)
    p.add_argument("--uniform-edits-per-layer", type=int, default=16)
    p.add_argument("--layer-probe-edits", type=int, default=8)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260901)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def finite_metrics(metrics: Dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def metric_delta(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {key: float(metrics[key] - baseline[key]) for key in metrics}


def changed_coordinates(
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]],
) -> int:
    return sum(
        int((states[layer][key] != code.T).sum().item())
        for layer, layer_codes in codes.items()
        for key, code in layer_codes.items()
    )


def per_layer_counts(edits: Sequence[AffineEdit]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edit in edits:
        counts[str(edit.layer)] = counts.get(str(edit.layer), 0) + 1
    return counts


def edit_id(edit: AffineEdit) -> Tuple[int, str, int, int, int, int]:
    return (
        int(edit.layer),
        str(edit.key),
        int(edit.row),
        int(edit.block),
        int(edit.donor),
        int(edit.receiver),
    )


def edit_record(edit: AffineEdit, reconstruction_score: float) -> Dict[str, object]:
    return {
        "layer": int(edit.layer),
        "key": str(edit.key),
        "row": int(edit.row),
        "block": int(edit.block),
        "donor": int(edit.donor),
        "receiver": int(edit.receiver),
        "donor_sign": int(edit.donor_sign),
        "receiver_sign": int(edit.receiver_sign),
        "task_score": float(edit.score_formula),
        "task_score_exact": float(edit.score_exact),
        "reconstruction_score": float(reconstruction_score),
    }


def select_layerwise(
    candidates_by_layer: Dict[int, List[AffineEdit]],
    selected_layers: Sequence[int],
    edits_per_layer: int,
    score_by_id: Dict[Tuple[int, str, int, int, int, int], float] | None = None,
) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    for layer in selected_layers:
        candidates = list(candidates_by_layer[layer])
        if score_by_id is not None:
            candidates.sort(key=lambda e: (-score_by_id[edit_id(e)], edit_id(e)))
        picked = select_unique_edits(candidates, edits_per_layer)
        if len(picked) != edits_per_layer:
            raise RuntimeError(
                f"cannot meet per-layer budget: layer={layer} "
                f"requested={edits_per_layer} got={len(picked)}"
            )
        selected.extend(picked)
    return selected


def apply_edits(
    codes: Dict[int, Dict[str, AffineCode]], edits: Sequence[AffineEdit]
) -> Dict[int, Dict[str, torch.Tensor]]:
    states = {
        layer: {key: code.T.clone() for key, code in layer_codes.items()}
        for layer, layer_codes in codes.items()
    }
    for edit in edits:
        state = states[edit.layer][edit.key]
        if int(state[edit.row, edit.block, edit.donor].item()) == 0:
            raise RuntimeError(f"donor is not active for edit {edit_id(edit)}")
        if int(state[edit.row, edit.block, edit.receiver].item()) != 0:
            raise RuntimeError(f"receiver is not zero for edit {edit_id(edit)}")
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = int(edit.receiver_sign)
    return states


def reconstruction_score(code: AffineCode, edit: AffineEdit) -> float:
    """Exact FP-weight MSE decrease for one legal affine relocation."""
    mu_d = float(code.mu[edit.row, edit.block, 0].item())
    alpha_d = float(code.alpha[edit.row, edit.block, 0].item())
    fp_d = float(code.fp_padded[edit.row, edit.block, edit.donor].item())
    fp_r = float(code.fp_padded[edit.row, edit.block, edit.receiver].item())
    q_d_before = mu_d + alpha_d * int(edit.donor_sign)
    q_r_before = mu_d
    q_d_after = mu_d
    q_r_after = mu_d + alpha_d * int(edit.receiver_sign)
    before = (q_d_before - fp_d) ** 2 + (q_r_before - fp_r) ** 2
    after = (q_d_after - fp_d) ** 2 + (q_r_after - fp_r) ** 2
    return float(before - after)


def reconstruction_stats(
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]],
) -> Dict[str, float]:
    total_sq = 0.0
    total_fp_sq = 0.0
    count = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            state = states[layer][key]
            q = code.mu + code.alpha * state.float()
            diff = (q - code.fp_padded) * code.valid.float()
            total_sq += float((diff * diff).sum().item())
            total_fp_sq += float(((code.fp_padded * code.valid.float()) ** 2).sum().item())
            count += int(code.valid.sum().item())
    return {
        "qk_reconstruction_mse": total_sq / max(count, 1),
        "qk_reconstruction_nmse": total_sq / max(total_fp_sq, 1e-12),
        "qk_reconstruction_sq_error": total_sq,
        "valid_coordinates": float(count),
    }


def rank_correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=np.float64)
    y = np.asarray(list(ys), dtype=np.float64)
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def build_c4_cached_batches(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    token_offset: int,
) -> List[torch.Tensor]:
    """Read the already cached C4 validation Arrow directly.

    The remote image currently has a broken pandas import inside ``datasets``;
    using the cached Arrow keeps the pre-registered C4 split unchanged and
    avoids installing or upgrading any package during the experiment.
    """
    if n_batches <= 0:
        return []
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as exc:
        raise RuntimeError(f"pyarrow is required for cached C4: {exc}") from exc

    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets"))
    )
    paths = sorted(cache_root.glob("allenai___c4/**/c4-validation.arrow"))
    if not paths:
        raise FileNotFoundError(f"C4 validation Arrow not found under {cache_root}")
    needed = token_offset + n_batches * batch_size * (seq_len + 1)
    token_chunks: List[torch.Tensor] = []
    total = 0
    for path in paths:
        with pa.memory_map(str(path), "r") as source:
            try:
                reader = ipc.open_file(source)
                batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
            except Exception:
                source.seek(0)
                reader = ipc.open_stream(source)
                batches = iter(reader)
            for record_batch in batches:
                texts = record_batch.column("text").to_pylist()
                for text in texts:
                    if not isinstance(text, str) or not text.strip():
                        continue
                    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    if ids.numel() == 0:
                        continue
                    token_chunks.append(ids.cpu())
                    total += int(ids.numel())
                    if total >= needed:
                        break
                if total >= needed:
                    break
        if total >= needed:
            break
    if total < needed:
        raise RuntimeError(f"not enough cached C4 tokens: have={total} need={needed}")
    ids = torch.cat(token_chunks, dim=0)[token_offset:needed]
    return [x.clone() for x in ids.view(n_batches, batch_size, seq_len + 1)]


def main() -> None:
    args = parse_args()
    started = time.time()
    timing: Dict[str, float] = {}
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("this mechanism runner requires one CUDA device")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device available")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    layers = parse_csv_ints(args.layers)
    if layers != LAYERS:
        raise ValueError("P11/P12 frozen protocol requires all OPT-350M layers 0--23")
    if args.group_size != 128 or args.edits_total != 384:
        raise ValueError("P11/P12 frozen protocol requires group_size=128 and edits_total=384")
    if args.hba_layers != 6 or args.edits_per_hba_layer != 64 or args.uniform_edits_per_layer != 16:
        raise ValueError("P11/P12 frozen allocation is HBA=6x64 and Uniform=24x16")

    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"P11/P12 loading {args.model} dtype={args.dtype} "
        f"gpu={torch.cuda.get_device_name(0)} layers=24"
    )

    t0 = time.time()
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
    c4 = build_c4_cached_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )
    timing["data_sec"] = time.time() - t0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    timing["model_load_sec"] = time.time() - t0

    t0 = time.time()
    fp_qk = snapshot_qk(model, layers)
    fp_metrics = eval_metrics(model, device, val, w2, c4)
    codes: Dict[int, Dict[str, AffineCode]] = {
        layer: {
            key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor)
            for key in ("q", "k")
        }
        for layer in layers
    }
    baseline_audit = audit_all(codes)
    apply_affine_patch(model, codes)
    affine_metrics = eval_metrics(model, device, val, w2, c4)
    affine_states = {layer: {key: code.T.clone() for key, code in cs.items()} for layer, cs in codes.items()}
    affine_reconstruction = reconstruction_stats(codes, affine_states)
    timing["baseline_eval_and_code_sec"] = time.time() - t0
    log(f"affine baseline metrics={with_ppl(affine_metrics)}")

    t0 = time.time()
    grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    timing["ce_gradient_sec"] = time.time() - t0

    t0 = time.time()
    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_ranking_rows: List[Dict[str, object]] = []
    reconstruction_by_id: Dict[Tuple[int, str, int, int, int, int], float] = {}
    all_candidates: List[AffineEdit] = []
    for layer in layers:
        layer_candidates: List[AffineEdit] = []
        for key in ("q", "k"):
            layer_candidates.extend(
                build_group_candidates(layer, key, codes[layer][key], grads[layer][key], "affine_fp")
            )
        layer_candidates.sort(key=lambda e: (-e.score_formula, edit_id(e)))
        candidates_by_layer[layer] = layer_candidates
        all_candidates.extend(layer_candidates)
        for edit in layer_candidates:
            reconstruction_by_id[edit_id(edit)] = reconstruction_score(codes[layer][edit.key], edit)
        probe = layer_candidates[: args.layer_probe_edits]
        layer_ranking_rows.append(
            {
                "layer": int(layer),
                "num_candidates": len(layer_candidates),
                "probe_edits": len(probe),
                "layer_task_score_top_probe_sum": float(sum(e.score_formula for e in probe)),
                "layer_reconstruction_score_top_probe_sum": float(
                    sum(reconstruction_by_id[edit_id(e)] for e in probe)
                ),
                "top_task_score": float(probe[0].score_formula) if probe else float("nan"),
                "top_reconstruction_score": (
                    float(reconstruction_by_id[edit_id(probe[0])]) if probe else float("nan")
                ),
            }
        )
    layer_ranking_rows.sort(
        key=lambda row: (-float(row["layer_task_score_top_probe_sum"]), int(row["layer"]))
    )
    all_candidates.sort(key=lambda e: (-e.score_formula, edit_id(e)))
    task_rec_rank_corr = rank_correlation(
        (e.score_formula for e in all_candidates),
        (reconstruction_by_id[edit_id(e)] for e in all_candidates),
    )
    timing["candidate_build_sec"] = time.time() - t0
    log(f"candidate pool={len(all_candidates)} task_vs_reconstruction_rank_corr={task_rec_rank_corr}")

    hba_layers = [int(row["layer"]) for row in layer_ranking_rows[: args.hba_layers]]
    hba_task = select_layerwise(candidates_by_layer, hba_layers, args.edits_per_hba_layer)
    flat_task = select_unique_edits(all_candidates, args.edits_total)
    uniform = select_layerwise(candidates_by_layer, layers, args.uniform_edits_per_layer)
    if len(hba_task) != args.edits_total or len(flat_task) != args.edits_total or len(uniform) != args.edits_total:
        raise RuntimeError("one P11 selection failed to meet the exact 384-edit budget")

    random_layer_selections: Dict[str, Dict[str, object]] = {}
    random_layer_task: Dict[str, List[AffineEdit]] = {}
    for seed in PRIMARY_RANDOM_SEEDS:
        rng = random.Random(seed)
        selected = sorted(rng.sample(layers, args.hba_layers))
        edits = select_layerwise(candidates_by_layer, selected, args.edits_per_hba_layer)
        name = f"random_layer_qgp_seed{seed}"
        random_layer_selections[name] = {"seed": seed, "selected_layers": selected}
        random_layer_task[name] = edits

    reconstruction_hba = select_layerwise(
        candidates_by_layer,
        hba_layers,
        args.edits_per_hba_layer,
        reconstruction_by_id,
    )
    reconstruction_global_candidates = sorted(
        all_candidates,
        key=lambda e: (-reconstruction_by_id[edit_id(e)], edit_id(e)),
    )
    reconstruction_global = select_unique_edits(reconstruction_global_candidates, args.edits_total)
    if len(reconstruction_hba) != args.edits_total or len(reconstruction_global) != args.edits_total:
        raise RuntimeError("one P12 selection failed to meet the exact 384-edit budget")

    selections: Dict[str, List[AffineEdit]] = {
        "hba_task": hba_task,
        "flat_global_task": flat_task,
        "uniform_task": uniform,
        "reconstruction_hba": reconstruction_hba,
        "reconstruction_global": reconstruction_global,
        **random_layer_task,
    }

    variants: Dict[str, Dict[str, object]] = {}
    eval_cache: Dict[Tuple[Tuple[int, str, int, int, int, int], ...], Dict[str, object]] = {}
    t0 = time.time()
    for name, edits in selections.items():
        signature = tuple(sorted(edit_id(e) for e in edits))
        if signature in eval_cache:
            variants[name] = eval_cache[signature]
            continue
        states = apply_edits(codes, edits)
        apply_affine_patch(model, codes, states)
        metrics = eval_metrics(model, device, val, w2, c4)
        audit = audit_all(codes, states)
        rec_stats = reconstruction_stats(codes, states)
        task_score_sum = float(sum(e.score_formula for e in edits))
        rec_score_sum = float(sum(reconstruction_by_id[edit_id(e)] for e in edits))
        row: Dict[str, object] = {
            "selected_layers": sorted({int(e.layer) for e in edits}),
            "num_edits": len(edits),
            "changed_coordinates": changed_coordinates(codes, states),
            "edits_per_layer": per_layer_counts(edits),
            "metrics": with_ppl(metrics),
            "delta_vs_affine_nll": metric_delta(metrics, affine_metrics),
            "audit": audit,
            "cardinality_violations": cardinality_violations(codes, states),
            "task_score_sum": task_score_sum,
            "reconstruction_score_sum": rec_score_sum,
            "score_identity_max_abs_error": max(
                (abs(e.score_formula - e.score_exact) for e in edits), default=0.0
            ),
            "reconstruction": rec_stats,
            "reconstruction_delta_vs_affine": {
                key: float(rec_stats[key] - affine_reconstruction[key])
                for key in ("qk_reconstruction_mse", "qk_reconstruction_nmse")
            },
            "selected_edits": [
                edit_record(e, reconstruction_by_id[edit_id(e)]) for e in edits
            ],
        }
        variants[name] = row
        eval_cache[signature] = row
        apply_affine_patch(model, codes)
        log(f"evaluated {name}: delta={row['delta_vs_affine_nll']}")
    timing["variant_eval_sec"] = time.time() - t0

    def nll(name: str, split: str) -> float:
        return float(variants[name]["metrics"][split]["nll"])

    p11_hba = variants["hba_task"]
    p11_uniform = variants["uniform_task"]
    p11_flat = variants["flat_global_task"]
    hba_better_uniform_w2 = nll("hba_task", "wikitext2_untouched") < nll("uniform_task", "wikitext2_untouched")
    hba_better_uniform_c4 = nll("hba_task", "c4_untouched") < nll("uniform_task", "c4_untouched")
    hba_not_worse_flat_w2 = nll("hba_task", "wikitext2_untouched") <= nll("flat_global_task", "wikitext2_untouched")
    hba_not_worse_flat_c4 = nll("hba_task", "c4_untouched") <= nll("flat_global_task", "c4_untouched")
    if hba_better_uniform_w2 and hba_better_uniform_c4 and hba_not_worse_flat_w2 and hba_not_worse_flat_c4:
        p11_classification = "STRONG_PASS"
    elif hba_better_uniform_w2 and hba_better_uniform_c4:
        p11_classification = "PASS_BOUNDED"
    else:
        p11_classification = "NEGATIVE"

    legality_pass = baseline_audit["total_illegal_states"] == 0
    finite_pass = finite_metrics(fp_metrics) and finite_metrics(affine_metrics)
    for row in variants.values():
        legality_pass = legality_pass and (
            row["audit"]["total_illegal_states"] == 0 and row["cardinality_violations"] == 0
        )
        finite_pass = finite_pass and finite_metrics(
            {key: value["nll"] for key, value in row["metrics"].items()}
        )

    restore_qk(model, fp_qk)
    elapsed = time.time() - started
    result = {
        "run_id": args.run_id,
        "experiment": "TernRefine P11/P12 matched-budget allocation and signal isolation",
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "ternary_codebook": "Q=mu+alpha*T, T in {-1,0,+1}",
            "scope": "all 24 OPT-350M Q/K projection modules",
            "group_size": 128,
            "threshold_factor": 0.75,
            "total_relocations": 384,
            "changed_coordinates": 768,
            "mu_alpha_refit": False,
            "selection_fit_only": True,
            "validation_or_untouched_selection": False,
            "qat_teacher": False,
            "qat_checkpoint_or_latent_weight": False,
            "optimizer_update": False,
            "action_space": "same-group active-to-zero plus zero-to-sign, fixed cardinality",
            "hba_rule": "top 6 layers by sum of top 8 task-score candidates; 64 per layer",
            "flat_rule": "global task-score ranking with select_unique_edits",
            "uniform_rule": "all 24 layers x 16 task-score candidates",
            "random_rule": "three fixed random 6-layer selections, QGP top 64 per layer",
            "p12_rule": "same HBA layers and 64/layer; replace task score with exact FP-weight MSE decrease",
        },
        "data": {
            "wikitext_source": wikitext_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "wikitext2_untouched_batches": len(w2),
            "c4_untouched_batches": len(c4),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "fit_token_offset": args.fit_token_offset,
            "val_token_offset": args.val_token_offset,
            "c4_token_offset": args.c4_token_offset,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "fp_metrics": with_ppl(fp_metrics),
        "affine_baseline_metrics": with_ppl(affine_metrics),
        "affine_baseline_audit": baseline_audit,
        "affine_baseline_reconstruction": affine_reconstruction,
        "candidate_summary": {
            "total_candidates": len(all_candidates),
            "task_vs_reconstruction_rank_correlation": task_rec_rank_corr,
            "layer_ranking_by_task": layer_ranking_rows,
        },
        "p11": {
            "hba_layers": hba_layers,
            "variants": {
                name: variants[name]
                for name in ["hba_task", "flat_global_task", "uniform_task", *random_layer_task.keys()]
            },
            "gate": {
                "legality_pass": legality_pass,
                "finite_pass": finite_pass,
                "hba_better_uniform_w2": hba_better_uniform_w2,
                "hba_better_uniform_c4": hba_better_uniform_c4,
                "hba_not_worse_flat_w2": hba_not_worse_flat_w2,
                "hba_not_worse_flat_c4": hba_not_worse_flat_c4,
                "classification": p11_classification,
            },
        },
        "p12_a": {
            "fixed_hba_layers": hba_layers,
            "variants": {
                "ternrefine_hba": variants["hba_task"],
                "reconstruction_cpsr_hba": variants["reconstruction_hba"],
                "reconstruction_global": variants["reconstruction_global"],
            },
            "signal_isolation": {
                "task_variant_name": "hba_task",
                "reconstruction_variant_name": "reconstruction_hba",
                "same_layer_set": variants["hba_task"]["selected_layers"] == variants["reconstruction_hba"]["selected_layers"],
                "same_num_edits": variants["hba_task"]["num_edits"] == variants["reconstruction_hba"]["num_edits"],
                "task_better_w2": nll("hba_task", "wikitext2_untouched") < nll("reconstruction_hba", "wikitext2_untouched"),
                "task_better_c4": nll("hba_task", "c4_untouched") < nll("reconstruction_hba", "c4_untouched"),
            },
            "external_reconstruction_baseline_status": "not_run_in_this_anonymous_artifact",
        },
        "timing": {**timing, "total_sec": elapsed},
        "gate": {
            "legality_pass": legality_pass,
            "finite_pass": finite_pass,
            "p11_classification": p11_classification,
            "p12_signal_isolation_complete": True,
            "exact_budget_all_variants": all(row["num_edits"] == 384 and row["changed_coordinates"] == 768 for row in variants.values()),
        },
    }
    out_path = out_dir / "p11_p12_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {out_path}")
    log(json.dumps(result["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
