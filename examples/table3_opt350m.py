#!/usr/bin/env python3
"""E1: quantized-point vs FP-point task-gradient ranking.

This runner keeps one ordinary affine ternary Q0 and one shared legal CPSR
candidate pool per fitting offset.  It compares four rankings on that same
candidate pool:

  * quantized-point CE gradient (QGP)
  * pre-quantization FP-point CE gradient
  * FP-weight reconstruction score
  * deterministic random score

Outputs are checkpointed after every offset so the run can be resumed without
repeating completed splits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ternrefine.ce_gradient import collect_ce_qk_grads
from ternrefine.task_vs_reconstruction import (
    LAYERS,
    build_c4_cached_batches,
    changed_coordinates,
    edit_id,
    finite_metrics,
    metric_delta,
    reconstruction_score,
)
from ternrefine.affine_adapter import (
    AffineCode,
    AffineEdit,
    apply_affine_patch,
    apply_edits,
    audit_all,
    build_group_candidates,
    cardinality_violations,
    eval_metrics,
    make_affine_code,
    snapshot_qk,
    with_ppl,
)
from ternrefine.data_eval import evaluate_nll, log, parse_csv_ints, read_wikitext_arrow_cache


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="results/table3_opt350m")
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    p.add_argument("--offsets", default="0,1024,2048,3072,4096")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--w2-batches", type=int, default=8)
    p.add_argument("--c4-batches", type=int, default=8)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--edits-total", type=int, default=384)
    p.add_argument("--single-sample", type=int, default=512)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--eval-pack-factor", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260910)
    return p.parse_args()


def ranks(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    out = np.empty(len(order), dtype=np.float64)
    out[order] = np.arange(len(order), dtype=np.float64)
    return out


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=np.float64)
    y_arr = np.asarray(list(y), dtype=np.float64)
    if x_arr.size < 3 or np.std(x_arr) == 0.0 or np.std(y_arr) == 0.0:
        return float("nan")
    return float(np.corrcoef(ranks(x_arr), ranks(y_arr))[0, 1])


def pad_grad_like_code(code: AffineCode, grad_2d: torch.Tensor) -> torch.Tensor:
    grad = torch.zeros_like(code.fp_padded, dtype=torch.float32)
    flat = grad.view(grad.shape[0], -1)
    gcols = min(flat.shape[1], grad_2d.shape[1])
    flat[:, :gcols] = grad_2d.detach().float().cpu()[:, :gcols]
    return grad


def score_with_padded_grad(code: AffineCode, grad: torch.Tensor, edit: AffineEdit) -> float:
    alpha = float(code.alpha[edit.row, edit.block, 0].item())
    return float(
        alpha
        * (
            float(grad[edit.row, edit.block, edit.donor]) * int(edit.donor_sign)
            - float(grad[edit.row, edit.block, edit.receiver]) * int(edit.receiver_sign)
        )
    )


def edit_with_score(edit: AffineEdit, score: float) -> AffineEdit:
    return replace(edit, score_formula=float(score), score_exact=float(score))


def unique_candidates(candidates: Sequence[AffineEdit]) -> List[AffineEdit]:
    seen = set()
    out: List[AffineEdit] = []
    for edit in candidates:
        ident = edit_id(edit)
        if ident in seen:
            continue
        seen.add(ident)
        out.append(edit)
    return out


def select_unique_ranked(edits: Sequence[AffineEdit], total: int) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    used = set()
    for edit in edits:
        donor = (int(edit.layer), edit.key, int(edit.row), int(edit.block), int(edit.donor))
        receiver = (int(edit.layer), edit.key, int(edit.row), int(edit.block), int(edit.receiver))
        if donor in used or receiver in used:
            continue
        selected.append(edit)
        used.add(donor)
        used.add(receiver)
        if len(selected) == total:
            break
    return selected


def per_layer_counts(edits: Sequence[AffineEdit]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edit in edits:
        counts[str(edit.layer)] = counts.get(str(edit.layer), 0) + 1
    return counts


def stratified_sample(rows: List[Dict[str, object]], n: int, seed: int) -> List[Dict[str, object]]:
    if len(rows) <= n:
        return list(rows)
    rng = random.Random(seed)
    sorted_rows = sorted(rows, key=lambda r: float(r["q_score"]))
    buckets = np.array_split(np.arange(len(sorted_rows)), 10)
    per_bucket = max(1, n // len(buckets))
    chosen = []
    for bucket in buckets:
        idxs = bucket.tolist()
        rng.shuffle(idxs)
        chosen.extend(sorted_rows[i] for i in idxs[:per_bucket])
    if len(chosen) < n:
        remaining_ids = {id(r) for r in chosen}
        rest = [r for r in rows if id(r) not in remaining_ids]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return chosen[:n]


def pack_batches(batches: Sequence[torch.Tensor], factor: int) -> List[torch.Tensor]:
    if factor <= 1:
        return list(batches)
    packed: List[torch.Tensor] = []
    for i in range(0, len(batches), factor):
        packed.append(torch.cat(list(batches[i : i + factor]), dim=0).contiguous())
    return packed


def top_stats(rows: List[Dict[str, object]], score_key: str, delta_key: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    ranked = sorted(rows, key=lambda r: -float(r[score_key]))
    for frac in (0.01, 0.05, 0.10, 0.20):
        k = max(1, int(round(len(ranked) * frac)))
        top = ranked[:k]
        out[f"precision_at_top_{int(frac * 100)}pct"] = float(np.mean([float(r[delta_key]) < 0.0 for r in top]))
        out[f"mean_gain_at_top_{int(frac * 100)}pct"] = float(np.mean([-float(r[delta_key]) for r in top]))
    return out


def build_wikitext_splits_arrow(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    fit_batches: int,
    val_batches: int,
    untouched_batches: int,
    fit_token_offset: int,
    val_token_offset: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], str]:
    train_text, valid_text = read_wikitext_arrow_cache()

    def make(text: str, n_batches: int, offset: int) -> List[torch.Tensor]:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        needed = n_batches * batch_size * (seq_len + 1)
        if ids.numel() < offset + needed:
            raise RuntimeError(f"not enough tokens for split: have={ids.numel()} need={offset + needed}")
        return [
            x.clone()
            for x in ids[offset : offset + needed].view(n_batches, batch_size, seq_len + 1)
        ]

    fit = make(train_text, fit_batches, fit_token_offset)
    val = make(valid_text, val_batches, val_token_offset)
    untouched_offset = val_token_offset + val_batches * batch_size * (seq_len + 1)
    untouched = make(valid_text, untouched_batches, untouched_offset)
    return fit, val, untouched, "wikitext-2-raw-v1-arrow-cache"


def evaluate_one_edit(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    edit: AffineEdit,
    fit: Sequence[torch.Tensor],
    val: Sequence[torch.Tensor],
    w2: Sequence[torch.Tensor],
    c4: Sequence[torch.Tensor],
    device: torch.device,
    baseline_fit: float,
    baseline_metrics: Dict[str, float],
) -> Dict[str, float]:
    states = apply_edits(codes, [edit])
    apply_affine_patch(model, codes, states)
    fit_nll = evaluate_nll(model, fit, device)
    metrics = eval_metrics(model, device, val, w2, c4)
    apply_affine_patch(model, codes)
    return {
        "fit_delta_nll": float(fit_nll - baseline_fit),
        "val_delta_nll": float(metrics["val"] - baseline_metrics["val"]),
        "w2_delta_nll": float(metrics["wikitext2_untouched"] - baseline_metrics["wikitext2_untouched"]),
        "c4_delta_nll": float(metrics["c4_untouched"] - baseline_metrics["c4_untouched"]),
    }


def evaluate_patch(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    edits: Sequence[AffineEdit],
    val: Sequence[torch.Tensor],
    w2: Sequence[torch.Tensor],
    c4: Sequence[torch.Tensor],
    device: torch.device,
    baseline_metrics: Dict[str, float],
) -> Dict[str, object]:
    states = apply_edits(codes, edits)
    apply_affine_patch(model, codes, states)
    metrics = eval_metrics(model, device, val, w2, c4)
    audit = audit_all(codes, states)
    apply_affine_patch(model, codes)
    return {
        "num_edits": len(edits),
        "changed_coordinates": changed_coordinates(codes, states),
        "edits_per_layer": per_layer_counts(edits),
        "metrics": with_ppl(metrics),
        "delta_vs_affine_nll": metric_delta(metrics, baseline_metrics),
        "audit": audit,
        "cardinality_violations": cardinality_violations(codes, states),
        "finite": finite_metrics(metrics),
        "selected_edit_ids": [edit_id(e) for e in edits],
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("E1 requires CUDA")
    layers = parse_csv_ints(args.layers)
    offsets = parse_csv_ints(args.offsets)
    if layers != LAYERS:
        raise ValueError("E1 OPT-350M protocol expects all 24 layers")
    if args.edits_total != 384 or args.group_size != 128:
        raise ValueError("E1 frozen protocol expects 384 relocations and group size 128")

    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    log(f"E1 loading tokenizer/model={args.model} dtype={args.dtype} offsets={offsets}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_summaries = []
    for split_idx, fit_offset in enumerate(offsets):
        split_dir = out / f"offset_{fit_offset}"
        split_dir.mkdir(parents=True, exist_ok=True)
        split_result_path = split_dir / "e1_split_result.json"
        if split_result_path.exists():
            split_summaries.append(json.loads(split_result_path.read_text()))
            log(f"E1 resume skip completed offset={fit_offset}")
            continue

        t0 = time.time()
        log(f"E1 offset={fit_offset} start")
        fit, val, w2, wikitext_source = build_wikitext_splits_arrow(
            tokenizer,
            args.seq_len,
            args.batch_size,
            args.fit_batches,
            args.val_batches,
            args.w2_batches,
            fit_offset,
            args.val_token_offset,
        )
        c4 = build_c4_cached_batches(tokenizer, args.seq_len, args.batch_size, args.c4_batches, args.c4_token_offset)
        fit_eval = pack_batches(fit, args.eval_pack_factor)
        val_eval = pack_batches(val, args.eval_pack_factor)
        w2_eval = pack_batches(w2, args.eval_pack_factor)
        c4_eval = pack_batches(c4, args.eval_pack_factor)
        log(
            f"E1 offset={fit_offset} data ready source={wikitext_source} "
            f"c4_batches={len(c4)} eval_pack_factor={args.eval_pack_factor} "
            f"packed_fit/val/w2/c4={len(fit_eval)}/{len(val_eval)}/{len(w2_eval)}/{len(c4_eval)}"
        )

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
        model.config.use_cache = False
        model.eval()
        log(f"E1 offset={fit_offset} model loaded")
        fp_qk = snapshot_qk(model, layers)
        fp_grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
        log(f"E1 offset={fit_offset} FP-point gradient ready")

        codes: Dict[int, Dict[str, AffineCode]] = {
            layer: {key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor) for key in ("q", "k")}
            for layer in layers
        }
        apply_affine_patch(model, codes)
        affine_metrics = eval_metrics(model, device, val_eval, w2_eval, c4_eval)
        affine_fit = evaluate_nll(model, fit_eval, device)
        log(f"E1 offset={fit_offset} affine baseline ready metrics={with_ppl(affine_metrics)}")
        q_grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
        log(f"E1 offset={fit_offset} quantized-point gradient ready")

        q_candidates: List[AffineEdit] = []
        fp_candidates: List[AffineEdit] = []
        for layer in layers:
            for key in ("q", "k"):
                q_candidates.extend(build_group_candidates(layer, key, codes[layer][key], q_grads[layer][key], "affine_fp"))
                fp_candidates.extend(build_group_candidates(layer, key, codes[layer][key], fp_grads[layer][key], "affine_fp"))
        pool = unique_candidates(q_candidates + fp_candidates)
        log(f"E1 offset={fit_offset} shared candidate pool built count={len(pool)}")
        rng = random.Random(args.seed + int(fit_offset))
        random_scores = {edit_id(edit): rng.random() for edit in pool}
        q_grad_padded = {
            layer: {key: pad_grad_like_code(codes[layer][key], q_grads[layer][key]) for key in ("q", "k")}
            for layer in layers
        }
        fp_grad_padded = {
            layer: {key: pad_grad_like_code(codes[layer][key], fp_grads[layer][key]) for key in ("q", "k")}
            for layer in layers
        }

        candidate_rows_path = split_dir / "e1_candidate_rows.json"
        scored_edits_path = split_dir / "e1_scored_edit_ids.json"
        candidate_rows = []
        scored = {"qgp": [], "fp_point": [], "reconstruction": [], "random": []}
        if candidate_rows_path.exists() and scored_edits_path.exists():
            candidate_rows = json.loads(candidate_rows_path.read_text())
            scored_ids = json.loads(scored_edits_path.read_text())
            edit_by_id_resume = {edit_id(edit): edit for edit in pool}
            for name, ids in scored_ids.items():
                for ident in ids:
                    edit = edit_by_id_resume[tuple(ident["id"])]
                    scored[name].append(edit_with_score(edit, float(ident["score"])))
            log(f"E1 offset={fit_offset} resumed scored candidate rows count={len(candidate_rows)}")
        else:
            for edit in pool:
                code = codes[int(edit.layer)][edit.key]
                ident = edit_id(edit)
                q_score = score_with_padded_grad(code, q_grad_padded[int(edit.layer)][edit.key], edit)
                fp_score = score_with_padded_grad(code, fp_grad_padded[int(edit.layer)][edit.key], edit)
                rec_score = reconstruction_score(code, edit)
                rand_score = random_scores[ident]
                row = {
                    "id": ident,
                    "layer": int(edit.layer),
                    "key": edit.key,
                    "q_score": float(q_score),
                    "fp_score": float(fp_score),
                    "reconstruction_score": float(rec_score),
                    "random_score": float(rand_score),
                }
                candidate_rows.append(row)
                scored["qgp"].append(edit_with_score(edit, q_score))
                scored["fp_point"].append(edit_with_score(edit, fp_score))
                scored["reconstruction"].append(edit_with_score(edit, rec_score))
                scored["random"].append(edit_with_score(edit, rand_score))

            for name in scored:
                scored[name].sort(key=lambda e: (-e.score_formula, edit_id(e)))
            candidate_rows_path.write_text(json.dumps(candidate_rows, ensure_ascii=False))
            scored_edits_path.write_text(
                json.dumps(
                    {
                        name: [{"id": edit_id(edit), "score": float(edit.score_formula)} for edit in edits]
                        for name, edits in scored.items()
                    },
                    ensure_ascii=False,
                )
            )
            log(f"E1 offset={fit_offset} scored candidate rows checkpointed count={len(candidate_rows)}")

        for name in scored:
            scored[name].sort(key=lambda e: (-e.score_formula, edit_id(e)))

        single_rows = stratified_sample(candidate_rows, args.single_sample, args.seed + 1000 + int(fit_offset))
        edit_by_id = {edit_id(edit): edit for edit in pool}
        for idx, row in enumerate(single_rows):
            deltas = evaluate_one_edit(
                model,
                codes,
                edit_by_id[tuple(row["id"])],
                fit_eval,
                val_eval,
                w2_eval,
                c4_eval,
                device,
                affine_fit,
                affine_metrics,
            )
            row.update(deltas)
            if (idx + 1) % 64 == 0:
                (split_dir / "e1_single_partial.json").write_text(json.dumps(single_rows[: idx + 1], indent=2))
                log(f"E1 offset={fit_offset} single-edit {idx + 1}/{len(single_rows)}")

        fixed_budget = {}
        for name, ranked in scored.items():
            chosen = select_unique_ranked(ranked, args.edits_total)
            if len(chosen) != args.edits_total:
                raise RuntimeError(f"{name} could not select exact {args.edits_total} edits from shared pool")
            fixed_budget[name] = evaluate_patch(model, codes, chosen, val_eval, w2_eval, c4_eval, device, affine_metrics)
            log(f"E1 offset={fit_offset} fixed {name}: {fixed_budget[name]['delta_vs_affine_nll']}")

        q_ids = [tuple(e) for e in fixed_budget["qgp"]["selected_edit_ids"]]
        fp_ids = [tuple(e) for e in fixed_budget["fp_point"]["selected_edit_ids"]]
        rec_ids = [tuple(e) for e in fixed_budget["reconstruction"]["selected_edit_ids"]]
        q_set, fp_set, rec_set = set(q_ids), set(fp_ids), set(rec_ids)

        single = {
            "evaluated_count": len(single_rows),
            "spearman_qgp_vs_real_fit_gain": spearman([r["q_score"] for r in single_rows], [-r["fit_delta_nll"] for r in single_rows]),
            "spearman_fp_point_vs_real_fit_gain": spearman([r["fp_score"] for r in single_rows], [-r["fit_delta_nll"] for r in single_rows]),
            "spearman_reconstruction_vs_real_fit_gain": spearman([r["reconstruction_score"] for r in single_rows], [-r["fit_delta_nll"] for r in single_rows]),
            "spearman_qgp_vs_real_val_gain": spearman([r["q_score"] for r in single_rows], [-r["val_delta_nll"] for r in single_rows]),
            "spearman_fp_point_vs_real_val_gain": spearman([r["fp_score"] for r in single_rows], [-r["val_delta_nll"] for r in single_rows]),
            "spearman_qgp_vs_fp_point": spearman([r["q_score"] for r in single_rows], [r["fp_score"] for r in single_rows]),
            "spearman_qgp_vs_reconstruction": spearman([r["q_score"] for r in single_rows], [r["reconstruction_score"] for r in single_rows]),
            "beneficial_density_fit": float(np.mean([float(r["fit_delta_nll"]) < 0.0 for r in single_rows])),
            "beneficial_density_val": float(np.mean([float(r["val_delta_nll"]) < 0.0 for r in single_rows])),
            "qgp_top_stats_fit": top_stats(single_rows, "q_score", "fit_delta_nll"),
            "fp_point_top_stats_fit": top_stats(single_rows, "fp_score", "fit_delta_nll"),
            "reconstruction_top_stats_fit": top_stats(single_rows, "reconstruction_score", "fit_delta_nll"),
        }

        split_result = {
            "status": "complete",
            "fit_offset": int(fit_offset),
            "protocol": {
                "same_q0": True,
                "same_shared_candidate_pool": True,
                "candidate_pool": "union of quantized-point and FP-point CPSR proposals, rescored by all objectives",
                "action_space_changed_by_fp_baseline": False,
                "single_sample": int(args.single_sample),
                "fixed_budget_relocations": int(args.edits_total),
                "changed_coordinates_expected": int(args.edits_total * 2),
                "grad_batches": int(args.grad_batches),
            },
            "data": {
                "source": wikitext_source,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "fit_batches": len(fit),
                "val_batches": len(val),
                "w2_batches": len(w2),
                "c4_batches": len(c4),
            },
            "affine_baseline_metrics": with_ppl(affine_metrics),
            "affine_fit_nll": float(affine_fit),
            "candidate_pool_count": len(pool),
            "single_relocation": single,
            "fixed_budget": fixed_budget,
            "fixed_budget_overlap": {
                "qgp_fp_point_jaccard": float(len(q_set & fp_set) / max(1, len(q_set | fp_set))),
                "qgp_reconstruction_jaccard": float(len(q_set & rec_set) / max(1, len(q_set | rec_set))),
                "fp_point_reconstruction_jaccard": float(len(fp_set & rec_set) / max(1, len(fp_set | rec_set))),
            },
            "candidate_rows_sampled": single_rows,
            "timing_sec": time.time() - t0,
        }
        split_result_path.write_text(json.dumps(split_result, indent=2, ensure_ascii=False))
        split_summaries.append({k: v for k, v in split_result.items() if k != "candidate_rows_sampled"})
        (out / "e1_partial_summary.json").write_text(json.dumps({"splits": split_summaries}, indent=2, ensure_ascii=False))
        del model
        torch.cuda.empty_cache()
        log(f"E1 offset={fit_offset} done in {time.time() - t0:.1f}s")

    def collect_delta(method: str, split_name: str) -> List[float]:
        return [float(s["fixed_budget"][method]["delta_vs_affine_nll"][split_name]) for s in split_summaries]

    methods = ["random", "reconstruction", "fp_point", "qgp"]
    aggregate = {}
    for method in methods:
        aggregate[method] = {}
        for split_name in ("val", "wikitext2_untouched", "c4_untouched"):
            vals = collect_delta(method, split_name)
            aggregate[method][split_name] = {
                "mean_delta_nll": float(np.mean(vals)),
                "std_delta_nll": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "improved_count": int(sum(v < 0.0 for v in vals)),
                "n": len(vals),
            }
    result = {
        "status": "complete",
        "experiment": "E1 quantized-point vs FP-point task gradient",
        "run_id": args.run_id,
        "aggregate": aggregate,
        "splits": split_summaries,
        "timing_sec": time.time() - started,
    }
    (out / "e1_qpoint_vs_fppoint_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"E1 complete result={out / 'e1_qpoint_vs_fppoint_result.json'}")


if __name__ == "__main__":
    main()
