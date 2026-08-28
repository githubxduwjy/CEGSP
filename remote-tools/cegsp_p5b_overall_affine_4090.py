#!/usr/bin/env python3
"""P5-B: whole-candidate-space affine CEGSP compatibility test.

The layer-selection rule is frozen before evaluation: score every candidate
OPT Q/K layer using fit-split quantized-point CE gradients, rank layers by the
sum of their top probe candidates, select top-4 and top-6, then apply a fixed
64 relocation pairs per selected layer. Validation and untouched splits are
never used for selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_ce_gradient_4090 import collect_ce_qk_grads
from cegsp_p5a_affine_adapter_feasibility_4090 import (
    AffineCode,
    AffineEdit,
    apply_affine_patch,
    apply_edits,
    audit_all,
    build_group_candidates,
    build_random_edits,
    cardinality_violations,
    eval_metrics,
    make_affine_code,
    restore_qk,
    select_unique_edits,
    snapshot_qk,
    with_ppl,
)
from cegsp_v2_p4_gap_cost_4090 import build_c4_untouched_batches
from tqgsp_support_projection_4090 import (
    build_wikitext_splits,
    log,
    parse_csv_ints,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default=",".join(str(x) for x in range(24)))
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
    p.add_argument("--layer-budgets", default="4,6")
    p.add_argument("--edits-per-layer", type=int, default=64)
    p.add_argument("--layer-probe-edits", type=int, default=8)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def finite_metrics(metrics: Dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def changed_coordinates(
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]],
) -> int:
    total = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            total += int((states[layer][key] != code.T).sum().item())
    return total


def per_layer_edit_counts(edits: Sequence[AffineEdit]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edit in edits:
        key = str(edit.layer)
        counts[key] = counts.get(key, 0) + 1
    return counts


def select_layerwise_edits(
    candidates_by_layer: Dict[int, List[AffineEdit]],
    selected_layers: Sequence[int],
    edits_per_layer: int,
) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    for layer in selected_layers:
        selected.extend(
            select_unique_edits(candidates_by_layer[layer], edits_per_layer)
        )
    return selected


def random_layerwise_edits(
    codes: Dict[int, Dict[str, AffineCode]],
    selected_layers: Sequence[int],
    per_layer_counts: Dict[str, int],
    seed: int,
) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    for offset, layer in enumerate(selected_layers):
        layer_codes = {layer: codes[layer]}
        selected.extend(
            build_random_edits(
                layer_codes,
                int(per_layer_counts[str(layer)]),
                seed + 1009 * (offset + 1),
            )
        )
    return selected


def metric_delta(
    metrics: Dict[str, float], baseline: Dict[str, float]
) -> Dict[str, float]:
    return {key: float(metrics[key] - baseline[key]) for key in metrics}


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("P5-B requires CUDA/RTX 4090")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    layers = parse_csv_ints(args.layers)
    layer_budgets = parse_csv_ints(args.layer_budgets)
    if layer_budgets != [4, 6]:
        raise ValueError("P5-B pre-registration requires --layer-budgets 4,6")
    if 6 > len(layers):
        raise ValueError("P5-B needs at least six candidate layers")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"loading {args.model} dtype={args.dtype} layers={layers} "
        f"budgets={layer_budgets} gpu={torch.cuda.get_device_name(0)}"
    )

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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()

    fp_qk = snapshot_qk(model, layers)
    fp_metrics = eval_metrics(model, device, val, w2, c4)
    codes: Dict[int, Dict[str, AffineCode]] = {
        layer: {
            key: make_affine_code(
                fp_qk[layer][key], args.group_size, args.threshold_factor
            )
            for key in ("q", "k")
        }
        for layer in layers
    }
    baseline_audit = audit_all(codes)
    apply_affine_patch(model, codes)
    affine_metrics = eval_metrics(model, device, val, w2, c4)

    # The only gradient signal used by the selection rule is fit-split CE at
    # the deployed affine ternary point.
    grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_ranking_rows: List[Dict[str, object]] = []
    for layer in layers:
        candidates: List[AffineEdit] = []
        for key in ("q", "k"):
            candidates.extend(
                build_group_candidates(
                    layer,
                    key,
                    codes[layer][key],
                    grads[layer][key],
                    "affine_fp",
                )
            )
        candidates.sort(key=lambda edit: (-edit.score_formula, edit.key, edit.row, edit.block))
        candidates_by_layer[layer] = candidates
        probe = candidates[: args.layer_probe_edits]
        layer_ranking_rows.append(
            {
                "layer": layer,
                "num_candidates": len(candidates),
                "probe_edits": len(probe),
                "layer_score_top_probe_sum": float(sum(e.score_formula for e in probe)),
                "top_score": float(probe[0].score_formula) if probe else float("nan"),
                "top_scores": [float(e.score_formula) for e in probe[:5]],
            }
        )
    layer_ranking_rows.sort(
        key=lambda row: (-float(row["layer_score_top_probe_sum"]), int(row["layer"]))
    )

    variants: Dict[str, Dict[str, object]] = {}
    for budget in layer_budgets:
        selected_layers = [
            int(row["layer"]) for row in layer_ranking_rows[:budget]
        ]
        ce_edits = select_layerwise_edits(
            candidates_by_layer,
            selected_layers,
            args.edits_per_layer,
        )
        ce_states = apply_edits(codes, ce_edits)
        apply_affine_patch(model, codes, ce_states)
        ce_metrics = eval_metrics(model, device, val, w2, c4)
        ce_audit = audit_all(codes, ce_states)
        per_layer = per_layer_edit_counts(ce_edits)
        ce_name = f"affine_ce_top{budget}"
        variants[ce_name] = {
            "selected_layers": selected_layers,
            "num_edits": len(ce_edits),
            "changed_coordinates": changed_coordinates(codes, ce_states),
            "edits_per_layer": per_layer,
            "metrics": with_ppl(ce_metrics),
            "delta_vs_affine_nll": metric_delta(ce_metrics, affine_metrics),
            "audit": ce_audit,
            "cardinality_violations": cardinality_violations(codes, ce_states),
            "score_identity_max_abs_error": max(
                (abs(e.score_formula - e.score_exact) for e in ce_edits),
                default=0.0,
            ),
        }

        random_edits = random_layerwise_edits(
            codes,
            selected_layers,
            per_layer,
            args.seed + budget,
        )
        random_states = apply_edits(codes, random_edits)
        apply_affine_patch(model, codes, random_states)
        random_metrics = eval_metrics(model, device, val, w2, c4)
        random_audit = audit_all(codes, random_states)
        random_name = f"random_matched_top{budget}"
        variants[random_name] = {
            "selected_layers": selected_layers,
            "num_edits": len(random_edits),
            "changed_coordinates": changed_coordinates(codes, random_states),
            "edits_per_layer": per_layer_edit_counts(random_edits),
            "metrics": with_ppl(random_metrics),
            "delta_vs_affine_nll": metric_delta(random_metrics, affine_metrics),
            "audit": random_audit,
            "cardinality_violations": cardinality_violations(codes, random_states),
        }
        apply_affine_patch(model, codes)

    def nll(name: str, split: str) -> float:
        return float(variants[name]["metrics"][split]["nll"])

    legality_pass = (
        baseline_audit["total_illegal_states"] == 0
        and baseline_audit["max_codebook_residual"] == 0.0
    )
    for variant in variants.values():
        legality_pass = legality_pass and (
            variant["audit"]["total_illegal_states"] == 0
            and variant["cardinality_violations"] == 0
        )
    finite_pass = finite_metrics(fp_metrics) and finite_metrics(affine_metrics)
    for variant in variants.values():
        finite_pass = finite_pass and finite_metrics(
            {key: value["nll"] for key, value in variant["metrics"].items()}
        )
    primary_ce = "affine_ce_top6"
    primary_random = "random_matched_top6"
    primary_improves_val = nll(primary_ce, "val") < float(affine_metrics["val"])
    primary_improves_w2 = nll(primary_ce, "wikitext2_untouched") < float(
        affine_metrics["wikitext2_untouched"]
    )
    primary_beats_random = nll(primary_ce, "wikitext2_untouched") < nll(
        primary_random, "wikitext2_untouched"
    )
    primary_pass = legality_pass and finite_pass and primary_improves_val and primary_improves_w2 and primary_beats_random
    secondary_pass = (
        nll("affine_ce_top4", "val") < float(affine_metrics["val"])
        and nll("affine_ce_top4", "wikitext2_untouched")
        < float(affine_metrics["wikitext2_untouched"])
        and nll("affine_ce_top4", "wikitext2_untouched")
        < nll("random_matched_top4", "wikitext2_untouched")
    )

    restore_qk(model, fp_qk)
    result = {
        "run_id": args.run_id,
        "experiment": "CEGSP-P5-B overall affine compatibility",
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "selection_uses_validation": False,
            "selection_uses_untouched": False,
            "selection_signal": "fit-split quantized-point CE gradient",
            "layer_score": "sum of top-8 legal affine_fp candidate scores per layer",
            "primary_budget": "top-6 layers x 64 edits/layer",
            "secondary_budget": "top-4 layers x 64 edits/layer",
            "random_control": "same selected layers and per-layer edit counts",
            "teacher_or_qat": False,
            "mu_alpha_refit": False,
        },
        "data": {
            "wikitext_source": wikitext_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "wikitext2_untouched_batches": len(w2),
            "c4_untouched_batches": len(c4),
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "fp_metrics": with_ppl(fp_metrics),
        "affine_baseline_metrics": with_ppl(affine_metrics),
        "affine_baseline_audit": baseline_audit,
        "layer_ranking": layer_ranking_rows,
        "variants": variants,
        "gate": {
            "legality_pass": legality_pass,
            "finite_pass": finite_pass,
            "primary_improves_val": primary_improves_val,
            "primary_improves_w2": primary_improves_w2,
            "primary_beats_random_w2": primary_beats_random,
            "primary_pass_overall_affine_compatibility": primary_pass,
            "secondary_top4_pass": secondary_pass,
        },
        "elapsed_sec": time.time() - started,
    }
    out_path = out_dir / "p5b_overall_affine_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {out_path}")
    log(json.dumps(result["gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
