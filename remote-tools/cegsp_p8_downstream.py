#!/usr/bin/env python3
"""P8-A: bounded downstream log-likelihood screening for frozen affine CEGSP.

This is deliberately a screening evaluator, not a new optimization stage.  It
reconstructs the already frozen P7 top-6 affine CEGSP rule from the Wikitext
fit split, then evaluates BF16, affine ternary, and affine+CEGSP on fixed
validation examples from PIQA and ARC-Easy.  Downstream examples never enter
layer or edit selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_p7_a100_scaling import (
    AffineCode,
    apply_affine_patch,
    apply_edits,
    audit_all,
    build_top_candidates,
    collect_grads,
    get_decoder_layers,
    make_affine_code,
    snapshot_qk,
    build_splits,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_tasks(text: str) -> List[str]:
    allowed = {"piqa", "arc_easy"}
    tasks = [x.strip() for x in text.split(",") if x.strip()]
    if not tasks or any(x not in allowed for x in tasks):
        raise ValueError(f"tasks must be a non-empty subset of {sorted(allowed)}")
    return tasks


def load_task_examples(tasks: Sequence[str], max_examples: int) -> Dict[str, List[Dict[str, object]]]:
    from datasets import load_dataset

    out: Dict[str, List[Dict[str, object]]] = {}
    for task in tasks:
        if task == "piqa":
            ds = load_dataset("piqa", split="validation")
            rows: List[Dict[str, object]] = []
            for row in ds:
                label = int(row["label"])
                rows.append(
                    {
                        "id": str(len(rows)),
                        "prefix": str(row["goal"]).strip() + "\nSolution:\n",
                        "options": [str(row["sol1"]).strip(), str(row["sol2"]).strip()],
                        "label": label,
                    }
                )
                if len(rows) >= max_examples:
                    break
            out[task] = rows
        elif task == "arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
            rows = []
            for row in ds:
                choices = row["choices"]
                labels = [str(x) for x in choices["label"]]
                options = [str(x).strip() for x in choices["text"]]
                answer_key = str(row["answerKey"])
                if answer_key in labels:
                    label = labels.index(answer_key)
                else:
                    label = int(answer_key) - 1
                rows.append(
                    {
                        "id": str(row.get("id", len(rows))),
                        "prefix": str(row["question"]).strip() + "\nAnswer:\n",
                        "options": options,
                        "label": label,
                    }
                )
                if len(rows) >= max_examples:
                    break
            out[task] = rows
    if any(not out[task] for task in tasks):
        raise RuntimeError("one or more downstream tasks yielded zero examples")
    return out


@torch.no_grad()
def score_choices(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[Dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, object]]:
    model.eval()
    rows: List[Dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        full_ids: List[List[int]] = []
        prefix_lens: List[int] = []
        option_counts: List[int] = []
        option_labels: List[int] = []
        row_ids: List[str] = []
        for ex in chunk:
            prefix = str(ex["prefix"])
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            options = [str(x) for x in ex["options"]]
            for option in options:
                ids = tokenizer(prefix + option, add_special_tokens=False)["input_ids"]
                if len(ids) <= len(prefix_ids):
                    raise RuntimeError("downstream option produced no answer tokens")
                full_ids.append(ids)
                prefix_lens.append(len(prefix_ids))
            option_counts.append(len(options))
            option_labels.append(int(ex["label"]))
            row_ids.append(str(ex["id"]))
        encoded = tokenizer.pad(
            {"input_ids": full_ids}, padding=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        flat_scores: List[float] = []
        for row_idx, (ids, prefix_len) in enumerate(zip(full_ids, prefix_lens)):
            token_scores = []
            for pos in range(prefix_len, len(ids)):
                token_id = int(ids[pos])
                token_scores.append(float(log_probs[row_idx, pos - 1, token_id].item()))
            flat_scores.append(sum(token_scores) / max(len(token_scores), 1))
        cursor = 0
        for ex, count, label, row_id in zip(chunk, option_counts, option_labels, row_ids):
            scores = flat_scores[cursor : cursor + count]
            cursor += count
            predicted = max(range(count), key=lambda i: scores[i])
            gold = scores[label]
            rows.append(
                {
                    "id": row_id,
                    "label": label,
                    "prediction": predicted,
                    "correct": int(predicted == label),
                    "gold_normalized_loglikelihood": gold,
                    "option_normalized_loglikelihood": scores,
                }
            )
    return rows


def summarize_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    accuracy = sum(int(row["correct"]) for row in rows) / max(len(rows), 1)
    gold_ll = sum(float(row["gold_normalized_loglikelihood"]) for row in rows) / max(len(rows), 1)
    margins = []
    for row in rows:
        scores = [float(x) for x in row["option_normalized_loglikelihood"]]
        label = int(row["label"])
        margins.append(scores[label] - max(scores[:label] + scores[label + 1 :]))
    return {
        "n_examples": int(len(rows)),
        "accuracy": float(accuracy),
        "mean_gold_normalized_loglikelihood": float(gold_ll),
        "mean_gold_margin": float(sum(margins) / max(len(margins), 1)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--tasks", default="piqa,arc_easy")
    p.add_argument("--max-examples", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--fit-batches", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--layer-probe-edits", type=int, default=8)
    p.add_argument("--edits-per-layer", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--layer-budget", type=int, default=6)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("P8 requires CUDA")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = parse_tasks(args.tasks)

    log(f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, _, _, _, fit_source = build_splits(
        tokenizer, args.seq_len, args.batch_size, args.fit_batches, 1, 1, 0, 0
    )
    task_examples = load_task_examples(tasks, args.max_examples)
    log(f"loading model {args.model} dtype={args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    model.config.use_cache = False
    model.eval()
    layers = list(range(len(get_decoder_layers(model))))
    log(f"model_type={model.config.model_type} layers={len(layers)} gpu={torch.cuda.get_device_name(0)}")

    fp_weights = snapshot_qk(model, layers)
    codes: Dict[int, Dict[str, AffineCode]] = {
        layer: {
            key: make_affine_code(fp_weights[layer][key], args.group_size, args.threshold_factor)
            for key in ("q", "k")
        }
        for layer in layers
    }
    baseline_audit = audit_all(codes)
    fp_task_rows = {task: score_choices(model, tokenizer, rows, device, args.eval_batch_size) for task, rows in task_examples.items()}
    apply_affine_patch(model, codes)
    affine_task_rows = {task: score_choices(model, tokenizer, rows, device, args.eval_batch_size) for task, rows in task_examples.items()}

    grads = collect_grads(model, fit, layers, device, args.grad_batches)
    keep_per_module = max(args.edits_per_layer * 2, args.layer_probe_edits * 2, 128)
    layer_candidates = {layer: build_top_candidates(codes, grads, layer, keep_per_module) for layer in layers}
    ranking = []
    for layer in layers:
        probe = layer_candidates[layer][: args.layer_probe_edits]
        ranking.append(
            {
                "layer": layer,
                "num_kept_candidates": len(layer_candidates[layer]),
                "probe_edits": len(probe),
                "layer_score_top_probe_sum": float(sum(edit.score for edit in probe)),
                "top_score": float(probe[0].score) if probe else float("nan"),
            }
        )
    ranking.sort(key=lambda row: (-float(row["layer_score_top_probe_sum"]), int(row["layer"])))
    selected_layers = [int(row["layer"]) for row in ranking[: args.layer_budget]]
    edits = []
    for layer in selected_layers:
        edits.extend(layer_candidates[layer][: args.edits_per_layer])
    states = apply_edits(codes, edits)
    apply_affine_patch(model, codes, states)
    cegsp_task_rows = {task: score_choices(model, tokenizer, rows, device, args.eval_batch_size) for task, rows in task_examples.items()}

    metrics = {}
    for task in tasks:
        metrics[task] = {
            "bf16": summarize_rows(fp_task_rows[task]),
            "affine_baseline": summarize_rows(affine_task_rows[task]),
            "affine_cegsp": summarize_rows(cegsp_task_rows[task]),
        }
        metrics[task]["delta_cegsp_vs_affine"] = {
            key: float(metrics[task]["affine_cegsp"][key] - metrics[task]["affine_baseline"][key])
            for key in ("accuracy", "mean_gold_normalized_loglikelihood", "mean_gold_margin")
        }

    result = {
        "run_id": args.run_id,
        "experiment": "CEGSP-P8-A bounded downstream log-likelihood screening",
        "status": "complete",
        "config": vars(args),
        "data_source": {
            "fit": fit_source,
            "downstream": {
                "piqa": "piqa:validation",
                "arc_easy": "allenai/ai2_arc:ARC-Easy:validation",
            },
        },
        "protocol": {
            "selection_signal": "Wikitext fit-split quantized-point CE gradient",
            "downstream_used_for_selection": False,
            "representation": "Q=mu+alpha*T, T in {-1,0,+1}",
            "scope": "all decoder layers, q_proj/k_proj only",
            "layer_budget": args.layer_budget,
            "edits_per_layer": args.edits_per_layer,
            "teacher_or_qat": False,
            "mu_alpha_refit": False,
            "primary_task_metric": "mean gold normalized log-likelihood",
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "baseline_audit": baseline_audit,
        "selected_layers": selected_layers,
        "num_edits": len(edits),
        "changed_coordinates": int(sum((states[layer][key] != codes[layer][key].T).sum().item() for layer in codes for key in codes[layer])),
        "cegsp_audit": audit_all(codes, states),
        "metrics": metrics,
        "raw_rows": {
            "bf16": fp_task_rows,
            "affine_baseline": affine_task_rows,
            "affine_cegsp": cegsp_task_rows,
        },
        "layer_ranking": ranking,
        "elapsed_sec": time.time() - started,
    }
    out_path = out_dir / "p8_downstream_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
