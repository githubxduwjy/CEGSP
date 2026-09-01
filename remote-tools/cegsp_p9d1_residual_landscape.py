#!/usr/bin/env python3
"""P9-D1: compare the single-move discrete landscape of two initializers.

This is a diagnostic, not a new CEGSP performance run.  It computes one
fit-split quantized-point gradient for ordinary affine ternarization and for
the detached real PT2 ATQ+SSR state.  It then evaluates a frozen, single
legal support relocation at a time on fixed validation and untouched W2
slices.  No candidate is selected using either evaluation split.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_p7_a100_scaling import (
    AffineCode,
    AffineEdit,
    affine_weight,
    apply_affine_patch,
    build_top_candidates,
    collect_grads,
    get_decoder_layers,
    make_affine_code,
    target_qk,
)
from cegsp_p9s2_detached_pt2_plugin import (
    apply_ssr_codes,
    load_detached_artifacts,
    set_module_weight,
    snapshot_qk,
)


RANKS = (1, 2, 4, 8, 16, 32, 64, 128)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    payload = f"dtype={cpu.dtype};shape={tuple(cpu.shape)};".encode("utf-8")
    payload += cpu.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_text(path: Path) -> Tuple[str, int]:
    from datasets import load_from_disk

    if not path.exists():
        raise FileNotFoundError(f"missing dataset: {path}")
    dataset = load_from_disk(str(path))
    chunks: List[str] = []
    rows = 0
    for row in dataset:
        text = str(row.get("text", ""))
        if text.strip():
            chunks.append(text)
            rows += 1
    if not chunks:
        raise RuntimeError(f"dataset has no usable text: {path}")
    return "\n".join(chunks), rows


def make_stream(ids: torch.Tensor, start: int, batches: int, batch_size: int, seq_len: int, label: str) -> List[torch.Tensor]:
    needed = batches * batch_size * (seq_len + 1)
    end = start + needed
    if ids.numel() < end:
        raise RuntimeError(f"{label} needs {end} tokens, only {ids.numel()} available")
    return [x.clone() for x in ids[start:end].view(batches, batch_size, seq_len + 1).contiguous()]


def load_fixed_splits(tokenizer, data_root: Path, seq_len: int, batch_size: int, fit_batches: int, val_batches: int, w2_batches: int):
    train_text, train_rows = load_text(data_root / "wikitext" / "traindata")
    w2_text, w2_rows = load_text(data_root / "wikitext" / "testdata")
    train_ids = tokenizer(train_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    w2_ids = tokenizer(w2_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    per_batch = batch_size * (seq_len + 1)
    fit = make_stream(train_ids, 0, fit_batches, batch_size, seq_len, "fit")
    val = make_stream(train_ids, fit_batches * per_batch, val_batches, batch_size, seq_len, "validation")
    w2 = make_stream(w2_ids, 0, w2_batches, batch_size, seq_len, "untouched Wikitext-2")
    meta = {
        "fit": {
            "source": str(data_root / "wikitext" / "traindata"),
            "split": "wikitext train prefix",
            "dataset_rows": train_rows,
            "batches": fit_batches,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "token_sha256": tensor_sha256(torch.cat(fit, dim=0).to(torch.int64)),
        },
        "validation": {
            "source": str(data_root / "wikitext" / "traindata"),
            "split": "wikitext train disjoint validation slice",
            "start_token": fit_batches * per_batch,
            "batches": val_batches,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "token_sha256": tensor_sha256(torch.cat(val, dim=0).to(torch.int64)),
        },
        "untouched_wikitext2": {
            "source": str(data_root / "wikitext" / "testdata"),
            "split": "wikitext-2 test prefix",
            "dataset_rows": w2_rows,
            "batches": w2_batches,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "token_sha256": tensor_sha256(torch.cat(w2, dim=0).to(torch.int64)),
        },
    }
    return fit, val, w2, meta


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, device: torch.device, batches: Iterable[torch.Tensor]) -> float:
    losses: List[float] = []
    for batch in batches:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.item()))
    value = float(sum(losses) / max(len(losses), 1))
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite NLL: {value}")
    return value


def edit_identity(edit: AffineEdit) -> Dict[str, object]:
    return {
        "layer": int(edit.layer),
        "projection": str(edit.key),
        "row": int(edit.row),
        "group": int(edit.block),
        "donor_index": int(edit.donor),
        "receiver_index": int(edit.receiver),
        "donor_sign": int(edit.donor_sign),
        "receiver_sign": int(edit.receiver_sign),
    }


def padded_gradient(code: AffineCode, grad_2d: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(code.T, dtype=torch.float32)
    out.view(code.T.shape[0], -1)[:, : grad_2d.shape[1]] = grad_2d.detach().float().cpu()
    return out


def score_edit(code: AffineCode, grad_2d: torch.Tensor, edit: AffineEdit) -> float:
    grad = padded_gradient(code, grad_2d)
    alpha = float(code.alpha[edit.row, edit.block, 0].item())
    donor_grad = float(grad[edit.row, edit.block, edit.donor].item())
    receiver_grad = float(grad[edit.row, edit.block, edit.receiver].item())
    return float(alpha * (edit.donor_sign * donor_grad - edit.receiver_sign * receiver_grad))


def legal_edit(code: AffineCode, edit: AffineEdit) -> bool:
    if not bool(code.valid[edit.row, edit.block, edit.donor].item()) or not bool(code.valid[edit.row, edit.block, edit.receiver].item()):
        return False
    if int(code.T[edit.row, edit.block, edit.donor].item()) != int(edit.donor_sign):
        return False
    if int(code.T[edit.row, edit.block, edit.receiver].item()) != 0:
        return False
    return edit.donor != edit.receiver and int(edit.donor_sign) in (-1, 1) and int(edit.receiver_sign) in (-1, 1)


def random_legal_edits(
    codes: Dict[int, Dict[str, AffineCode]],
    gradients: Dict[int, Dict[str, torch.Tensor]],
    layers: Sequence[int],
    count_per_layer: int,
    seed: int,
) -> List[AffineEdit]:
    rng = random.Random(seed)
    out: List[AffineEdit] = []
    for layer in layers:
        used = set()
        attempts = 0
        while sum(1 for e in out if e.layer == layer) < count_per_layer and attempts < 100000:
            attempts += 1
            key = rng.choice(["q", "k"])
            code = codes[layer][key]
            grad = gradients[layer][key]
            row = rng.randrange(code.T.shape[0])
            block = rng.randrange(code.T.shape[1])
            valid = code.valid[row, block]
            donors = [int(x) for x in torch.where((code.T[row, block] != 0) & valid)[0].tolist()]
            receivers = [int(x) for x in torch.where((code.T[row, block] == 0) & valid)[0].tolist()]
            if not donors or not receivers:
                continue
            donor = rng.choice(donors)
            receiver = rng.choice(receivers)
            identity = (layer, key, row, block, donor, receiver)
            if identity in used:
                continue
            centered = float(code.fp_padded[row, block, receiver] - code.mu[row, block, 0])
            receiver_sign = 1 if centered >= 0 else -1
            edit = AffineEdit(layer, key, row, block, donor, receiver, int(code.T[row, block, donor].item()), receiver_sign, 0.0)
            edit = AffineEdit(*edit_identity(edit).values(), score_edit(code, grad, edit))
            out.append(edit)
            used.add(identity)
        if sum(1 for e in out if e.layer == layer) != count_per_layer:
            raise RuntimeError(f"could not sample {count_per_layer} random legal edits at layer {layer}")
    return out


def candidate_weight(code: AffineCode, edit: AffineEdit, perm: torch.Tensor | None) -> torch.Tensor:
    state = code.T.clone()
    state[edit.row, edit.block, edit.donor] = 0
    state[edit.row, edit.block, edit.receiver] = edit.receiver_sign
    q = affine_weight(code, state)
    if perm is None:
        return q
    inverse = torch.argsort(perm)
    return q[:, inverse].contiguous()


def rank_values(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        average = 0.5 * (pos + 1 + end)
        for index in order[pos:end]:
            ranks[index] = average
        pos = end
    return ranks


def pearson(x: List[float], y: List[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return float(sum(a * b for a, b in zip(dx, dy)) / denom) if denom > 0 else 0.0


def spearman(records: List[Dict[str, object]], split: str) -> float:
    scores = [float(row["score"]) for row in records]
    targets = [-float(row[f"delta_{split}_nll"]) for row in records]
    return pearson(rank_values(scores), rank_values(targets))


def summarize_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "candidate_count": len(records),
        "finite": all(math.isfinite(float(row["score"])) and math.isfinite(float(row["delta_val_nll"])) and math.isfinite(float(row["delta_untouched_w2_nll"])) for row in records),
        "nonfinite_count": 0,
        "rho_val": spearman(records, "val"),
        "rho_untouched_w2": spearman(records, "untouched_w2"),
    }
    result["nonfinite_count"] = sum(
        1 for row in records
        if not all(math.isfinite(float(row[key])) for key in ("score", "delta_val_nll", "delta_untouched_w2_nll"))
    )
    groups = {
        "all": records,
        "rank1": [row for row in records if row["sampling"] == "ranked" and int(row["rank"]) == 1],
        "rank_sample": [row for row in records if row["sampling"] == "ranked"],
        "random": [row for row in records if row["sampling"] == "random"],
    }
    for name, rows in groups.items():
        result[f"{name}_count"] = len(rows)
        for split in ("val", "untouched_w2"):
            deltas = [float(row[f"delta_{split}_nll"]) for row in rows]
            result[f"{name}_mean_delta_{split}_nll"] = float(sum(deltas) / len(deltas)) if deltas else float("nan")
            result[f"{name}_positive_density_{split}"] = float(sum(delta < 0 for delta in deltas) / len(deltas)) if deltas else float("nan")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--pt2-data-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="/root/tqgsp-runs")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fit-batches", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--w2-batches", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--threshold-factor", type=float, default=0.75)
    parser.add_argument("--candidate-topk-per-module", type=int, default=128)
    parser.add_argument("--random-count-per-layer", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def validate_codebook(codes: Dict[int, Dict[str, AffineCode]], expected_layers: int = 32) -> Dict[str, int]:
    illegal = 0
    nonfinite = 0
    modules = 0
    for layer_codes in codes.values():
        for code in layer_codes.values():
            modules += 1
            illegal += int(((code.T < -1) | (code.T > 1) | (~code.valid & (code.T != 0))).sum().item())
            nonfinite += int((~torch.isfinite(code.mu)).sum().item()) + int((~torch.isfinite(code.alpha)).sum().item())
    if len(codes) != expected_layers or modules != 2 * expected_layers:
        raise RuntimeError(f"expected {expected_layers} layers and {2 * expected_layers} Q/K modules, got {len(codes)} and {modules}")
    return {"layers": len(codes), "qk_modules": modules, "illegal_states": illegal, "nonfinite_code_values": nonfinite}


def run_initializer(
    name: str,
    args: argparse.Namespace,
    tokenizer,
    fit: List[torch.Tensor],
    val: List[torch.Tensor],
    w2: List[torch.Tensor],
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    pt2_codes: Dict[int, Dict[str, AffineCode]],
    pt2_perms: Dict[int, Dict[str, torch.Tensor]],
    device: torch.device,
) -> Dict[str, object]:
    layers = list(range(32))
    if name == "ordinary_affine":
        codes = {layer: {key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor) for key in ("q", "k")} for layer in layers}
        perms = {layer: {"q": torch.arange(fp_qk[layer]["q"].shape[1]), "k": torch.arange(fp_qk[layer]["k"].shape[1])} for layer in layers}
    elif name == "pt2_atq_ssr":
        codes = pt2_codes
        perms = pt2_perms
    else:
        raise ValueError(name)
    audit = validate_codebook(codes)
    log(f"loading {name} model")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    model.config.use_cache = False
    model.eval()
    if name == "ordinary_affine":
        apply_affine_patch(model, codes)
    else:
        apply_ssr_codes(model, codes, perms, None)
    baseline_q = snapshot_qk(model, layers)
    baseline = {
        "val_nll": evaluate_nll(model, device, val),
        "untouched_w2_nll": evaluate_nll(model, device, w2),
    }
    log(f"{name}: one fit backward over 32 layers / 64 QK modules")
    grads_original = collect_grads(model, fit, layers, device, 1)
    grads_code = {
        layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")}
        for layer in layers
    }
    gradient_fingerprint = json_sha256({str(layer): {key: tensor_sha256(grads_original[layer][key]) for key in ("q", "k")} for layer in layers})
    candidates: List[Tuple[AffineEdit, str, int]] = []
    for layer in layers:
        ranked = build_top_candidates(codes, grads_code, layer, args.candidate_topk_per_module)
        if len(ranked) < max(RANKS):
            raise RuntimeError(f"{name} layer {layer} has only {len(ranked)} ranked candidates")
        for rank in RANKS:
            edit = ranked[rank - 1]
            candidates.append((edit, "ranked", rank))
    random_edits = random_legal_edits(codes, grads_code, layers, args.random_count_per_layer, args.seed)
    candidates.extend((edit, "random", 0) for edit in random_edits)
    if len(candidates) != 32 * (len(RANKS) + args.random_count_per_layer):
        raise RuntimeError(f"{name} candidate count mismatch: {len(candidates)}")
    for edit, _, _ in candidates:
        if not legal_edit(codes[edit.layer][edit.key], edit):
            raise RuntimeError(f"illegal candidate: {edit}")
    baseline_records: Dict[Tuple[int, str], torch.Tensor] = {(layer, key): baseline_q[layer][key] for layer in layers for key in ("q", "k")}
    records: List[Dict[str, object]] = []
    total = len(candidates)
    for index, (edit, sampling, rank) in enumerate(candidates, start=1):
        code = codes[edit.layer][edit.key]
        perm = perms[edit.layer][edit.key] if name == "pt2_atq_ssr" else None
        ref = target_qk(model, edit.layer)[edit.key]
        set_module_weight(ref.module, candidate_weight(code, edit, perm))
        val_nll = evaluate_nll(model, device, val)
        w2_nll = evaluate_nll(model, device, w2)
        set_module_weight(ref.module, baseline_records[(edit.layer, edit.key)])
        identity = edit_identity(edit)
        identity.update({
            "initializer": name,
            "sampling": sampling,
            "rank": rank,
            "score": float(edit.score),
            "baseline_val_nll": float(baseline["val_nll"]),
            "baseline_untouched_w2_nll": float(baseline["untouched_w2_nll"]),
            "candidate_val_nll": float(val_nll),
            "candidate_untouched_w2_nll": float(w2_nll),
            "delta_val_nll": float(val_nll - baseline["val_nll"]),
            "delta_untouched_w2_nll": float(w2_nll - baseline["untouched_w2_nll"]),
        })
        records.append(identity)
        if index == 1 or index % 32 == 0 or index == total:
            log(f"{name}: evaluated {index}/{total} single moves")
    result = {
        "initializer": name,
        "baseline": baseline,
        "codebook_audit": audit,
        "gradient_sha256": gradient_fingerprint,
        "records": records,
        "summary": summarize_records(records),
        "max_memory_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("P9-D1 requires CUDA")
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = list(range(32))
    sidecar_dir = Path(args.sidecar_dir)
    log(f"loading detached sidecar={sidecar_dir}")
    pt2_codes, pt2_perms, qk_checkpoint = load_detached_artifacts(sidecar_dir)
    if sorted(pt2_codes) != layers or any(sorted(pt2_codes[layer]) != ["k", "q"] for layer in layers):
        raise RuntimeError("detached sidecar does not cover all 32 Q/K layers")
    sidecar_audit = validate_codebook(pt2_codes)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, w2, data_meta = load_fixed_splits(tokenizer, Path(args.pt2_data_root), args.seq_len, args.batch_size, args.fit_batches, args.val_batches, args.w2_batches)
    fp_qk = torch.load(sidecar_dir / "fp_qk.pt", map_location="cpu")
    init_results: Dict[str, object] = {}
    for name in ("ordinary_affine", "pt2_atq_ssr"):
        init_results[name] = run_initializer(name, args, tokenizer, fit, val, w2, fp_qk, pt2_codes, pt2_perms, device)
    expected = 32 * (len(RANKS) + args.random_count_per_layer)
    if any(int(init_results[name]["summary"]["candidate_count"]) != expected for name in init_results):
        raise RuntimeError("candidate count gate failed")
    all_records = [row for value in init_results.values() for row in value["records"]]
    result = {
        "format": "CEGSP_P9D1_RESIDUAL_LANDSCAPE_V1",
        "run_id": args.run_id,
        "experiment": "P9-D1 ordinary-affine vs PT2 residual discrete landscape",
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "diagnostic_only": True,
            "initializer_only_difference": True,
            "scope": "all 32 decoder layers, Q/K only",
            "one_backward_per_initializer": True,
            "single_move_only": True,
            "ranked_ranks_per_layer": list(RANKS),
            "random_legal_moves_per_layer": args.random_count_per_layer,
            "candidate_count_per_initializer": expected,
            "total_candidate_count": len(all_records),
            "candidate_selection_uses_validation": False,
            "candidate_selection_uses_untouched_w2": False,
            "mu_alpha_frozen": True,
            "canonical_rule_tuned": False,
            "c4_candidate_eval": False,
        },
        "data": data_meta,
        "sidecar_audit": sidecar_audit,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_gb": float(torch.cuda.max_memory_allocated() / (1024 ** 3)),
        },
        "initializers": init_results,
        "integrity": {
            "candidate_count_complete": all(int(init_results[name]["summary"]["candidate_count"]) == expected for name in init_results),
            "finite": all(bool(init_results[name]["summary"]["finite"]) for name in init_results),
            "nonfinite_count": sum(int(init_results[name]["summary"]["nonfinite_count"]) for name in init_results),
            "legal_candidates": True,
            "state_cardinality_preserved_by_each_move": True,
            "split_leakage_by_construction": False,
        },
        "elapsed_sec": float(time.time() - started),
    }
    result_path = out_dir / "p9d1_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    log(f"wrote {result_path}")


if __name__ == "__main__":
    main()
