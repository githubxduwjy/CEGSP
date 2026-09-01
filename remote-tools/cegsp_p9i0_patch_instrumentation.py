#!/usr/bin/env python3
"""P9-I0: validate patch-level CEGSP instrumentation on a detached sidecar.

This is deliberately not a performance experiment.  It reloads the real
P9-S2 ternary sidecar, performs one scoped quantized-point backward pass, saves
the complete candidate table and selected patch, then checks save/reload,
hash, legality, and cardinality invariants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_p7_a100_scaling import (
    AffineCode,
    AffineEdit,
    apply_edits,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    get_decoder_layers,
)
from cegsp_p9s2_detached_pt2_plugin import (
    apply_ssr_codes,
    load_detached_artifacts,
    target_qk,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_layers(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor bytes without converting large tensors to Python lists."""
    cpu = tensor.detach().contiguous().cpu()
    try:
        payload = cpu.numpy().tobytes()
    except Exception as exc:  # pragma: no cover - only for broken NumPy bridges
        raise RuntimeError("P9-I0 requires a working tensor-to-NumPy byte bridge") from exc
    prefix = f"dtype={cpu.dtype};shape={tuple(cpu.shape)};".encode("utf-8")
    return hashlib.sha256(prefix + payload).hexdigest()


def json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def state_fingerprint(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]] | None = None) -> str:
    rows: List[Dict[str, object]] = []
    for layer in sorted(codes):
        for key in sorted(codes[layer]):
            code = codes[layer][key]
            state = code.T if states is None else states[layer][key]
            rows.append({
                "layer": int(layer),
                "key": key,
                "T_sha256": tensor_sha256(state.to(torch.int8)),
                "shape": list(state.shape),
            })
    return json_sha256(rows)


def snapshot_qk(model: torch.nn.Module, layers: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        out[layer] = {key: ref.module.weight.detach().float().cpu().clone() for key, ref in target_qk(model, layer).items()}
    return out


def load_calibration(tokenizer, data_root: Path, seq_len: int, batch_size: int, batches: int) -> Tuple[List[torch.Tensor], Dict[str, object]]:
    """Load the pinned PT2 Wikitext train Arrow dataset without importing PT2."""
    from datasets import load_from_disk

    train_path = data_root / "wikitext" / "traindata"
    if not train_path.exists():
        raise FileNotFoundError(f"missing PT2 calibration dataset: {train_path}")
    dataset = load_from_disk(str(train_path))
    texts = [str(row.get("text", "")) for row in dataset if str(row.get("text", "")).strip()]
    text = "\n".join(texts)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    needed = batches * batch_size * (seq_len + 1)
    if ids.numel() < needed:
        raise RuntimeError(f"not enough calibration tokens: {ids.numel()} < {needed}")
    stream = ids[:needed].view(batches, batch_size, seq_len + 1).contiguous()
    return [row.clone() for row in stream], {
        "path": str(train_path),
        "split": "wikitext train",
        "batches": batches,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "token_count": int(needed),
        "token_sha256": tensor_sha256(stream.to(torch.int64)),
    }


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


def candidate_record(edit: AffineEdit, rank: int, selected: bool) -> Dict[str, object]:
    row = edit_identity(edit)
    row.update({"rank": int(rank), "score": float(edit.score), "selected": bool(selected)})
    return row


def select_nonoverlap(candidates_by_layer: Dict[int, List[AffineEdit]], layers: Sequence[int], edits_per_layer: int) -> List[AffineEdit]:
    selected: List[AffineEdit] = []
    for layer in layers:
        used = set()
        for edit in candidates_by_layer[layer]:
            coords = {(edit.key, edit.row, edit.block, edit.donor), (edit.key, edit.row, edit.block, edit.receiver)}
            if used.intersection(coords):
                continue
            selected.append(edit)
            used.update(coords)
            if sum(1 for item in selected if item.layer == layer) >= edits_per_layer:
                break
    return selected


def cardinality_violations(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> int:
    violations = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            before = (code.T.abs() * code.valid).sum(dim=-1)
            after = (states[layer][key].abs() * code.valid).sum(dim=-1)
            violations += int((before != after).sum().item())
    return violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sidecar-dir", required=True)
    parser.add_argument("--pt2-data-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="/root/tqgsp-runs")
    parser.add_argument("--layers", default="4,10,11,9,14,5")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-batches", type=int, default=1)
    parser.add_argument("--candidate-topk", type=int, default=128)
    parser.add_argument("--edits-per-layer", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.candidate_topk < args.edits_per_layer + 6:
        raise ValueError("candidate-topk must leave room for the pre-registered boundary margins")
    if not torch.cuda.is_available():
        raise RuntimeError("P9-I0 requires CUDA")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    dtype = torch.bfloat16
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    log(f"loading sidecar={args.sidecar_dir}")
    sidecar_dir = Path(args.sidecar_dir)
    codes_a, perms_a, qk_checkpoint_a = load_detached_artifacts(sidecar_dir)
    codes_b, perms_b, qk_checkpoint_b = load_detached_artifacts(sidecar_dir)
    if state_fingerprint(codes_a) != state_fingerprint(codes_b):
        raise RuntimeError("sidecar reload state fingerprint mismatch")
    if sorted(perms_a) != sorted(perms_b):
        raise RuntimeError("sidecar reload layer mismatch")
    if state_fingerprint(codes_a) != state_fingerprint(codes_b):
        raise RuntimeError("nondeterministic state fingerprint")
    if sorted(codes_a) != sorted(codes_b) or any(sorted(codes_a[layer]) != sorted(codes_b[layer]) for layer in codes_a):
        raise RuntimeError("sidecar reload module mismatch")
    codes = codes_a
    perms = perms_a

    log(f"loading tokenizer/model={args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, data_meta = load_calibration(tokenizer, Path(args.pt2_data_root), args.seq_len, args.batch_size, args.grad_batches)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True).to(device)
    model.config.use_cache = False
    model.eval()
    total_layers = len(get_decoder_layers(model))
    if any(layer < 0 or layer >= total_layers for layer in layers):
        raise ValueError(f"invalid layer in {layers}; model has {total_layers} layers")
    apply_ssr_codes(model, codes, perms, None)
    baseline_q = snapshot_qk(model, layers)
    # The detached checkpoint is serialized in FP32 while the model is
    # intentionally deployed in BF16 for this smoke.  Use a fixed dtype-aware
    # reconstruction tolerance; this is not a tunable experiment parameter.
    deployment_q_tolerance = 5e-3
    baseline_q_residual = max(
        float((baseline_q[layer][key] - qk_checkpoint_a[layer][key].float()).abs().max().item())
        for layer in layers for key in ("q", "k")
    )
    if baseline_q_residual >= deployment_q_tolerance:
        raise RuntimeError(f"sidecar deployment exceeds BF16 qk tolerance: {baseline_q_residual}")
    baseline_q_hash = json_sha256({str(layer): {key: tensor_sha256(value) for key, value in baseline_q[layer].items()} for layer in layers})
    log(f"one backward over layers={layers}")
    grads_original = collect_grads(model, fit, layers, device, args.grad_batches)
    gradients_ssr = {layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")} for layer in layers}
    gradient_hash = json_sha256({str(layer): {key: tensor_sha256(grads_original[layer][key]) for key in ("q", "k")} for layer in layers})

    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    ranking: List[Dict[str, object]] = []
    for layer in layers:
        candidates = build_top_candidates(codes, gradients_ssr, layer, args.candidate_topk)
        if len(candidates) < args.edits_per_layer + 1:
            raise RuntimeError(f"insufficient candidates at layer {layer}: {len(candidates)}")
        candidates_by_layer[layer] = candidates
        scores = [float(edit.score) for edit in candidates]
        margins = {}
        for left, right in ((60, 61), (64, 65), (70, 71)):
            if right <= len(scores):
                margins[f"S{left}-S{right}"] = float(scores[left - 1] - scores[right - 1])
        ranking.append({
            "layer": int(layer),
            "num_candidates": len(candidates),
            "top_scores": scores,
            "boundary_margins": margins,
        })
    selected = select_nonoverlap(candidates_by_layer, layers, args.edits_per_layer)
    selected_ids = {tuple(edit_identity(edit).values()) for edit in selected}
    candidate_rows: List[Dict[str, object]] = []
    for layer in layers:
        for rank, edit in enumerate(candidates_by_layer[layer], start=1):
            identity = tuple(edit_identity(edit).values())
            candidate_rows.append(candidate_record(edit, rank, identity in selected_ids))
    identity_fields = ("layer", "projection", "row", "group", "donor_index", "receiver_index", "donor_sign", "receiver_sign")
    selected_rows = sorted((row for row in candidate_rows if row["selected"]), key=lambda row: tuple(row[key] for key in identity_fields))
    patch_identity = [{key: row[key] for key in identity_fields} for row in selected_rows]
    patch_hash = json_sha256(patch_identity)
    states = apply_edits(codes, selected)
    state_before_hash = state_fingerprint(codes)
    state_after_hash = state_fingerprint(codes, states)
    audit = audit_all(codes, states)
    card_violations = cardinality_violations(codes, states)
    if len(selected) != len(layers) * args.edits_per_layer:
        raise RuntimeError(f"selected edit count mismatch: {len(selected)}")
    if changed_coordinates(codes, states) != 2 * len(selected):
        raise RuntimeError("changed coordinate count is not two per relocation")
    if audit["total_illegal_states"] != 0 or card_violations != 0:
        raise RuntimeError(f"legality/cardinality failure audit={audit} cardinality={card_violations}")

    payload = {
        "format": "CEGSP_P9I0_INSTRUMENTATION_V1",
        "run_id": args.run_id,
        "config": vars(args),
        "layers": layers,
        "data": data_meta,
        "fingerprints": {
            "calibration_token_sha256": data_meta["token_sha256"],
            "gradient_sha256": gradient_hash,
            "baseline_q_sha256": baseline_q_hash,
            "state_before_sha256": state_before_hash,
            "state_after_sha256": state_after_hash,
            "patch_sha256": patch_hash,
        },
        "candidate_count": len(candidate_rows),
        "candidate_count_per_layer": {str(layer): len(candidates_by_layer[layer]) for layer in layers},
        "selected_count": len(selected_rows),
        "changed_coordinates": changed_coordinates(codes, states),
        "candidate_records": candidate_rows,
        "selected_patch": selected_rows,
        "layer_ranking": ranking,
        "legality": audit,
        "cardinality_violations": card_violations,
    }
    payload_path = out_dir / "candidate_records.json"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    reloaded = json.loads(payload_path.read_text())
    reload_patch_hash = json_sha256([{key: row[key] for key in identity_fields} for row in reloaded["selected_patch"]])
    reload_state = apply_edits(codes, [
        AffineEdit(
            int(row["layer"]), str(row["projection"]), int(row["row"]), int(row["group"]),
            int(row["donor_index"]), int(row["receiver_index"]), int(row["donor_sign"]), int(row["receiver_sign"]), float(row["score"]),
        ) for row in reloaded["selected_patch"]
    ])
    reload_state_hash = state_fingerprint(codes, reload_state)
    result = {
        "run_id": args.run_id,
        "experiment": "P9-I0 patch-level instrumentation validation",
        "status": "complete",
        "protocol": {
            "pt2_rerun": False,
            "performance_eval": False,
            "layers": layers,
            "qk_only": True,
            "one_backward": True,
            "candidate_topk_per_module": args.candidate_topk,
            "candidate_topk_per_layer": 2 * args.candidate_topk,
            "selected_edits_per_layer": args.edits_per_layer,
            "untouched_used_for_selection": False,
            "parameter_search": False,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "sidecar_reload": {
            "pass": True,
            "two_loads_consistent": True,
            "qk_checkpoint_modules": sum(len(value) for value in qk_checkpoint_a.values()),
            "baseline_q_vs_checkpoint_max_abs": baseline_q_residual,
            "baseline_q_tolerance": deployment_q_tolerance,
            "baseline_q_reconstruction_pass": baseline_q_residual < deployment_q_tolerance,
            "qk_checkpoint_reload_consistent": sorted(qk_checkpoint_a) == sorted(qk_checkpoint_b),
        },
        "instrumentation_gate": {
            "candidate_table_complete": len(candidate_rows) == len(layers) * 2 * args.candidate_topk,
            "selected_patch_complete": len(selected_rows) == len(layers) * args.edits_per_layer,
            "patch_reload_hash_match": patch_hash == reload_patch_hash,
            "patch_reconstruction_hash_match": state_after_hash == reload_state_hash,
            "candidate_identity_fields_present": all(all(field in row for field in ("layer", "projection", "row", "group", "donor_index", "receiver_index", "receiver_sign", "score", "selected")) for row in candidate_rows),
            "boundary_margins_saved": all(len(row["boundary_margins"]) == 3 for row in ranking),
            "finite_scores": all(math.isfinite(float(row["score"])) for row in candidate_rows),
            "legality_pass": audit["total_illegal_states"] == 0,
            "cardinality_pass": card_violations == 0,
        },
        "fingerprints": payload["fingerprints"],
        "artifact": str(payload_path),
        "elapsed_sec": time.time() - started,
    }
    result["instrumentation_gate"]["pass"] = all(bool(value) for value in result["instrumentation_gate"].values())
    result_path = out_dir / "p9i0_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {result_path}; instrumentation_pass={result['instrumentation_gate']['pass']}")


if __name__ == "__main__":
    main()
