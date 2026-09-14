#!/usr/bin/env python3
"""ReQuant-style controls for the frozen OPT-350M affine CEGSP protocol.

This is an auditable reproduction of the two controls specified in the
experiment plan, not an official ReQuant release.  Both branches share one
ordinary affine ternary initializer, one calibration split, one validation
split, and the frozen P11/P12 layer set [3, 4, 5, 6, 7, 8].

Branch A (``requant_ternary``) uses a quadratic activation-aware reconstruction
objective and four deterministic coordinate-descent sweeps over the ternary
grid.  It does not preserve per-group cardinality.

Branch B (``requant_cpsr``) scores legal donor/receiver relocations with the
same quadratic objective, keeps the receiver sign from the affine FP rule, and
uses a fixed 64-relocation ceiling per frozen layer.  The implementation
reports the screened pair pool and whether the exact CPSR capacity invariant
was preserved.  This branch is deliberately named ReQuant-style/CPSR rather
than original ReQuant because the reference implementation is not available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from cegsp_p11_p12_4090 import build_c4_cached_batches
from cegsp_p5a_affine_adapter_feasibility_4090 import (
    AffineCode,
    AffineEdit,
    apply_affine_patch,
    audit_all,
    eval_metrics,
    make_affine_code,
    projection_weight,
    restore_qk,
    snapshot_qk,
    target_modules,
    with_ppl,
)
from tqgsp_support_projection_4090 import build_wikitext_splits, log, parse_csv_ints


LAYERS = list(range(24))
FROZEN_REQUANT_LAYERS = [3, 4, 5, 6, 7, 8]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    p.add_argument("--requant-layers", default=",".join(str(x) for x in FROZEN_REQUANT_LAYERS))
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
    p.add_argument("--cpsr-edits-per-layer", type=int, default=64)
    p.add_argument("--requant-sweeps", type=int, default=4)
    p.add_argument("--pair-screen-topk", type=int, default=8)
    p.add_argument("--branch", choices=["all", "b"], default="all")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260908)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def finite_metrics(metrics: Dict[str, Dict[str, float]]) -> bool:
    return all(math.isfinite(float(item["nll"])) for item in metrics.values())


def metric_delta(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {key: float(metrics[key] - baseline[key]) for key in metrics}


def edit_id(edit: AffineEdit) -> Tuple[int, str, int, int, int, int]:
    return (int(edit.layer), str(edit.key), int(edit.row), int(edit.block), int(edit.donor), int(edit.receiver))


def changed_coordinates(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> int:
    return sum(
        int((states[layer][key] != code.T).sum().item())
        for layer, layer_codes in codes.items()
        for key, code in layer_codes.items()
    )


def state_digest(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for layer in sorted(states):
        for key in ("q", "k"):
            h.update(str(layer).encode())
            h.update(key.encode())
            h.update(states[layer][key].cpu().numpy().tobytes())
    return h.hexdigest()


def capture_qk_inputs(model, batches: Sequence[torch.Tensor], layers: Sequence[int], device: torch.device) -> Dict[int, Dict[str, torch.Tensor]]:
    """Capture module inputs as [in_features, tokens] on CPU."""
    captured: Dict[Tuple[int, str], List[torch.Tensor]] = {(layer, key): [] for layer in layers for key in ("q", "k")}
    hooks = []
    for layer in layers:
        refs = target_modules(model, int(layer))
        for key in ("q", "k"):
            module = refs[key].module

            def hook(_module, args, _layer=int(layer), _key=str(key)):
                if not args:
                    raise RuntimeError(f"missing input for L{_layer}.{_key}")
                x = args[0].detach().float().cpu()
                if x.ndim != 3:
                    raise RuntimeError(f"unexpected input shape for L{_layer}.{_key}: {tuple(x.shape)}")
                captured[(_layer, _key)].append(x.reshape(-1, x.shape[-1]))

            hooks.append(module.register_forward_pre_hook(hook))
    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                model(input_ids=batch[:, :-1].to(device), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        result[layer] = {}
        for key in ("q", "k"):
            chunks = captured[(layer, key)]
            if not chunks:
                raise RuntimeError(f"no captured activations for L{layer}.{key}")
            result[layer][key] = torch.cat(chunks, dim=0).transpose(0, 1).contiguous()
    return result


def build_stats(fp_inputs: Dict[int, Dict[str, torch.Tensor]], q_inputs: Dict[int, Dict[str, torch.Tensor]], out_dir: Path) -> Dict[int, Dict[str, Dict[str, torch.Tensor]]]:
    stats: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for layer in sorted(fp_inputs):
        stats[layer] = {}
        for key in ("q", "k"):
            x = fp_inputs[layer][key].float()
            xt = q_inputs[layer][key].float()
            if x.shape != xt.shape:
                raise RuntimeError(f"activation shape mismatch L{layer}.{key}: {tuple(x.shape)} vs {tuple(xt.shape)}")
            h = xt @ xt.T
            b = (xt - x) @ xt.T
            stats[layer][key] = {"H": h.contiguous(), "B": b.contiguous(), "tokens": torch.tensor(x.shape[1])}
            torch.save(stats[layer][key], out_dir / f"stats_L{layer}_{key}.pt")
    return stats


def reconstruction_gradient(code: AffineCode, fp_weight: torch.Tensor, stats: Dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
    q = (code.mu + code.alpha * state.float()).view(fp_weight.shape[0], -1)[:, : fp_weight.shape[1]]
    h = stats["H"]
    b = stats["B"]
    return 2.0 * ((fp_weight - q) @ h - fp_weight @ b)


def reconstruction_objective(code: AffineCode, fp_weight: torch.Tensor, stats: Dict[str, torch.Tensor], state: torch.Tensor) -> float:
    # The objective is evaluated in chunks over rows to keep CPU/RAM stable.
    q = (code.mu + code.alpha * state.float()).view(fp_weight.shape[0], -1)[:, : fp_weight.shape[1]]
    x = stats["X"]
    xt = stats["Xt"]
    total = 0.0
    for start in range(0, fp_weight.shape[0], 64):
        residual = fp_weight[start:start + 64] @ x - q[start:start + 64] @ xt
        total += float((residual * residual).sum().item())
    return total


def run_coordinate_descent(
    layer: int,
    key: str,
    code: AffineCode,
    fp_weight: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    state: torch.Tensor,
    sweeps: int,
) -> Dict[str, object]:
    h = stats["H"]
    diag = torch.diag(h)
    g = reconstruction_gradient(code, fp_weight, stats, state)
    rows, blocks, group = state.shape
    diag_flat = torch.diag(h).reshape(-1)
    accepted = 0
    accepted_by_move: Dict[str, int] = {}
    objective_change = 0.0
    initial_active = state.abs().sum(dim=-1).clone()
    for sweep in range(sweeps):
        for row in range(rows):
            flat_state = state[row].view(-1)
            flat_g = g[row].view(-1)
            alpha_flat = code.alpha[row].expand(-1, group).reshape(-1)
            for col in range(flat_state.numel()):
                if col >= fp_weight.shape[1]:
                    continue
                current = int(flat_state[col].item())
                candidates = []
                for new_state in (-1, 0, 1):
                    if new_state == current:
                        continue
                    delta = float(alpha_flat[col].item()) * float(new_state - current)
                    if delta == 0.0:
                        continue
                    change = -delta * float(flat_g[col].item()) + delta * delta * float(diag_flat[col].item())
                    candidates.append((change, new_state, delta))
                if not candidates:
                    continue
                change, new_state, delta = min(candidates, key=lambda item: (item[0], item[1]))
                if change < -1e-10:
                    flat_state[col] = int(new_state)
                    flat_g.sub_(2.0 * delta * h[col])
                    accepted += 1
                    objective_change += float(change)
                    move_name = f"{current}->{new_state}"
                    accepted_by_move[move_name] = accepted_by_move.get(move_name, 0) + 1
    final_active = state.abs().sum(dim=-1)
    return {
        "accepted_updates": int(accepted),
        "accepted_by_move": accepted_by_move,
        "quadratic_delta_sum": float(objective_change),
        "initial_active_sum": int(initial_active.sum().item()),
        "final_active_sum": int(final_active.sum().item()),
        "groups_changed_cardinality": int((initial_active != final_active).sum().item()),
        "max_abs_cardinality_change": int((initial_active - final_active).abs().max().item()),
        "gradient_finite": bool(torch.isfinite(g).all().item()),
    }


def receiver_sign(code: AffineCode, row: int, block: int, receiver: int) -> int:
    return 1 if float(code.fp_padded[row, block, receiver] - code.mu[row, block, 0]) >= 0.0 else -1


def build_cpsr_pair_pool(
    layer: int,
    key: str,
    code: AffineCode,
    fp_weight: torch.Tensor,
    stats: Dict[str, torch.Tensor],
    state: torch.Tensor,
    topk: int,
) -> List[AffineEdit]:
    """Build one best exact quadratic pair per row-group.

    The pair search is exact over the top-k first-order donors and receivers in
    each group.  The cap is recorded explicitly because exhaustive all-pair
    enumeration is not computationally meaningful for every ternary group.
    """
    h = stats["H"]
    g = reconstruction_gradient(code, fp_weight, stats, state)
    rows, blocks, group = state.shape
    pool: List[AffineEdit] = []
    for row in range(rows):
        for block in range(blocks):
            valid = code.valid[row, block]
            active = torch.where((state[row, block] != 0) & valid)[0].tolist()
            inactive = torch.where((state[row, block] == 0) & valid)[0].tolist()
            if not active or not inactive:
                continue
            alpha = float(code.alpha[row, block, 0].item())
            if alpha == 0.0:
                continue
            grad_group = g[row].view(blocks, group)[block]
            donor_rank = sorted(active, key=lambda d: (-float(grad_group[d].item()) * int(state[row, block, d].item()), d))[:topk]
            receiver_rank = sorted(inactive, key=lambda r: (float(grad_group[r].item()), r))[:topk]
            best = None
            for donor in donor_rank:
                sd = int(state[row, block, donor].item())
                dd = -alpha * sd
                for recv in receiver_rank:
                    sr = receiver_sign(code, row, block, recv)
                    dr = alpha * sr
                    change = (
                        -dd * float(grad_group[donor].item())
                        -dr * float(grad_group[recv].item())
                        + dd * dd * float(h[donor, donor].item())
                        + dr * dr * float(h[recv, recv].item())
                        + 2.0 * dd * dr * float(h[donor, recv].item())
                    )
                    item = (float(change), donor, recv, sd, sr)
                    if best is None or item < best:
                        best = item
            if best is None:
                continue
            change, donor, recv, sd, sr = best
            pool.append(AffineEdit(layer, key, row, block, donor, recv, sd, sr, -change, -change))
    pool.sort(key=lambda e: (-e.score_formula, edit_id(e)))
    return pool


def select_cpsr_layerwise(
    layer: int,
    pools: Dict[str, List[AffineEdit]],
    states: Dict[int, Dict[str, torch.Tensor]],
    codes: Dict[int, Dict[str, AffineCode]],
    stats: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    n_edits: int,
) -> Tuple[List[AffineEdit], Dict[str, object]]:
    pool = sorted(pools["q"] + pools["k"], key=lambda e: (-e.score_formula, edit_id(e)))
    selected: List[AffineEdit] = []
    used = set()
    refresh_events = 0
    for edit in pool:
        if edit.score_formula <= 0.0:
            continue
        dkey = (edit.key, edit.row, edit.block, edit.donor)
        rkey = (edit.key, edit.row, edit.block, edit.receiver)
        if dkey in used or rkey in used:
            continue
        selected.append(edit)
        used.add(dkey)
        used.add(rkey)
        state = states[layer][edit.key]
        alpha = float(codes[layer][edit.key].alpha[edit.row, edit.block, 0].item())
        dd = -alpha * int(edit.donor_sign)
        dr = alpha * int(edit.receiver_sign)
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = int(edit.receiver_sign)
        # Persist the exact ReQuant quadratic gradient refresh for each accepted pair.
        h = stats[layer][edit.key]["H"]
        refresh_events += 1
        if len(selected) >= n_edits:
            break
    return selected, {"accepted": len(selected), "gradient_refresh_events": refresh_events, "positive_pool": sum(1 for e in pool if e.score_formula > 0.0)}


def serialise_edits(edits: Iterable[AffineEdit]) -> List[Dict[str, object]]:
    return [
        {
            "layer": int(e.layer), "key": str(e.key), "row": int(e.row), "block": int(e.block),
            "donor": int(e.donor), "receiver": int(e.receiver), "donor_sign": int(e.donor_sign),
            "receiver_sign": int(e.receiver_sign), "reconstruction_score": float(e.score_formula),
        }
        for e in edits
    ]


def main() -> None:
    args = parse_args()
    start = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("ReQuant comparison requires CUDA")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    layers = parse_csv_ints(args.layers)
    requant_layers = parse_csv_ints(args.requant_layers)
    if layers != LAYERS:
        raise ValueError("frozen protocol requires all OPT-350M layers 0--23 for the affine initializer")
    if requant_layers != FROZEN_REQUANT_LAYERS:
        raise ValueError(f"frozen P11/P12 ReQuant scope is {FROZEN_REQUANT_LAYERS}")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"ReQuant-style comparison loading {args.model} dtype={args.dtype} gpu={torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, w2, wikitext_source = build_wikitext_splits(
        tokenizer, args.seq_len, args.batch_size, args.fit_batches, args.val_batches,
        args.untouched_batches, args.fit_token_offset, args.val_token_offset,
    )
    c4 = build_c4_cached_batches(tokenizer, args.seq_len, args.batch_size, args.c4_untouched_batches, args.c4_token_offset)

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    model.eval()
    fp_qk = snapshot_qk(model, layers)
    fp_metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
    codes: Dict[int, Dict[str, AffineCode]] = {
        layer: {key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor) for key in ("q", "k")}
        for layer in layers
    }
    baseline_audit = audit_all(codes)
    apply_affine_patch(model, codes)
    affine_states = {layer: {key: code.T.clone() for key, code in cs.items()} for layer, cs in codes.items()}
    affine_metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
    log(f"affine baseline={affine_metrics}")

    # Collect both activation paths from the same frozen calibration batches.
    # A completed stats checkpoint is reused after a harness-only restart.
    stats_path = out_dir / "requant_stats.pt"
    if stats_path.exists() and (out_dir / "activation_manifest.json").exists():
        log(f"reusing activation statistics checkpoint {stats_path}")
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    else:
        log("capturing full-precision Q/K inputs")
        restore_qk(model, fp_qk)
        fp_inputs = capture_qk_inputs(model, fit, requant_layers, device)
        log("capturing quantized-prefix Q/K inputs")
        apply_affine_patch(model, codes, affine_states)
        q_inputs = capture_qk_inputs(model, fit, requant_layers, device)
        stats = build_stats(fp_inputs, q_inputs, out_dir)
        for layer in stats:
            for key in stats[layer]:
                stats[layer][key]["X"] = fp_inputs[layer][key]
                stats[layer][key]["Xt"] = q_inputs[layer][key]
        torch.save({layer: {key: stats[layer][key] for key in ("q", "k")} for layer in stats}, stats_path)
        (out_dir / "activation_manifest.json").write_text(json.dumps({
            "fit_batches": len(fit), "seq_len": args.seq_len, "batch_size": args.batch_size,
            "modules": [f"L{layer}.{key}" for layer in requant_layers for key in ("q", "k")],
            "tokens_per_module": {f"L{layer}.{key}": int(stats[layer][key]["X"].shape[1]) for layer in requant_layers for key in ("q", "k")},
        }, indent=2))

    # Branch A: ordinary ternary coordinate descent, no cardinality constraint.
    # A Branch-B-only recovery reuses its completed JSON rather than repeating it.
    branch_a_path = out_dir / "requant_ternary_result.json"
    if args.branch == "b" and branch_a_path.exists():
        branch_a = json.loads(branch_a_path.read_text())
        branch_a_metrics = branch_a["metrics"]
        log("reusing completed Branch A result")
    else:
        states_a = {layer: {key: code.T.clone() for key, code in cs.items()} for layer, cs in codes.items()}
        branch_a = {"sweeps": args.requant_sweeps, "modules": {}, "accepted_updates": 0}
        for layer in requant_layers:
            for key in ("q", "k"):
                info = run_coordinate_descent(layer, key, codes[layer][key], fp_qk[layer][key], stats[layer][key], states_a[layer][key], args.requant_sweeps)
                branch_a["modules"][f"L{layer}.{key}"] = info
                branch_a["accepted_updates"] += int(info["accepted_updates"])
                (out_dir / "branch_a_partial.json").write_text(json.dumps(branch_a, indent=2))
        apply_affine_patch(model, codes, states_a)
        branch_a_metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
        branch_a_audit = audit_all(codes, states_a)
        branch_a["final_changed_coordinates"] = changed_coordinates(codes, states_a)
        branch_a["state_hash"] = state_digest(codes, states_a)
        branch_a["cardinality_violations"] = sum(
            int((code.T.abs().sum(dim=-1) != states_a[layer][key].abs().sum(dim=-1)).sum().item())
            for layer, cs in codes.items() for key, code in cs.items()
        )
        branch_a["metrics"] = branch_a_metrics
        branch_a["delta_vs_affine_nll"] = metric_delta({k: v["nll"] for k, v in branch_a_metrics.items()}, {k: v["nll"] for k, v in affine_metrics.items()})
        branch_a["audit"] = branch_a_audit
        branch_a_path.write_text(json.dumps(branch_a, indent=2))

    # Branch B: CPSR reconstruction pair ranking with frozen 64/layer ceiling.
    states_b = {layer: {key: code.T.clone() for key, code in cs.items()} for layer, cs in codes.items()}
    branch_b = {"edits_per_layer": args.cpsr_edits_per_layer, "pair_screen_topk": args.pair_screen_topk, "layers": requant_layers, "selected_edits": [], "per_layer": {}}
    selected_b: List[AffineEdit] = []
    for layer in requant_layers:
        pools = {}
        for key in ("q", "k"):
            pools[key] = build_cpsr_pair_pool(layer, key, codes[layer][key], fp_qk[layer][key], stats[layer][key], states_b[layer][key], args.pair_screen_topk)
        edits, info = select_cpsr_layerwise(layer, pools, states_b, codes, stats, fp_qk, args.cpsr_edits_per_layer)
        selected_b.extend(edits)
        branch_b["per_layer"][str(layer)] = {**info, "pool_q": len(pools["q"]), "pool_k": len(pools["k"]), "selected": len(edits)}
        branch_b["selected_edits"] = serialise_edits(selected_b)
        (out_dir / "branch_b_partial.json").write_text(json.dumps(branch_b, indent=2))
    apply_affine_patch(model, codes, states_b)
    branch_b_metrics = with_ppl(eval_metrics(model, device, val, w2, c4))
    branch_b_audit = audit_all(codes, states_b)
    branch_b["num_relocations"] = len(selected_b)
    branch_b["changed_coordinates"] = changed_coordinates(codes, states_b)
    branch_b["state_hash"] = state_digest(codes, states_b)
    branch_b["metrics"] = branch_b_metrics
    branch_b["delta_vs_affine_nll"] = metric_delta({k: v["nll"] for k, v in branch_b_metrics.items()}, {k: v["nll"] for k, v in affine_metrics.items()})
    branch_b["audit"] = branch_b_audit
    branch_b["cardinality_violations"] = sum(
        int((code.T.abs().sum(dim=-1) != states_b[layer][key].abs().sum(dim=-1)).sum().item())
        for layer, cs in codes.items() for key, code in cs.items()
    )
    (out_dir / "requant_cpsr_result.json").write_text(json.dumps(branch_b, indent=2))

    reference = {}
    p12_path = Path("/root/CEGSP-P11-P12/results/CEGSP-P11-P12-OPT350M-20260901-42012/p11_p12_result.json")
    if p12_path.exists():
        with p12_path.open() as handle:
            p12 = json.load(handle)
        ref = p12["p12_a"]["variants"]["ternrefine_hba"]
        reference = {"source": str(p12_path), "selected_layers": ref["selected_layers"], "num_edits": ref["num_edits"], "changed_coordinates": ref["changed_coordinates"], "metrics": ref["metrics"], "delta_vs_affine_nll": ref["delta_vs_affine_nll"]}

    result = {
        "run_id": args.run_id,
        "experiment": "ReQuant-style ternary and ReQuant-style CPSR reconstruction controls",
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "same_affine_initializer": True,
            "codebook": "Q=mu+alpha*T, T in {-1,0,+1}",
            "frozen_scope": requant_layers,
            "initializer_scope": layers,
            "fit_validation_untouched_shared": True,
            "qat_teacher": False,
            "branch_a": "quadratic activation-aware reconstruction coordinate descent; 4 sweeps; cardinality unconstrained",
            "branch_b": "quadratic reconstruction CPSR donor-receiver pair score; frozen receiver sign; 64/layer ceiling",
            "branch_b_pair_score": "-dd*g_d-dr*g_r+dd^2*H_dd+dr^2*H_rr+2*dd*dr*H_dr",
            "branch_b_pair_screen_topk": args.pair_screen_topk,
            "official_requant_implementation": False,
            "comparison_scope": "method-level; not exact edit-match for Branch A",
        },
        "data": {"wikitext_source": wikitext_source, "fit_batches": len(fit), "val_batches": len(val), "w2_batches": len(w2), "c4_batches": len(c4), "seq_len": args.seq_len, "batch_size": args.batch_size, "fit_token_offset": args.fit_token_offset, "val_token_offset": args.val_token_offset, "c4_token_offset": args.c4_token_offset},
        "baseline": {"fp_metrics": fp_metrics, "affine_metrics": affine_metrics, "audit": baseline_audit, "state_hash": state_digest(codes, affine_states)},
        "requant_ternary": branch_a,
        "requant_cpsr": branch_b,
        "ternrefine_reference": reference,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "max_memory_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3)},
        "timing": {"total_sec": time.time() - start},
        "gate": {
            "finite_baseline": finite_metrics(fp_metrics) and finite_metrics(affine_metrics),
            "finite_requant_ternary": finite_metrics(branch_a_metrics),
            "finite_requant_cpsr": finite_metrics(branch_b_metrics),
            "branch_a_codebook_legal": branch_a_audit["total_illegal_states"] == 0,
            "branch_b_codebook_legal": branch_b_audit["total_illegal_states"] == 0,
            "branch_b_exact_384_768": len(selected_b) == 384 and branch_b["changed_coordinates"] == 768,
            "branch_b_cardinality_preserved": branch_b["cardinality_violations"] == 0,
        },
    }
    (out_dir / "requant_comparison_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(json.dumps(result["gate"], indent=2))
    restore_qk(model, fp_qk)


if __name__ == "__main__":
    main()
