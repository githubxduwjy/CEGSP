#!/usr/bin/env python3
"""Clean-room TQG-SP support-projection validation.

This script tests the narrowed post-TDBT claim:

    Ternary zero-support structure gives quantized-point gradients a useful
    PTQ-only projection target: exchange one active ternary weight with one
    inactive zero weight while preserving the nonzero budget.

It never loads QAT checkpoints, QAT logits, QAT latent weights, or QAT state
priors.  The only backward path is a small number of gradients at the deployed
ternary point, computed on calibration batches.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MATRIX_KEYS = ("q", "k", "v", "o")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="0,7,15,23")
    p.add_argument("--operators", default="qk")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=16)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=8)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--candidate-pool", type=int, default=512)
    p.add_argument("--max-swaps", type=int, default=64)
    p.add_argument("--tau", type=float, default=1.05)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--nll-sanity", action="store_true")
    p.add_argument("--e2e-nll", action="store_true")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip().lower() for x in text.split(",") if x.strip()]


def build_wikitext_splits(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    fit_batches: int,
    val_batches: int,
    untouched_batches: int,
    fit_token_offset: int = 0,
    val_token_offset: int = 0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], str]:
    try:
        from datasets import load_dataset

        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        valid = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        train_text = "\n".join(x["text"] for x in train if x["text"].strip())
        valid_text = "\n".join(x["text"] for x in valid if x["text"].strip())
        source = "wikitext-2-raw-v1"
    except Exception as exc:
        try:
            train_text, valid_text = read_wikitext_arrow_cache()
            source = f"wikitext-2-raw-v1-arrow-cache-after-{type(exc).__name__}"
        except Exception as arrow_exc:
            # Sanity runs may still validate tensor plumbing when the remote
            # image has a broken datasets/pandas install.  These fallback runs
            # are not allowed to satisfy the Wikitext evidence gate.
            source = f"deterministic-fallback:{type(exc).__name__}:{type(arrow_exc).__name__}"
            train_text = (
                "Ternary quantization uses negative, zero, and positive states. "
                "Support transport swaps one active weight with one inactive weight. "
                "This calibration text is deterministic and is only a harness fallback. "
            ) * 3000
            valid_text = (
                "Holdout text remains disjoint from calibration text in this fallback. "
                "A real Wikitext run is required before any method claim is made. "
            ) * 1200

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
    return fit, val, untouched, source


def read_arrow_text(path: Path) -> str:
    import pyarrow as pa
    import pyarrow.ipc as ipc

    chunks: List[str] = []
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_stream(source)
        for batch in reader:
            col = batch.column("text")
            chunks.extend(x for x in col.to_pylist() if isinstance(x, str) and x.strip())
    return "\n".join(chunks)


def read_wikitext_arrow_cache() -> Tuple[str, str]:
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", str(Path.home() / ".cache/huggingface/datasets")))
    candidates = list(cache_root.glob("wikitext/wikitext-2-raw-v1/0.0.0/*"))
    for base in candidates:
        train = base / "wikitext-train.arrow"
        valid = base / "wikitext-validation.arrow"
        if train.exists() and valid.exists():
            return read_arrow_text(train), read_arrow_text(valid)
    raise FileNotFoundError(f"wikitext arrow cache not found under {cache_root}")


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, batches: List[torch.Tensor], device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for batch in batches:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError("nonfinite evaluation loss")
        losses.append(value)
    return float(np.mean(losses))


def iter_linear_modules(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            yield name, module


def direct_ternary_weight(weight: torch.Tensor, group_size: int, threshold_factor: float) -> Tuple[torch.Tensor, int]:
    out, inp = weight.shape
    pad = (group_size - inp % group_size) % group_size
    w = F.pad(weight.float(), (0, pad)).view(out, -1, group_size)
    mean_abs = w.abs().mean(dim=-1, keepdim=True)
    threshold = threshold_factor * mean_abs
    mask = w.abs() > threshold
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    alpha = (w.abs() * mask).sum(dim=-1, keepdim=True) / denom
    state = torch.where(mask, w.sign(), torch.zeros_like(w))
    q = (alpha * state).reshape(out, inp + pad)[:, :inp].to(weight.dtype)
    return q, int(state.ne(0).sum().item())


def apply_direct_ptq_local(model: torch.nn.Module, group_size: int, threshold_factor: float) -> Dict[str, int]:
    counts = {"linear_modules": 0, "weights": 0, "nonzero_states": 0}
    with torch.no_grad():
        for _, module in iter_linear_modules(model):
            q, nonzero = direct_ternary_weight(module.weight.data, group_size, threshold_factor)
            module.weight.data.copy_(q)
            counts["linear_modules"] += 1
            counts["weights"] += module.weight.numel()
            counts["nonzero_states"] += nonzero
    return counts


def get_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    return model.model.decoder.layers[layer_idx]


def target_modules(model: torch.nn.Module, layer_idx: int) -> Dict[str, torch.nn.Module]:
    attn = get_layer(model, layer_idx).self_attn
    return {
        "q": attn.q_proj,
        "k": attn.k_proj,
        "v": attn.v_proj,
        "o": attn.out_proj,
    }


@torch.no_grad()
def collect_hidden_states(
    model: torch.nn.Module,
    batches: List[torch.Tensor],
    layers: List[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    rows: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    model.eval()
    for batch in batches:
        x = batch[:, :-1].to(device)
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
        for layer in layers:
            rows[layer].append(out.hidden_states[layer].detach().float().cpu())
    return rows


@dataclass
class Code:
    fp_padded: torch.Tensor
    state: torch.Tensor
    alpha: torch.Tensor
    valid: torch.Tensor
    shape: Tuple[int, int]
    pad: int
    group_size: int


def make_code(weight: torch.Tensor, group_size: int, threshold_factor: float) -> Code:
    weight = weight.detach().float().cpu()
    out, inp = weight.shape
    pad = (group_size - inp % group_size) % group_size
    padded = F.pad(weight, (0, pad)).view(out, -1, group_size)
    valid = torch.ones(out, inp + pad, dtype=torch.bool)
    if pad:
        valid[:, inp:] = False
    valid = valid.view(out, -1, group_size)
    mean_abs = padded.abs().mean(dim=-1, keepdim=True)
    threshold = threshold_factor * mean_abs
    mask = (padded.abs() > threshold) & valid
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    alpha = (padded.abs() * mask).sum(dim=-1, keepdim=True) / denom
    state = torch.where(mask, torch.where(padded >= 0, torch.ones_like(padded), -torch.ones_like(padded)), torch.zeros_like(padded))
    return Code(
        fp_padded=padded,
        state=state,
        alpha=alpha.squeeze(-1),
        valid=valid,
        shape=(out, inp),
        pad=pad,
        group_size=group_size,
    )


def refit_alpha(code: Code, state: torch.Tensor) -> torch.Tensor:
    active = state.ne(0) & code.valid
    denom = active.sum(dim=-1).clamp_min(1)
    return (code.fp_padded.abs() * active).sum(dim=-1) / denom


def weight_from_state(code: Code, state: torch.Tensor, refit: bool) -> torch.Tensor:
    alpha = refit_alpha(code, state) if refit else code.alpha
    q = alpha.unsqueeze(-1) * state
    out, inp = code.shape
    return q.reshape(out, inp + code.pad)[:, :inp].contiguous()


def code_weight(code: Code, refit: bool = False) -> torch.Tensor:
    return weight_from_state(code, code.state, refit=refit)


def fp_weight_from_code(code: Code) -> torch.Tensor:
    out, inp = code.shape
    return code.fp_padded.reshape(out, inp + code.pad)[:, :inp].contiguous()


def sign_from_fp(code: Code, flat_idx: int) -> float:
    value = float(code.fp_padded.flatten()[flat_idx])
    return 1.0 if value >= 0 else -1.0


def x_column_energy(xs: List[torch.Tensor], inp: int) -> torch.Tensor:
    acc = torch.zeros(inp, dtype=torch.float32)
    count = 0
    for x in xs:
        flat = x.reshape(-1, x.shape[-1]).float()
        acc += (flat * flat).sum(dim=0)
        count += flat.shape[0]
    return acc / max(count, 1)


def pad_by_code(vec: torch.Tensor, code: Code) -> torch.Tensor:
    out, inp = code.shape
    padded = F.pad(vec.float().view(1, inp).expand(out, inp), (0, code.pad))
    return padded.view(out, -1, code.group_size)


def nmse(num: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return (num - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-8)


@torch.no_grad()
def precompute_targets(
    xs: List[torch.Tensor],
    fp_weights: Dict[str, torch.Tensor],
    operator: str,
    device: torch.device,
) -> List[torch.Tensor]:
    targets: List[torch.Tensor] = []
    weights = {k: v.to(device) for k, v in fp_weights.items()}
    scale = math.sqrt(float(weights["q"].shape[0]))
    for x_cpu in xs:
        x = x_cpu.to(device)
        if operator == "qk":
            q = torch.matmul(x, weights["q"].t())
            k = torch.matmul(x, weights["k"].t())
            target = torch.matmul(q, k.transpose(-1, -2)) / scale
        elif operator == "vo":
            v = torch.matmul(x, weights["v"].t())
            target = torch.matmul(v, weights["o"].t())
        else:
            raise ValueError(f"unknown operator: {operator}")
        targets.append(target.detach().float().cpu())
    return targets


@torch.no_grad()
def operator_loss(
    xs: List[torch.Tensor],
    targets: List[torch.Tensor],
    weights_cpu: Dict[str, torch.Tensor],
    operator: str,
    device: torch.device,
) -> float:
    weights = {k: v.to(device) for k, v in weights_cpu.items()}
    scale = math.sqrt(float(weights["q"].shape[0]))
    values: List[float] = []
    for x_cpu, target_cpu in zip(xs, targets):
        x = x_cpu.to(device)
        target = target_cpu.to(device)
        if operator == "qk":
            q = torch.matmul(x, weights["q"].t())
            k = torch.matmul(x, weights["k"].t())
            pred = torch.matmul(q, k.transpose(-1, -2)) / scale
        else:
            v = torch.matmul(x, weights["v"].t())
            pred = torch.matmul(v, weights["o"].t())
        values.append(float(nmse(pred.float(), target.float()).detach().cpu()))
    return float(np.mean(values))


@torch.no_grad()
def local_matrix_loss(
    xs: List[torch.Tensor],
    matrix_weight: torch.Tensor,
    fp_weight: torch.Tensor,
    device: torch.device,
) -> float:
    w = matrix_weight.to(device)
    w_fp = fp_weight.to(device)
    values: List[float] = []
    for x_cpu in xs:
        x = x_cpu.to(device)
        pred = torch.matmul(x, w.t())
        ref = torch.matmul(x, w_fp.t())
        values.append(float(nmse(pred.float(), ref.float()).detach().cpu()))
    return float(np.mean(values))


def compose_weights(codes: Dict[str, Code], states: Dict[str, torch.Tensor], refit: bool) -> Dict[str, torch.Tensor]:
    return {key: weight_from_state(codes[key], states[key], refit=refit) for key in MATRIX_KEYS}


@dataclass(frozen=True)
class Candidate:
    matrix_key: str
    donor: int
    receiver: int
    score: float


@dataclass(frozen=True)
class FlipCandidate:
    matrix_key: str
    index: int
    new_state: float
    score: float


def top_indices(values: torch.Tensor, eligible: torch.Tensor, k: int, largest: bool) -> List[int]:
    masked = torch.where(eligible, values, torch.full_like(values, -float("inf") if largest else float("inf")))
    count = int(eligible.sum().item())
    if count <= 0:
        return []
    kk = min(k, count)
    return torch.topk(masked, kk, largest=largest).indices.cpu().tolist()


def forward_candidates(
    codes: Dict[str, Code],
    xs_fit: List[torch.Tensor],
    operator: str,
    pool: int,
) -> List[Candidate]:
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    all_candidates: List[Candidate] = []
    per_side = max(8, int(math.sqrt(pool)) * 3)
    for key in keys:
        code = codes[key]
        x2 = pad_by_code(x_column_energy(xs_fit, code.shape[1]), code)
        q = code.alpha.unsqueeze(-1) * code.state
        fp = code.fp_padded
        receiver_value = code.alpha.unsqueeze(-1) * torch.where(fp >= 0, torch.ones_like(fp), -torch.ones_like(fp))
        active = code.state.ne(0) & code.valid
        inactive = code.state.eq(0) & code.valid
        donor_cost = (((0.0 - fp).pow(2) - (q - fp).pow(2)) * x2).flatten()
        receiver_gain = (((0.0 - fp).pow(2) - (receiver_value - fp).pow(2)) * x2).flatten()
        donors = top_indices(-donor_cost, active.flatten(), per_side, largest=True)
        receivers = top_indices(receiver_gain, inactive.flatten(), per_side, largest=True)
        for d in donors:
            for r in receivers:
                score = float(receiver_gain[r] - donor_cost[d])
                all_candidates.append(Candidate(key, d, r, score))
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    return all_candidates[:pool]


def forward_projection_candidates_unique(
    codes: Dict[str, Code],
    xs_fit: List[torch.Tensor],
    operator: str,
    budget: int,
) -> List[Candidate]:
    """Budget-matched forward baseline with unique donors/receivers.

    The earlier pairwise candidate list can under-use the edit budget because
    many high-ranked pairs share the same donor or receiver.  For mechanism
    controls we need a fair same-budget projection, so we independently rank
    donors and receivers, pair them once, then rank matrix-level pairs.
    """
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    rows: List[Candidate] = []
    per_key = max(1, budget)
    for key in keys:
        code = codes[key]
        x2 = pad_by_code(x_column_energy(xs_fit, code.shape[1]), code)
        q = code.alpha.unsqueeze(-1) * code.state
        fp = code.fp_padded
        receiver_value = code.alpha.unsqueeze(-1) * torch.where(fp >= 0, torch.ones_like(fp), -torch.ones_like(fp))
        active = code.state.ne(0) & code.valid
        inactive = code.state.eq(0) & code.valid
        donor_cost = (((0.0 - fp).pow(2) - (q - fp).pow(2)) * x2).flatten()
        receiver_gain = (((0.0 - fp).pow(2) - (receiver_value - fp).pow(2)) * x2).flatten()
        donors = top_indices(-donor_cost, active.flatten(), per_key, largest=True)
        receivers = top_indices(receiver_gain, inactive.flatten(), per_key, largest=True)
        for d, r in zip(donors, receivers):
            score = float(receiver_gain[r] - donor_cost[d])
            rows.append(Candidate(key, d, r, score))
    rows.sort(key=lambda c: c.score, reverse=True)
    return rows[:budget]


def gradient_for_operator(
    xs_fit: List[torch.Tensor],
    targets: List[torch.Tensor],
    base_weights: Dict[str, torch.Tensor],
    operator: str,
    device: torch.device,
    grad_batches: int,
) -> Dict[str, torch.Tensor]:
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    params = {k: base_weights[k].detach().clone().to(device).requires_grad_(True) for k in keys}
    const = {k: base_weights[k].to(device) for k in MATRIX_KEYS if k not in params}
    scale = math.sqrt(float(base_weights["q"].shape[0]))
    loss = torch.zeros((), device=device)
    used = min(grad_batches, len(xs_fit))
    for x_cpu, target_cpu in zip(xs_fit[:used], targets[:used]):
        x = x_cpu.to(device)
        target = target_cpu.to(device)
        if operator == "qk":
            q = torch.matmul(x, params["q"].t())
            k = torch.matmul(x, params["k"].t())
            pred = torch.matmul(q, k.transpose(-1, -2)) / scale
        else:
            v = torch.matmul(x, params["v"].t())
            pred = torch.matmul(v, params["o"].t())
        loss = loss + nmse(pred.float(), target.float()) / max(used, 1)
    loss.backward()
    grads = {k: params[k].grad.detach().cpu().float() for k in keys}
    for key in MATRIX_KEYS:
        if key not in grads:
            grads[key] = torch.zeros_like(base_weights[key])
    return grads


def gradient_candidates(
    codes: Dict[str, Code],
    grads: Dict[str, torch.Tensor],
    operator: str,
    pool: int,
) -> List[Candidate]:
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    all_candidates: List[Candidate] = []
    per_side = max(8, int(math.sqrt(pool)) * 3)
    for key in keys:
        code = codes[key]
        grad = F.pad(grads[key], (0, code.pad)).view_as(code.state)
        q = code.alpha.unsqueeze(-1) * code.state
        receiver_value = code.alpha.unsqueeze(-1) * torch.where(code.fp_padded >= 0, torch.ones_like(code.fp_padded), -torch.ones_like(code.fp_padded))
        active = code.state.ne(0) & code.valid
        inactive = code.state.eq(0) & code.valid
        donor_gain = (-(grad * (0.0 - q))).flatten()
        receiver_gain = (-(grad * receiver_value)).flatten()
        donors = top_indices(donor_gain, active.flatten(), per_side, largest=True)
        receivers = top_indices(receiver_gain, inactive.flatten(), per_side, largest=True)
        for d in donors:
            for r in receivers:
                score = float(donor_gain[d] + receiver_gain[r])
                all_candidates.append(Candidate(key, d, r, score))
    all_candidates.sort(key=lambda c: c.score, reverse=True)
    return all_candidates[:pool]


def gradient_projection_candidates_unique(
    codes: Dict[str, Code],
    grads: Dict[str, torch.Tensor],
    operator: str,
    budget: int,
) -> List[Candidate]:
    """Budget-matched TQG-SP candidate list with unique support exchanges."""
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    rows: List[Candidate] = []
    per_key = max(1, budget)
    for key in keys:
        code = codes[key]
        grad = F.pad(grads[key], (0, code.pad)).view_as(code.state)
        q = code.alpha.unsqueeze(-1) * code.state
        receiver_value = code.alpha.unsqueeze(-1) * torch.where(code.fp_padded >= 0, torch.ones_like(code.fp_padded), -torch.ones_like(code.fp_padded))
        active = code.state.ne(0) & code.valid
        inactive = code.state.eq(0) & code.valid
        donor_gain = (-(grad * (0.0 - q))).flatten()
        receiver_gain = (-(grad * receiver_value)).flatten()
        donors = top_indices(donor_gain, active.flatten(), per_key, largest=True)
        receivers = top_indices(receiver_gain, inactive.flatten(), per_key, largest=True)
        for d, r in zip(donors, receivers):
            rows.append(Candidate(key, d, r, float(donor_gain[d] + receiver_gain[r])))
    rows.sort(key=lambda c: c.score, reverse=True)
    return rows[:budget]


def gradient_signflip_candidates(
    codes: Dict[str, Code],
    grads: Dict[str, torch.Tensor],
    operator: str,
    pool: int,
) -> List[FlipCandidate]:
    """Gradient-ranked control that does not use zero support.

    This is deliberately a hard control: it can only flip +alpha <-> -alpha on
    already-active ternary weights.  If it matches support swaps, the mechanism
    is likely generic low-bit gradient editing rather than ternary zero-support
    projection.
    """
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    rows: List[FlipCandidate] = []
    per_key = max(8, pool // max(len(keys), 1))
    for key in keys:
        code = codes[key]
        grad = F.pad(grads[key], (0, code.pad)).view_as(code.state)
        q = code.alpha.unsqueeze(-1) * code.state
        delta = -2.0 * q
        score = (-(grad * delta)).flatten()
        active = (code.state.ne(0) & code.valid).flatten()
        for idx in top_indices(score, active, per_key, largest=True):
            old = float(code.state.flatten()[idx])
            rows.append(FlipCandidate(key, idx, -old, float(score[idx])))
    rows.sort(key=lambda c: c.score, reverse=True)
    return rows[:pool]


def random_support_candidates(
    codes: Dict[str, Code],
    operator: str,
    pool: int,
    seed: int,
) -> List[Candidate]:
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    rng = np.random.default_rng(seed)
    rows: List[Candidate] = []
    for key in keys:
        code = codes[key]
        active = torch.nonzero((code.state.ne(0) & code.valid).flatten(), as_tuple=False).flatten().cpu().numpy()
        inactive = torch.nonzero((code.state.eq(0) & code.valid).flatten(), as_tuple=False).flatten().cpu().numpy()
        if active.size == 0 or inactive.size == 0:
            continue
        n = min(max(1, pool // max(len(keys), 1)), active.size, inactive.size)
        donors = rng.choice(active, size=n, replace=False)
        receivers = rng.choice(inactive, size=n, replace=False)
        for d, r in zip(donors.tolist(), receivers.tolist()):
            rows.append(Candidate(key, int(d), int(r), float(rng.random())))
    rng.shuffle(rows)
    return rows[:pool]


def valid_candidate(states: Dict[str, torch.Tensor], candidate: Candidate) -> bool:
    state = states[candidate.matrix_key].flatten()
    return bool(state[candidate.donor] != 0 and state[candidate.receiver] == 0)


def apply_candidate(
    states: Dict[str, torch.Tensor],
    codes: Dict[str, Code],
    candidate: Candidate,
    endpoint: bool,
) -> Dict[str, torch.Tensor]:
    new_states = {k: v.clone() for k, v in states.items()}
    flat = new_states[candidate.matrix_key].flatten()
    flat[candidate.donor] = 0.0
    if endpoint:
        flat[candidate.receiver] = sign_from_fp(codes[candidate.matrix_key], candidate.receiver)
    return new_states


def valid_flip_candidate(states: Dict[str, torch.Tensor], candidate: FlipCandidate) -> bool:
    state = states[candidate.matrix_key].flatten()
    return bool(state[candidate.index] != 0 and float(state[candidate.index]) != float(candidate.new_state))


def apply_flip_candidate(
    states: Dict[str, torch.Tensor],
    candidate: FlipCandidate,
) -> Dict[str, torch.Tensor]:
    new_states = {k: v.clone() for k, v in states.items()}
    flat = new_states[candidate.matrix_key].flatten()
    flat[candidate.index] = float(candidate.new_state)
    return new_states


def evaluate_state(
    states: Dict[str, torch.Tensor],
    codes: Dict[str, Code],
    xs: List[torch.Tensor],
    targets: List[torch.Tensor],
    operator: str,
    device: torch.device,
    refit: bool,
) -> float:
    return operator_loss(xs, targets, compose_weights(codes, states, refit=refit), operator, device)


def run_one_shot(
    states0: Dict[str, torch.Tensor],
    codes: Dict[str, Code],
    candidates: List[Candidate],
    max_swaps: int,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, object]]]:
    states = {k: v.clone() for k, v in states0.items()}
    rows: List[Dict[str, object]] = []
    for candidate in candidates:
        if len(rows) >= max_swaps:
            break
        if not valid_candidate(states, candidate):
            continue
        states = apply_candidate(states, codes, candidate, endpoint=True)
        rows.append({"candidate": candidate.__dict__, "accepted": True})
    return states, rows


def run_one_shot_flips(
    states0: Dict[str, torch.Tensor],
    candidates: List[FlipCandidate],
    max_flips: int,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, object]]]:
    states = {k: v.clone() for k, v in states0.items()}
    rows: List[Dict[str, object]] = []
    for candidate in candidates:
        if len(rows) >= max_flips:
            break
        if not valid_flip_candidate(states, candidate):
            continue
        states = apply_flip_candidate(states, candidate)
        rows.append({"candidate": candidate.__dict__, "accepted": True})
    return states, rows


def run_greedy(
    states0: Dict[str, torch.Tensor],
    codes: Dict[str, Code],
    candidates: List[Candidate],
    xs_fit: List[torch.Tensor],
    targets_fit: List[torch.Tensor],
    operator: str,
    device: torch.device,
    max_swaps: int,
    eval_candidates: int,
    tau: float,
    use_barrier: bool,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, object]]]:
    states = {k: v.clone() for k, v in states0.items()}
    current = evaluate_state(states, codes, xs_fit, targets_fit, operator, device, refit=False)
    base_local = {
        key: local_matrix_loss(xs_fit, code_weight(codes[key]), fp_weight_from_code(codes[key]), device)
        for key in MATRIX_KEYS
    }
    rows: List[Dict[str, object]] = []
    used: set[Tuple[str, int, int]] = set()
    for step in range(max_swaps):
        best = None
        checked = 0
        for candidate in candidates:
            ident = (candidate.matrix_key, candidate.donor, candidate.receiver)
            if ident in used or not valid_candidate(states, candidate):
                continue
            checked += 1
            if checked > eval_candidates:
                break
            mid_states = apply_candidate(states, codes, candidate, endpoint=False)
            end_states = apply_candidate(states, codes, candidate, endpoint=True)
            if use_barrier:
                key = candidate.matrix_key
                mid_w = weight_from_state(codes[key], mid_states[key], refit=False)
                end_w = weight_from_state(codes[key], end_states[key], refit=False)
                mid_local = local_matrix_loss(xs_fit, mid_w, fp_weight_from_code(codes[key]), device)
                end_local = local_matrix_loss(xs_fit, end_w, fp_weight_from_code(codes[key]), device)
                if mid_local > tau * base_local[key] or end_local > tau * base_local[key]:
                    continue
                mid_loss = evaluate_state(mid_states, codes, xs_fit, targets_fit, operator, device, refit=False)
            else:
                mid_loss = None
                mid_local = None
                end_local = None
            end_loss = evaluate_state(end_states, codes, xs_fit, targets_fit, operator, device, refit=False)
            if end_loss < current and (best is None or end_loss < best["end_loss"]):
                best = {
                    "candidate": candidate,
                    "states": end_states,
                    "end_loss": end_loss,
                    "mid_loss": mid_loss,
                    "mid_local": mid_local,
                    "end_local": end_local,
                }
        if best is None:
            break
        states = best["states"]
        used.add((best["candidate"].matrix_key, best["candidate"].donor, best["candidate"].receiver))
        rows.append(
            {
                "step": step,
                "candidate": best["candidate"].__dict__,
                "previous_fit_loss": current,
                "new_fit_loss": best["end_loss"],
                "mid_fit_loss": best["mid_loss"],
                "mid_local_loss": best["mid_local"],
                "end_local_loss": best["end_local"],
            }
        )
        current = float(best["end_loss"])
    return states, rows


def summarize_variant(
    name: str,
    states: Dict[str, torch.Tensor],
    trace: List[Dict[str, object]],
    codes: Dict[str, Code],
    xs_by_split: Dict[str, List[torch.Tensor]],
    targets_by_split: Dict[str, List[torch.Tensor]],
    operator: str,
    device: torch.device,
) -> Dict[str, object]:
    metrics = {
        split: evaluate_state(states, codes, xs, targets_by_split[split], operator, device, refit=True)
        for split, xs in xs_by_split.items()
    }
    return {
        "variant": name,
        "accepted_edits": len(trace),
        "accepted_swaps": len(trace),
        "metrics_refit": metrics,
        "trace": trace,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    timing: Dict[str, float] = {}
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_csv_ints(args.layers)
    operators = parse_csv_strings(args.operators)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    log(f"loading {args.model} dtype={args.dtype} gpu={torch.cuda.get_device_name(0)}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, data_source = build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
    )
    timing["load_tokenizer_and_data_sec"] = time.time() - t0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    timing["load_model_sec"] = time.time() - t0

    t0 = time.time()
    fp_weights_by_layer = {
        layer: {key: module.weight.detach().float().cpu() for key, module in target_modules(model, layer).items()}
        for layer in layers
    }
    hidden = {
        "fit": collect_hidden_states(model, fit, layers, device),
        "val": collect_hidden_states(model, val, layers, device),
        "untouched_w": collect_hidden_states(model, untouched, layers, device),
    }
    timing["collect_fp_hidden_sec"] = time.time() - t0

    fp_nll = None
    if args.nll_sanity or args.e2e_nll:
        t0 = time.time()
        fp_nll = {
            "val": evaluate_nll(model, val, device),
            "untouched_w": evaluate_nll(model, untouched, device),
        }
        timing["fp_nll_eval_sec"] = time.time() - t0

    results: List[Dict[str, object]] = []
    codes_by_layer: Dict[int, Dict[str, Code]] = {}
    states_for_e2e: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
    proxy_t0 = time.time()
    for layer in layers:
        log(f"layer={layer} preparing codes and targets")
        fp_weights = fp_weights_by_layer[layer]
        codes = {key: make_code(fp_weights[key], args.group_size, args.threshold_factor) for key in MATRIX_KEYS}
        codes_by_layer[layer] = codes
        base_states = {key: code.state.clone() for key, code in codes.items()}
        xs_by_split = {
            "fit": hidden["fit"][layer],
            "val": hidden["val"][layer],
            "untouched_w": hidden["untouched_w"][layer],
        }
        for operator in operators:
            log(f"layer={layer} operator={operator} running variants")
            targets_by_split = {
                split: precompute_targets(xs, fp_weights, operator, device)
                for split, xs in xs_by_split.items()
            }
            base_metrics = {
                split: evaluate_state(base_states, codes, xs, targets_by_split[split], operator, device, refit=True)
                for split, xs in xs_by_split.items()
            }
            base_weights = compose_weights(codes, base_states, refit=False)
            grads = gradient_for_operator(
                xs_by_split["fit"],
                targets_by_split["fit"],
                base_weights,
                operator,
                device,
                args.grad_batches,
            )
            f_candidates = forward_projection_candidates_unique(codes, xs_by_split["fit"], operator, args.max_swaps)
            g_candidates = gradient_projection_candidates_unique(codes, grads, operator, args.max_swaps)
            flip_candidates = gradient_signflip_candidates(codes, grads, operator, args.candidate_pool)
            rand_candidates = random_support_candidates(codes, operator, args.candidate_pool, args.seed + layer * 17)

            variants: List[Dict[str, object]] = []
            random_states, random_trace = run_one_shot(base_states, codes, rand_candidates, args.max_swaps)
            forward_states, forward_trace = run_one_shot(base_states, codes, f_candidates, args.max_swaps)
            tqgsp_states, tqgsp_trace = run_one_shot(base_states, codes, g_candidates, args.max_swaps)
            signflip_states, signflip_trace = run_one_shot_flips(base_states, flip_candidates, args.max_swaps)

            variant_state_rows = [
                ("support-random", random_states, random_trace),
                ("support-forward", forward_states, forward_trace),
                ("TQGSP-support-G", tqgsp_states, tqgsp_trace),
                ("NZ-signflip-G", signflip_states, signflip_trace),
            ]
            for name, states, trace in variant_state_rows:
                variants.append(summarize_variant(name, states, trace, codes, xs_by_split, targets_by_split, operator, device))
                if args.e2e_nll:
                    keys = ("q", "k") if operator == "qk" else ("v", "o")
                    states_for_e2e.setdefault(name, {}).setdefault(layer, {})
                    for key in keys:
                        states_for_e2e[name][layer][key] = states[key].clone()

            results.append(
                {
                    "layer": layer,
                    "operator": operator,
                    "base_metrics_refit": base_metrics,
                    "candidate_counts": {
                        "random_support": len(rand_candidates),
                        "forward_support": len(f_candidates),
                        "gradient_support": len(g_candidates),
                        "gradient_signflip": len(flip_candidates),
                    },
                    "variants": variants,
                }
            )
            log(f"layer={layer} operator={operator} base_val={base_metrics['val']:.6g}")
    timing["proxy_validation_sec"] = time.time() - proxy_t0

    nll_sanity = None
    e2e_nll = None
    if args.nll_sanity or args.e2e_nll:
        log("running direct PTQ and end-to-end NLL checks")
        t0 = time.time()
        quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
        timing["direct_ptq_apply_sec"] = time.time() - t0

        selected_direct_weights: Dict[str, torch.Tensor] = {}
        for layer in layers:
            modules = target_modules(model, layer)
            for key, module in modules.items():
                selected_direct_weights[f"{layer}:{key}"] = module.weight.detach().float().cpu().clone()

        t0 = time.time()
        direct_ptq_nll = {
            "val": evaluate_nll(model, val, device),
            "untouched_w": evaluate_nll(model, untouched, device),
        }
        timing["direct_ptq_nll_eval_sec"] = time.time() - t0
        nll_sanity = {
            "fp": fp_nll,
            "direct_ptq": direct_ptq_nll,
            "quant_counts": quant_counts,
        }

        if args.e2e_nll:
            e2e_nll = {"direct-ternary": direct_ptq_nll, "variants": {}}
            for variant_name, by_layer in states_for_e2e.items():
                with torch.no_grad():
                    for layer in layers:
                        modules = target_modules(model, layer)
                        for key, module in modules.items():
                            module.weight.data.copy_(
                                selected_direct_weights[f"{layer}:{key}"].to(
                                    device=module.weight.device,
                                    dtype=module.weight.dtype,
                                )
                            )
                    for layer, state_by_key in by_layer.items():
                        modules = target_modules(model, layer)
                        for key, state in state_by_key.items():
                            module = modules[key]
                            patched = weight_from_state(codes_by_layer[layer][key], state, refit=True)
                            module.weight.data.copy_(patched.to(device=module.weight.device, dtype=module.weight.dtype))
                t0 = time.time()
                scores = {
                    "val": evaluate_nll(model, val, device),
                    "untouched_w": evaluate_nll(model, untouched, device),
                }
                timing[f"e2e_nll_eval_{variant_name}_sec"] = time.time() - t0
                e2e_nll["variants"][variant_name] = {
                    **scores,
                    "delta_vs_direct_val": scores["val"] - direct_ptq_nll["val"],
                    "delta_vs_direct_untouched_w": scores["untouched_w"] - direct_ptq_nll["untouched_w"],
                }

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "config": vars(args),
        "validation_version": {
            "name": "TQGSP-01B-budget-matched",
            "primary_claim": "ternary zero-support swaps guided by quantized-point gradients improve PTQ without QAT artifacts",
            "anti_claims": [
                "gain is just random support movement",
                "gain is just forward salience",
                "gain is generic nonzero sign-flip editing rather than ternary zero-support projection",
                "operator-proxy gain does not transfer to end-to-end language-model NLL",
                "cost approaches QAT rather than PTQ post-processing",
            ],
            "gate": {
                "mechanism": "TQGSP-support-G should beat support-forward, support-random, and NZ-signflip-G on held-out operator NMSE in most tested layer/operator pairs",
                "transfer": "patched end-to-end NLL should not degrade on untouched split; any improvement is positive evidence but not required for this first validation",
                "cost": "report wall-clock breakdown; final method should target <=3x a calibrated PTQ-style pass and remain far below QAT",
            },
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "clean_room_invariants": {
            "uses_qat_checkpoint": False,
            "uses_qat_logits": False,
            "uses_qat_latent_weights": False,
            "uses_qat_state_prior": False,
            "uses_quantized_point_operator_gradient": True,
            "uses_path_barrier_or_tdbt_transport": False,
        },
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched),
            "split": "Wikitext-2 train fit / validation val and later validation untouched",
            "c4": "not_requested_in_this_run",
        },
        "metric_definition": {
            "qk": "NMSE((X Wq^T)(X Wk^T)^T / sqrt(d), FP reference)",
            "vo": "NMSE((X Wv^T) Wo^T, FP reference)",
            "support_swap": "one active ternary state is moved to zero and one inactive zero state becomes sign(fp)*alpha; nonzero budget is preserved",
            "NZ_signflip": "nonzero-only +alpha <-> -alpha control; does not use the ternary zero state as a transport/projection target",
            "final_alpha": "one refit pass for reported metrics",
        },
        "nll_sanity": nll_sanity,
        "e2e_nll": e2e_nll,
        "timing": timing,
        "results": results,
        "status": "complete",
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
