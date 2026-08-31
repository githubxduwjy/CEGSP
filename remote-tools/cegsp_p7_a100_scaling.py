#!/usr/bin/env python3
"""P7: A100 smoke and 7B/8B frozen-canonical affine CEGSP scaling.

This script keeps the P5-B affine CEGSP rule intact while replacing the
OPT-only adapter with a generic Q/K projection adapter for OPT, Llama-family,
and Qwen-family decoder blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


@dataclass
class ProjectionRef:
    module: torch.nn.Module
    name: str


@dataclass
class AffineCode:
    mu: torch.Tensor
    alpha: torch.Tensor
    T: torch.Tensor
    valid: torch.Tensor
    original_shape: Tuple[int, int]
    group_size: int
    fp_padded: torch.Tensor


@dataclass(frozen=True)
class AffineEdit:
    layer: int
    key: str
    row: int
    block: int
    donor: int
    receiver: int
    donor_sign: int
    receiver_sign: int
    score: float


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_ppl(nll: float) -> float:
    try:
        return float(math.exp(float(nll)))
    except OverflowError:
        return float("inf")


def with_ppl(metrics: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    return {k: {"nll": float(v), "ppl": safe_ppl(float(v))} for k, v in metrics.items()}


def get_decoder_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "decoder"):
        return model.model.decoder.layers
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    raise RuntimeError(f"unsupported architecture: {getattr(model.config, 'model_type', None)}")


def target_qk(model: torch.nn.Module, layer_idx: int) -> Dict[str, ProjectionRef]:
    layer = get_decoder_layers(model)[layer_idx]
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    if attn is None:
        raise RuntimeError(f"layer {layer_idx} has no attention module")
    if hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
        return {"q": ProjectionRef(attn.q_proj, "q_proj"), "k": ProjectionRef(attn.k_proj, "k_proj")}
    if hasattr(attn, "query_key_value"):
        fused = attn.query_key_value
        hidden = int(model.config.hidden_size)
        if int(fused.out_features) != 3 * hidden:
            raise RuntimeError("fused query_key_value layout is not q,k,v by hidden-size rows")
        raise RuntimeError("fused QKV layers are not enabled in P7 scaling script")
    raise RuntimeError(f"layer {layer_idx} has no recognized q/k projection")


def set_weight(module: torch.nn.Module, weight: torch.Tensor) -> None:
    module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))


def pad_columns(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    blocks = (cols + group_size - 1) // group_size
    padded_cols = blocks * group_size
    padded = torch.zeros((rows, padded_cols), dtype=torch.float32)
    padded[:, :cols] = weight.detach().float().cpu()
    valid = torch.zeros((rows, padded_cols), dtype=torch.bool)
    valid[:, :cols] = True
    return padded.view(rows, blocks, group_size), valid.view(rows, blocks, group_size)


def make_affine_code(weight: torch.Tensor, group_size: int, threshold_factor: float) -> AffineCode:
    padded, valid = pad_columns(weight, group_size)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1).float()
    mu = (padded * valid.float()).sum(dim=-1, keepdim=True) / count
    centered = (padded - mu) * valid.float()
    threshold = threshold_factor * centered.abs().sum(dim=-1, keepdim=True) / count
    T = torch.zeros_like(padded, dtype=torch.int8)
    T[(centered > threshold) & valid] = 1
    T[(centered < -threshold) & valid] = -1
    active = T.abs().float()
    denom = active.sum(dim=-1, keepdim=True)
    alpha_raw = (centered * T.float()).sum(dim=-1, keepdim=True)
    alpha = torch.where(denom > 0, alpha_raw / denom.clamp_min(1.0), torch.zeros_like(alpha_raw))
    return AffineCode(mu, alpha, T, valid, tuple(weight.shape), group_size, padded)


def affine_weight(code: AffineCode, T: torch.Tensor | None = None) -> torch.Tensor:
    state = code.T if T is None else T
    q = code.mu + code.alpha * state.float()
    rows, cols = code.original_shape
    return q.view(rows, -1)[:, :cols].contiguous()


def codebook_audit(code: AffineCode, T: torch.Tensor | None = None) -> Dict[str, float]:
    state = code.T if T is None else T
    q = code.mu + code.alpha * state.float()
    residual = torch.minimum(
        torch.minimum((q - (code.mu - code.alpha)).abs(), (q - code.mu).abs()),
        (q - (code.mu + code.alpha)).abs(),
    )
    illegal = ((state < -1) | (state > 1) | (~code.valid & (state != 0))).sum()
    valid_residual = residual[code.valid]
    return {
        "num_illegal_states": int(illegal.item()),
        "max_codebook_residual": float(valid_residual.max().item()) if valid_residual.numel() else 0.0,
        "active_support": int((state.abs()[code.valid] > 0).sum().item()),
    }


def audit_all(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]] | None = None) -> Dict[str, float]:
    total_illegal = 0
    max_residual = 0.0
    active_support = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            state = None if states is None else states[layer][key]
            row = codebook_audit(code, state)
            total_illegal += int(row["num_illegal_states"])
            max_residual = max(max_residual, float(row["max_codebook_residual"]))
            active_support += int(row["active_support"])
    return {
        "total_illegal_states": total_illegal,
        "max_codebook_residual": max_residual,
        "active_support": active_support,
    }


def snapshot_qk(model: torch.nn.Module, layers: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        refs = target_qk(model, layer)
        out[layer] = {key: ref.module.weight.detach().float().cpu().clone() for key, ref in refs.items()}
    return out


def restore_qk(model: torch.nn.Module, weights: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer, layer_weights in weights.items():
        refs = target_qk(model, layer)
        for key, weight in layer_weights.items():
            set_weight(refs[key].module, weight)


def apply_affine_patch(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    states: Dict[int, Dict[str, torch.Tensor]] | None = None,
) -> None:
    for layer, layer_codes in codes.items():
        refs = target_qk(model, layer)
        for key, code in layer_codes.items():
            T = None if states is None else states[layer][key]
            set_weight(refs[key].module, affine_weight(code, T))


def tokenize_text(tokenizer, text: str, n_batches: int, batch_size: int, seq_len: int, offset: int) -> List[torch.Tensor]:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    needed = n_batches * batch_size * (seq_len + 1)
    if ids.numel() < offset + needed:
        reps = (offset + needed) // max(ids.numel(), 1) + 2
        ids = ids.repeat(reps)
    return [
        x.clone()
        for x in ids[offset : offset + needed].view(n_batches, batch_size, seq_len + 1)
    ]


def collect_texts(
    dataset_name: str,
    config: str | None,
    split: str,
    text_key: str,
    limit: int,
    streaming: bool = False,
) -> Tuple[str, str]:
    from datasets import load_dataset

    if config:
        ds = load_dataset(dataset_name, config, split=split, streaming=streaming)
    else:
        ds = load_dataset(dataset_name, split=split, streaming=streaming)
    chunks: List[str] = []
    for row in ds:
        text = str(row.get(text_key, "")).strip()
        if text:
            chunks.append(text)
        if len(chunks) >= limit:
            break
    return "\n".join(chunks), f"{dataset_name}:{config}:{split}:{text_key}:{len(chunks)}"


def build_splits(tokenizer, seq_len: int, batch_size: int, fit_batches: int, val_batches: int, w2_batches: int, c4_batches: int, offset: int):
    try:
        train_text, train_src = collect_texts("wikitext", "wikitext-2-raw-v1", "train", "text", 20000)
        valid_text, valid_src = collect_texts("wikitext", "wikitext-2-raw-v1", "validation", "text", 8000)
        source = {"wikitext_train": train_src, "wikitext_validation": valid_src}
    except Exception as exc:
        source = {"wikitext": f"deterministic-fallback:{type(exc).__name__}:{exc}"}
        train_text = ("CEGSP evaluates ternary support relocation from quantized point gradients. ") * 20000
        valid_text = ("Untouched validation text is deterministic fallback and cannot support a paper claim. ") * 10000
    c4_text = ""
    if c4_batches > 0:
        try:
            # C4 is intentionally streamed and bounded.  A non-streaming
            # validation load can materialize hundreds of shards before the
            # evaluator needs only a few thousand tokens.
            c4_text, c4_src = collect_texts(
                "allenai/c4", "en", "validation", "text", 8000, streaming=True
            )
            source["c4"] = c4_src
        except Exception as exc:
            source["c4"] = f"deterministic-fallback:{type(exc).__name__}:{exc}"
            c4_text = ("C4 fallback text is only for harness validation, not for evidence claims. ") * 10000
    else:
        source["c4"] = "skipped"
    fit = tokenize_text(tokenizer, train_text, fit_batches, batch_size, seq_len, offset)
    val = tokenize_text(tokenizer, valid_text, val_batches, batch_size, seq_len, offset)
    untouched_offset = offset + val_batches * batch_size * (seq_len + 1)
    w2 = tokenize_text(tokenizer, valid_text, w2_batches, batch_size, seq_len, untouched_offset)
    c4 = tokenize_text(tokenizer, c4_text, c4_batches, batch_size, seq_len, offset) if c4_batches > 0 else []
    return fit, val, w2, c4, source


@torch.no_grad()
def evaluate_nll(model: torch.nn.Module, device: torch.device, batches: Iterable[torch.Tensor]) -> float:
    losses: List[float] = []
    for batch in batches:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(float(loss.item()))
    return float(sum(losses) / max(len(losses), 1))


def eval_metrics(model, device, val, w2, c4) -> Dict[str, float]:
    model.eval()
    metrics = {
        "val": evaluate_nll(model, device, val),
        "wikitext2_untouched": evaluate_nll(model, device, w2),
    }
    if c4:
        metrics["c4_untouched"] = evaluate_nll(model, device, c4)
    return metrics


def collect_grads(model, fit, layers: Sequence[int], device: torch.device, grad_batches: int) -> Dict[int, Dict[str, torch.Tensor]]:
    old = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        for ref in target_qk(model, layer).values():
            ref.module.weight.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    used = min(grad_batches, len(fit))
    for batch in fit[:used]:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) / max(used, 1)
        loss.backward()
    grads: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        grads[layer] = {}
        for key, ref in target_qk(model, layer).items():
            grad = ref.module.weight.grad
            if grad is None:
                raise RuntimeError(f"missing gradient for layer={layer} key={key}")
            grads[layer][key] = grad.detach().float().cpu().clone()
    model.zero_grad(set_to_none=True)
    for name, p in model.named_parameters():
        p.requires_grad_(old[name])
    return grads


def top_candidates_for_code(layer: int, key: str, code: AffineCode, grad_2d: torch.Tensor, top_k: int) -> List[AffineEdit]:
    rows, blocks, group = code.T.shape
    grad = torch.zeros((rows, blocks, group), dtype=torch.float32)
    grad.view(rows, -1)[:, : grad_2d.shape[1]] = grad_2d.detach().float().cpu()
    active = (code.T != 0) & code.valid
    inactive = (code.T == 0) & code.valid
    alpha = code.alpha
    donor_values = alpha * grad * code.T.float()
    donor_values = donor_values.masked_fill(~active, -float("inf"))
    donor_score, donor_idx = donor_values.max(dim=-1)
    receiver_signs = torch.where((code.fp_padded - code.mu) >= 0, 1, -1).to(torch.int8)
    receiver_values = -alpha * grad * receiver_signs.float()
    receiver_values = receiver_values.masked_fill(~inactive, -float("inf"))
    receiver_score, receiver_idx = receiver_values.max(dim=-1)
    scores = donor_score + receiver_score
    ok = torch.isfinite(scores) & (code.alpha.squeeze(-1) != 0)
    flat = scores.flatten()
    ok_flat = ok.flatten()
    valid_count = int(ok_flat.sum().item())
    if valid_count == 0:
        return []
    k = min(top_k, valid_count)
    masked = flat.masked_fill(~ok_flat, -float("inf"))
    values, indices = torch.topk(masked, k=k)
    edits: List[AffineEdit] = []
    for value, flat_idx in zip(values.tolist(), indices.tolist()):
        row = int(flat_idx // blocks)
        block = int(flat_idx % blocks)
        donor = int(donor_idx[row, block].item())
        receiver = int(receiver_idx[row, block].item())
        edits.append(
            AffineEdit(
                layer=layer,
                key=key,
                row=row,
                block=block,
                donor=donor,
                receiver=receiver,
                donor_sign=int(code.T[row, block, donor].item()),
                receiver_sign=int(receiver_signs[row, block, receiver].item()),
                score=float(value),
            )
        )
    return edits


def build_top_candidates(codes, grads, layer: int, top_k_per_module: int) -> List[AffineEdit]:
    edits: List[AffineEdit] = []
    for key in ("q", "k"):
        edits.extend(top_candidates_for_code(layer, key, codes[layer][key], grads[layer][key], top_k_per_module))
    edits.sort(key=lambda e: (-e.score, e.key, e.row, e.block))
    return edits


def apply_edits(codes, edits: Sequence[AffineEdit]) -> Dict[int, Dict[str, torch.Tensor]]:
    states = {layer: {key: code.T.clone() for key, code in layer_codes.items()} for layer, layer_codes in codes.items()}
    for edit in edits:
        state = states[edit.layer][edit.key]
        state[edit.row, edit.block, edit.donor] = 0
        state[edit.row, edit.block, edit.receiver] = edit.receiver_sign
    return states


def random_edits(codes, selected_layers: Sequence[int], count_per_layer: int, seed: int) -> List[AffineEdit]:
    rng = random.Random(seed)
    out: List[AffineEdit] = []
    for layer in selected_layers:
        for _ in range(count_per_layer):
            for _attempt in range(1000):
                key = rng.choice(["q", "k"])
                code = codes[layer][key]
                rows, blocks, group = code.T.shape
                row = rng.randrange(rows)
                block = rng.randrange(blocks)
                valid = code.valid[row, block]
                active_idx = torch.where((code.T[row, block] != 0) & valid)[0].tolist()
                inactive_idx = torch.where((code.T[row, block] == 0) & valid)[0].tolist()
                if active_idx and inactive_idx:
                    donor = int(rng.choice(active_idx))
                    receiver = int(rng.choice(inactive_idx))
                    receiver_sign = 1 if float(code.fp_padded[row, block, receiver] - code.mu[row, block, 0]) >= 0 else -1
                    out.append(AffineEdit(layer, key, row, block, donor, receiver, int(code.T[row, block, donor]), receiver_sign, 0.0))
                    break
    return out


def changed_coordinates(codes, states) -> int:
    return sum(int((states[layer][key] != code.T).sum().item()) for layer, layer_codes in codes.items() for key, code in layer_codes.items())


def metric_delta(metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {key: float(metrics[key] - baseline[key]) for key in metrics}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "affine"], default="smoke")
    p.add_argument("--model", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="all")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--fit-batches", type=int, default=4)
    p.add_argument("--val-batches", type=int, default=4)
    p.add_argument("--untouched-batches", type=int, default=4)
    p.add_argument("--c4-untouched-batches", type=int, default=4)
    p.add_argument("--token-offset", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.75)
    p.add_argument("--layer-budgets", default="4,6")
    p.add_argument("--edits-per-layer", type=int, default=64)
    p.add_argument("--layer-probe-edits", type=int, default=8)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("P7 requires CUDA")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, w2, c4, data_source = build_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.c4_untouched_batches,
        args.token_offset,
    )
    log(f"loading model {args.model} dtype={args.dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    model.eval()
    total_layers = len(get_decoder_layers(model))
    layers = list(range(total_layers)) if args.layers == "all" else parse_csv_ints(args.layers)
    log(f"model_type={model.config.model_type} layers={layers} gpu={torch.cuda.get_device_name(0)}")

    fp_metrics = {"val": evaluate_nll(model, device, val[:1])}
    if args.mode == "smoke":
        grads = collect_grads(model, fit, layers, device, args.grad_batches)
        grad_norms = {str(layer): {key: float(value.norm().item()) for key, value in layer_grads.items()} for layer, layer_grads in grads.items()}
        result = {
            "run_id": args.run_id,
            "experiment": "CEGSP-P7-S0 A100 8B memory smoke",
            "status": "complete",
            "config": vars(args),
            "data_source": data_source,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "bf16": torch.cuda.is_bf16_supported(),
                "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            },
            "fp_val_one_batch": with_ppl(fp_metrics),
            "grad_norms": grad_norms,
            "finite_pass": math.isfinite(fp_metrics["val"]) and all(math.isfinite(v) for x in grad_norms.values() for v in x.values()),
            "elapsed_sec": time.time() - started,
        }
        out_path = out_dir / "p7_s0_smoke_result.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"wrote {out_path}")
        return

    fp_qk = snapshot_qk(model, layers)
    fp_metrics = eval_metrics(model, device, val, w2, c4)
    codes = {
        layer: {
            key: make_affine_code(fp_qk[layer][key], args.group_size, args.threshold_factor)
            for key in ("q", "k")
        }
        for layer in layers
    }
    baseline_audit = audit_all(codes)
    apply_affine_patch(model, codes)
    affine_metrics = eval_metrics(model, device, val, w2, c4)
    grads = collect_grads(model, fit, layers, device, args.grad_batches)

    keep_per_module = max(args.edits_per_layer * 2, args.layer_probe_edits * 2, 128)
    layer_candidates = {layer: build_top_candidates(codes, grads, layer, keep_per_module) for layer in layers}
    layer_ranking = []
    for layer in layers:
        probe = layer_candidates[layer][: args.layer_probe_edits]
        layer_ranking.append({
            "layer": layer,
            "num_kept_candidates": len(layer_candidates[layer]),
            "probe_edits": len(probe),
            "layer_score_top_probe_sum": float(sum(e.score for e in probe)),
            "top_score": float(probe[0].score) if probe else float("nan"),
        })
    layer_ranking.sort(key=lambda row: (-float(row["layer_score_top_probe_sum"]), int(row["layer"])))

    variants: Dict[str, Dict[str, object]] = {}
    for budget in parse_csv_ints(args.layer_budgets):
        selected_layers = [int(row["layer"]) for row in layer_ranking[:budget]]
        ce_edits: List[AffineEdit] = []
        for layer in selected_layers:
            ce_edits.extend(layer_candidates[layer][: args.edits_per_layer])
        ce_states = apply_edits(codes, ce_edits)
        apply_affine_patch(model, codes, ce_states)
        ce_metrics = eval_metrics(model, device, val, w2, c4)
        random_patch = random_edits(codes, selected_layers, args.edits_per_layer, args.seed + budget)
        random_states = apply_edits(codes, random_patch)
        apply_affine_patch(model, codes, random_states)
        random_metrics = eval_metrics(model, device, val, w2, c4)
        variants[f"affine_ce_top{budget}"] = {
            "selected_layers": selected_layers,
            "num_edits": len(ce_edits),
            "changed_coordinates": changed_coordinates(codes, ce_states),
            "metrics": with_ppl(ce_metrics),
            "delta_vs_affine_nll": metric_delta(ce_metrics, affine_metrics),
            "audit": audit_all(codes, ce_states),
        }
        variants[f"random_matched_top{budget}"] = {
            "selected_layers": selected_layers,
            "num_edits": len(random_patch),
            "changed_coordinates": changed_coordinates(codes, random_states),
            "metrics": with_ppl(random_metrics),
            "delta_vs_affine_nll": metric_delta(random_metrics, affine_metrics),
            "audit": audit_all(codes, random_states),
        }
        apply_affine_patch(model, codes)

    result = {
        "run_id": args.run_id,
        "experiment": "CEGSP-P7-A/B A100 frozen-canonical affine scaling",
        "status": "complete",
        "config": vars(args),
        "data_source": data_source,
        "protocol": {
            "selection_signal": "fit-split quantized-point CE gradient",
            "scope": "all decoder layers, q_proj/k_proj only",
            "teacher_or_qat": False,
            "mu_alpha_refit": False,
            "selection_uses_validation_or_untouched": False,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "fp_metrics": with_ppl(fp_metrics),
        "affine_baseline_metrics": with_ppl(affine_metrics),
        "affine_baseline_audit": baseline_audit,
        "layer_ranking": layer_ranking,
        "variants": variants,
        "elapsed_sec": time.time() - started,
    }
    out_path = out_dir / "p7_affine_scaling_result.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
