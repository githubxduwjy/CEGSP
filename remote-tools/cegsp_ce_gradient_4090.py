#!/usr/bin/env python3
"""CEGSP-01A: CE-gradient support projection at ternary PTQ weights.

Strict PTQ: no QAT teacher, no QAT checkpoint/logits/latent weights, no
optimizer update.  We compute a small CE gradient at the deployed ternary point
and use it to propose discrete ternary edits.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tqgsp_support_projection_4090 import (
    MATRIX_KEYS,
    apply_direct_ptq_local,
    build_wikitext_splits,
    evaluate_nll,
    gradient_projection_candidates_unique,
    gradient_signflip_candidates,
    log,
    make_code,
    parse_csv_ints,
    random_support_candidates,
    run_one_shot,
    run_one_shot_flips,
    weight_from_state,
    FlipCandidate,
)


@dataclass(frozen=True)
class ProjectionRef:
    """A named Q/K/V/O projection, possibly a slice of a fused QKV Linear."""

    module: torch.nn.Module
    start: int
    end: int
    name: str

    @property
    def weight(self) -> torch.Tensor:
        return self.module.weight[self.start : self.end]


def _architecture_name(model: torch.nn.Module) -> str:
    model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
    if model_type == "opt" or (
        hasattr(model, "model") and hasattr(model.model, "decoder")
    ):
        return "opt"
    if model_type in {"gpt_neox", "pythia"} or hasattr(model, "gpt_neox"):
        return "gpt_neox"
    raise RuntimeError(
        "Unsupported architecture for CEGSP adapter: "
        f"model_type={model_type!r}; supported families are OPT and GPT-NeoX/Pythia"
    )


def get_layer(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    architecture = _architecture_name(model)
    if architecture == "opt":
        return model.model.decoder.layers[layer_idx]
    return model.gpt_neox.layers[layer_idx]


def target_modules(model: torch.nn.Module, layer_idx: int) -> Dict[str, ProjectionRef]:
    """Return Q/K/V/O references for separate or fused attention layouts.

    GPT-NeoX stores Q, K and V consecutively in one Linear. The adapter exposes
    row slices while retaining the original fused parameter for forward/backward
    execution, so direct PTQ and CE-gradient editing remain unchanged.
    """

    architecture = _architecture_name(model)
    layer = get_layer(model, layer_idx)
    if architecture == "opt":
        attn = layer.self_attn
        return {
            "q": ProjectionRef(attn.q_proj, 0, attn.q_proj.out_features, "q_proj"),
            "k": ProjectionRef(attn.k_proj, 0, attn.k_proj.out_features, "k_proj"),
            "v": ProjectionRef(attn.v_proj, 0, attn.v_proj.out_features, "v_proj"),
            "o": ProjectionRef(attn.out_proj, 0, attn.out_proj.out_features, "out_proj"),
        }

    attention = layer.attention
    fused = attention.query_key_value
    hidden = int(getattr(model.config, "hidden_size"))
    if int(fused.out_features) != 3 * hidden:
        raise RuntimeError(
            "GPT-NeoX adapter expected query_key_value.out_features == 3 * hidden_size; "
            f"got {fused.out_features} vs {3 * hidden}"
        )
    return {
        "q": ProjectionRef(fused, 0, hidden, "query_key_value[q]"),
        "k": ProjectionRef(fused, hidden, 2 * hidden, "query_key_value[k]"),
        "v": ProjectionRef(fused, 2 * hidden, 3 * hidden, "query_key_value[v]"),
        "o": ProjectionRef(attention.dense, 0, attention.dense.out_features, "dense"),
    }


def projection_weight(ref: ProjectionRef) -> torch.Tensor:
    return ref.weight


def set_projection_weight(ref: ProjectionRef, weight: torch.Tensor) -> None:
    with torch.no_grad():
        ref.module.weight.data[ref.start : ref.end].copy_(
            weight.to(device=ref.module.weight.device, dtype=ref.module.weight.dtype)
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--run-id", required=True)
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=8)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--max-edits", type=int, default=64)
    p.add_argument("--grad-batches", type=int, default=1)
    p.add_argument("--support-topk", type=int, default=6)
    p.add_argument("--signflip-topk", type=int, default=6)
    p.add_argument("--k-sweep", default="")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--fit-token-offset", type=int, default=0)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-untouched-batches", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--random-control-repeats", type=int, default=0)
    p.add_argument("--cloze-examples", type=int, default=0)
    p.add_argument("--cloze-patch-prefixes", default="ksweep-joint")
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    return p.parse_args()


def build_c4_untouched_batches(
    tokenizer: AutoTokenizer,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    token_offset: int,
) -> Optional[List[torch.Tensor]]:
    if n_batches <= 0:
        return None
    from datasets import load_dataset

    needed = token_offset + n_batches * batch_size * (seq_len + 1)
    texts: List[str] = []
    token_count = 0
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        # Cheap over-estimate progress check; final slicing below is exact.
        token_count += max(1, len(text) // 4)
        if token_count >= needed * 2:
            ids = tokenizer("\n".join(texts), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if ids.numel() >= needed:
                break
    ids = tokenizer("\n".join(texts), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() < needed:
        raise RuntimeError(f"not enough C4 validation tokens: have={ids.numel()} need={needed}")
    sliced = ids[token_offset : token_offset + n_batches * batch_size * (seq_len + 1)]
    return [x.clone() for x in sliced.view(n_batches, batch_size, seq_len + 1)]


def build_lambada_cloze_examples(
    tokenizer: AutoTokenizer,
    seq_len: int,
    n_examples: int,
) -> Dict[str, object]:
    if n_examples <= 0:
        return {"source": "disabled", "examples": []}
    from datasets import load_dataset

    attempts = [
        ("lambada", None, "validation"),
        ("lambada", None, "test"),
        ("EleutherAI/lambada_openai", None, "test"),
        ("EleutherAI/lambada_openai", None, "validation"),
    ]
    errors: List[str] = []
    for name, config, split in attempts:
        try:
            if config is None:
                ds = load_dataset(name, split=split, streaming=True)
            else:
                ds = load_dataset(name, config, split=split, streaming=True)
            examples: List[torch.Tensor] = []
            for row in ds:
                text = row.get("text") if isinstance(row, dict) else None
                if not isinstance(text, str) or not text.strip():
                    continue
                ids = tokenizer(text.strip(), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                if ids.numel() < 2:
                    continue
                examples.append(ids[-(seq_len + 1) :].clone())
                if len(examples) >= n_examples:
                    return {"source": f"{name}:{split}", "examples": examples}
            errors.append(f"{name}:{split}: only {len(examples)} usable examples")
        except Exception as exc:
            errors.append(f"{name}:{split}: {type(exc).__name__}: {exc}")
    return {"source": "unavailable", "examples": [], "errors": errors}


@torch.no_grad()
def evaluate_last_token_cloze(
    model: torch.nn.Module,
    examples: List[torch.Tensor],
    device: torch.device,
) -> Dict[str, float]:
    if not examples:
        return {"n": 0, "nll": float("nan"), "top1": float("nan"), "top5": float("nan")}
    model.eval()
    total_loss = 0.0
    top1 = 0
    top5 = 0
    for ids in examples:
        x = ids[:-1].unsqueeze(0).to(device)
        y = ids[-1].view(1).to(device)
        logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
        total_loss += float(F.cross_entropy(logits.view(1, -1), y))
        top = torch.topk(logits, k=min(5, logits.numel())).indices
        target = int(y.item())
        top1 += int(int(top[0].item()) == target)
        top5 += int(target in {int(v) for v in top.tolist()})
    n = len(examples)
    return {"n": n, "nll": total_loss / n, "top1": top1 / n, "top5": top5 / n}


def random_signflip_candidates(
    codes: Dict[str, object],
    operator: str,
    pool: int,
    seed: int,
) -> List[FlipCandidate]:
    keys = ("q", "k") if operator == "qk" else ("v", "o")
    rng = np.random.default_rng(seed)
    rows: List[FlipCandidate] = []
    for key in keys:
        code = codes[key]
        active = torch.nonzero((code.state.ne(0) & code.valid).flatten(), as_tuple=False).flatten().cpu().numpy()
        if active.size == 0:
            continue
        n = min(max(1, pool // max(len(keys), 1)), active.size)
        indices = rng.choice(active, size=n, replace=False)
        flat_state = code.state.flatten()
        for idx in indices.tolist():
            old = float(flat_state[int(idx)])
            rows.append(FlipCandidate(key, int(idx), -old, float(rng.random())))
    rng.shuffle(rows)
    return rows[:pool]


def set_layer_qk_weights(model: torch.nn.Module, layer: int, weights: Dict[str, torch.Tensor]) -> None:
    modules = target_modules(model, layer)
    for key in ("q", "k"):
        set_projection_weight(modules[key], weights[key])


def snapshot_qk(model: torch.nn.Module, layers: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    rows: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        modules = target_modules(model, layer)
        rows[layer] = {key: projection_weight(modules[key]).detach().float().cpu().clone() for key in ("q", "k")}
    return rows


def patch_set(
    model: torch.nn.Module,
    layers: List[int],
    direct_qk: Dict[int, Dict[str, torch.Tensor]],
    edited_qk: Dict[int, Dict[str, torch.Tensor]],
    selected: List[int],
) -> None:
    for layer in layers:
        set_layer_qk_weights(model, layer, direct_qk[layer])
    for layer in selected:
        set_layer_qk_weights(model, layer, edited_qk[layer])


def collect_ce_qk_grads(
    model: torch.nn.Module,
    batches: List[torch.Tensor],
    layers: List[int],
    device: torch.device,
    grad_batches: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    old_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    for layer in layers:
        modules = target_modules(model, layer)
        # For GPT-NeoX, Q and K are slices of one fused parameter. Enabling the
        # owning parameter is sufficient; we read only the corresponding slices
        # of its gradient below. For OPT this enables the two separate Linear
        # parameters as before.
        modules["q"].module.weight.requires_grad_(True)
        modules["k"].module.weight.requires_grad_(True)

    model.eval()
    model.zero_grad(set_to_none=True)
    used = min(grad_batches, len(batches))
    for batch in batches[:used]:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        logits = model(input_ids=x, use_cache=False).logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)) / max(used, 1)
        loss.backward()

    grads: Dict[int, Dict[str, torch.Tensor]] = {}
    for layer in layers:
        modules = target_modules(model, layer)
        grads[layer] = {
            "q": modules["q"].module.weight.grad[modules["q"].start : modules["q"].end].detach().float().cpu().clone(),
            "k": modules["k"].module.weight.grad[modules["k"].start : modules["k"].end].detach().float().cpu().clone(),
        }
    model.zero_grad(set_to_none=True)
    for name, p in model.named_parameters():
        p.requires_grad_(old_requires_grad[name])
    return grads


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
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    log(f"loading {args.model} dtype={args.dtype} layers={len(layers)} gpu={torch.cuda.get_device_name(0)}")
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
        args.fit_token_offset,
        args.val_token_offset,
    )
    c4_untouched = build_c4_untouched_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )
    cloze_data = build_lambada_cloze_examples(tokenizer, args.seq_len, args.cloze_examples)
    cloze_examples = cloze_data["examples"]
    timing["load_tokenizer_and_data_sec"] = time.time() - t0

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model.config.use_cache = False
    architecture = _architecture_name(model)
    adapter_layout = "separate_qk_linear" if architecture == "opt" else "gpt_neox_fused_qkv_row_slices"
    log(f"architecture_adapter={architecture}:{adapter_layout}")
    timing["load_model_sec"] = time.time() - t0

    t0 = time.time()
    fp_weights_by_layer = {
        layer: {key: projection_weight(ref).detach().float().cpu() for key, ref in target_modules(model, layer).items()}
        for layer in layers
    }
    fp_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    if c4_untouched is not None:
        fp_nll["untouched_c4"] = evaluate_nll(model, c4_untouched, device)
    fp_cloze = evaluate_last_token_cloze(model, cloze_examples, device)
    timing["snapshot_fp_and_fp_nll_sec"] = time.time() - t0

    t0 = time.time()
    quant_counts = apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
    direct_qk = snapshot_qk(model, layers)
    timing["direct_ptq_apply_sec"] = time.time() - t0

    t0 = time.time()
    direct_nll = {
        "val": evaluate_nll(model, val, device),
        "untouched_w": evaluate_nll(model, untouched, device),
    }
    if c4_untouched is not None:
        direct_nll["untouched_c4"] = evaluate_nll(model, c4_untouched, device)
    direct_cloze = evaluate_last_token_cloze(model, cloze_examples, device)
    timing["direct_ptq_nll_eval_sec"] = time.time() - t0

    t0 = time.time()
    ce_grads = collect_ce_qk_grads(model, fit, layers, device, args.grad_batches)
    timing["ce_gradient_collection_sec"] = time.time() - t0

    support_qk: Dict[int, Dict[str, torch.Tensor]] = {}
    signflip_qk: Dict[int, Dict[str, torch.Tensor]] = {}
    random_support_qk: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {
        rep: {} for rep in range(max(0, args.random_control_repeats))
    }
    random_signflip_qk: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {
        rep: {} for rep in range(max(0, args.random_control_repeats))
    }
    per_layer: List[Dict[str, object]] = []
    edit_t0 = time.time()
    for layer in layers:
        log(f"layer={layer} CE-gradient edits and single-layer NLL")
        fp_weights = fp_weights_by_layer[layer]
        codes = {key: make_code(fp_weights[key], args.group_size, args.threshold_factor) for key in MATRIX_KEYS}
        base_states = {key: code.state.clone() for key, code in codes.items()}
        grads = {
            "q": ce_grads[layer]["q"],
            "k": ce_grads[layer]["k"],
            "v": torch.zeros_like(fp_weights["v"]),
            "o": torch.zeros_like(fp_weights["o"]),
        }

        support_candidates = gradient_projection_candidates_unique(codes, grads, "qk", args.max_edits)
        support_states, support_trace = run_one_shot(base_states, codes, support_candidates, args.max_edits)
        support_qk[layer] = {
            key: weight_from_state(codes[key], support_states[key], refit=True)
            for key in ("q", "k")
        }

        signflip_candidates = gradient_signflip_candidates(codes, grads, "qk", args.max_edits)
        signflip_states, signflip_trace = run_one_shot_flips(base_states, signflip_candidates, args.max_edits)
        signflip_qk[layer] = {
            key: weight_from_state(codes[key], signflip_states[key], refit=True)
            for key in ("q", "k")
        }

        random_rows: Dict[str, float] = {}
        for rep in range(max(0, args.random_control_repeats)):
            rand_seed = int(args.seed + 100000 + rep * 1009 + layer * 37)
            rand_support_candidates = random_support_candidates(codes, "qk", args.max_edits, rand_seed)
            rand_support_states, rand_support_trace = run_one_shot(
                base_states, codes, rand_support_candidates, args.max_edits
            )
            random_support_qk[rep][layer] = {
                key: weight_from_state(codes[key], rand_support_states[key], refit=True)
                for key in ("q", "k")
            }

            rand_flip_candidates = random_signflip_candidates(codes, "qk", args.max_edits, rand_seed + 17)
            rand_flip_states, rand_flip_trace = run_one_shot_flips(
                base_states, rand_flip_candidates, args.max_edits
            )
            random_signflip_qk[rep][layer] = {
                key: weight_from_state(codes[key], rand_flip_states[key], refit=True)
                for key in ("q", "k")
            }

            set_layer_qk_weights(model, layer, random_support_qk[rep][layer])
            rand_support_val = evaluate_nll(model, val, device)
            set_layer_qk_weights(model, layer, direct_qk[layer])

            set_layer_qk_weights(model, layer, random_signflip_qk[rep][layer])
            rand_signflip_val = evaluate_nll(model, val, device)
            set_layer_qk_weights(model, layer, direct_qk[layer])

            random_rows[f"random_support_r{rep}_accepted_edits"] = len(rand_support_trace)
            random_rows[f"random_signflip_r{rep}_accepted_edits"] = len(rand_flip_trace)
            random_rows[f"random_support_r{rep}_single_val_nll"] = rand_support_val
            random_rows[f"random_support_r{rep}_single_val_delta"] = rand_support_val - direct_nll["val"]
            random_rows[f"random_signflip_r{rep}_single_val_nll"] = rand_signflip_val
            random_rows[f"random_signflip_r{rep}_single_val_delta"] = rand_signflip_val - direct_nll["val"]

        set_layer_qk_weights(model, layer, support_qk[layer])
        support_val = evaluate_nll(model, val, device)
        set_layer_qk_weights(model, layer, direct_qk[layer])

        set_layer_qk_weights(model, layer, signflip_qk[layer])
        signflip_val = evaluate_nll(model, val, device)
        set_layer_qk_weights(model, layer, direct_qk[layer])

        per_layer.append(
            {
                "layer": layer,
                "support_accepted_edits": len(support_trace),
                "signflip_accepted_edits": len(signflip_trace),
                "support_single_val_nll": support_val,
                "support_single_val_delta": support_val - direct_nll["val"],
                "signflip_single_val_nll": signflip_val,
                "signflip_single_val_delta": signflip_val - direct_nll["val"],
                "support_selected_by_val": bool(support_val <= direct_nll["val"]),
                "signflip_selected_by_val": bool(signflip_val <= direct_nll["val"]),
                **random_rows,
            }
        )
    timing["edit_generation_and_single_layer_eval_sec"] = time.time() - edit_t0

    support_selected = [int(row["layer"]) for row in per_layer if row["support_selected_by_val"]]
    signflip_selected = [int(row["layer"]) for row in per_layer if row["signflip_selected_by_val"]]
    support_top = [
        int(row["layer"])
        for row in sorted(per_layer, key=lambda r: float(r["support_single_val_delta"]))[: args.support_topk]
    ]
    signflip_top = [
        int(row["layer"])
        for row in sorted(per_layer, key=lambda r: float(r["signflip_single_val_delta"]))[: args.signflip_topk]
    ]

    patch_defs = {
        "cegsp-support-all-qk": ("support", list(layers)),
        "cegsp-support-selected-qk": ("support", support_selected),
        "cegsp-support-topk-qk": ("support", support_top),
        "ce-signflip-all-qk": ("signflip", list(layers)),
        "ce-signflip-selected-qk": ("signflip", signflip_selected),
        "ce-signflip-topk-qk": ("signflip", signflip_top),
    }
    if args.k_sweep.strip():
        k_values = [int(x.strip()) for x in args.k_sweep.split(",") if x.strip()]
        support_ranked = [
            int(row["layer"])
            for row in sorted(per_layer, key=lambda r: float(r["support_single_val_delta"]))
        ]
        signflip_ranked = [
            int(row["layer"])
            for row in sorted(per_layer, key=lambda r: float(r["signflip_single_val_delta"]))
        ]
        joint_rows = []
        for row in per_layer:
            supp_delta = float(row["support_single_val_delta"])
            flip_delta = float(row["signflip_single_val_delta"])
            if supp_delta <= flip_delta:
                joint_rows.append((int(row["layer"]), "support", supp_delta))
            else:
                joint_rows.append((int(row["layer"]), "signflip", flip_delta))
        joint_ranked = sorted(joint_rows, key=lambda x: x[2])
        for k in k_values:
            kk = max(0, min(k, len(layers)))
            ce_joint_selected = [(layer, kind) for layer, kind, _ in joint_ranked[:kk]]
            ce_joint_layers = [layer for layer, _ in ce_joint_selected]
            support_top_layers = support_ranked[:kk]
            signflip_top_layers = signflip_ranked[:kk]
            patch_defs[f"ksweep-support-top{kk}-qk"] = ("support", support_ranked[:kk])
            patch_defs[f"ksweep-signflip-top{kk}-qk"] = ("signflip", signflip_ranked[:kk])
            # Joint patch sets need custom handling because different layers use
            # different edit families.
            patch_defs[f"ksweep-joint-top{kk}-qk"] = ("joint", ce_joint_selected)
            patch_defs[f"matched-support-on-joint-layers-top{kk}-qk"] = ("support", ce_joint_layers)
            patch_defs[f"matched-signflip-on-joint-layers-top{kk}-qk"] = ("signflip", ce_joint_layers)
            patch_defs[f"matched-signflip-on-support-layers-top{kk}-qk"] = ("signflip", support_top_layers)
            patch_defs[f"matched-support-on-signflip-layers-top{kk}-qk"] = ("support", signflip_top_layers)
            for rep in range(max(0, args.random_control_repeats)):
                rand_support_ranked = [
                    int(row["layer"])
                    for row in sorted(
                        per_layer,
                        key=lambda r, rep=rep: float(r[f"random_support_r{rep}_single_val_delta"]),
                    )
                ]
                rand_signflip_ranked = [
                    int(row["layer"])
                    for row in sorted(
                        per_layer,
                        key=lambda r, rep=rep: float(r[f"random_signflip_r{rep}_single_val_delta"]),
                    )
                ]
                rand_joint_rows = []
                for row in per_layer:
                    layer = int(row["layer"])
                    supp_delta = float(row[f"random_support_r{rep}_single_val_delta"])
                    flip_delta = float(row[f"random_signflip_r{rep}_single_val_delta"])
                    if supp_delta <= flip_delta:
                        rand_joint_rows.append((layer, "random_support", supp_delta))
                    else:
                        rand_joint_rows.append((layer, "random_signflip", flip_delta))
                rand_joint_ranked = sorted(rand_joint_rows, key=lambda x: x[2])
                rand_joint_selected = [(layer, kind) for layer, kind, _ in rand_joint_ranked[:kk]]
                patch_defs[f"random-r{rep}-support-top{kk}-qk"] = (
                    "random_support",
                    (rep, rand_support_ranked[:kk]),
                )
                patch_defs[f"random-r{rep}-signflip-top{kk}-qk"] = (
                    "random_signflip",
                    (rep, rand_signflip_ranked[:kk]),
                )
                patch_defs[f"random-r{rep}-joint-top{kk}-qk"] = (
                    "random_joint",
                    (rep, rand_joint_selected),
                )
                patch_defs[f"matched-r{rep}-random-candidate-on-ce-joint-top{kk}-qk"] = (
                    "random_joint",
                    (rep, [(layer, f"random_{kind}") for layer, kind in ce_joint_selected]),
                )
                patch_defs[f"matched-r{rep}-ce-candidate-on-random-joint-top{kk}-qk"] = (
                    "joint",
                    [(layer, kind.replace("random_", "")) for layer, kind in rand_joint_selected],
                )
    patch_results: Dict[str, object] = {}
    cloze_prefixes = tuple(x.strip() for x in args.cloze_patch_prefixes.split(",") if x.strip())
    patch_t0 = time.time()
    for name, (kind, selected) in patch_defs.items():
        if kind == "joint":
            log(f"evaluating patch set {name} layer_edits={selected}")
            for layer in layers:
                set_layer_qk_weights(model, layer, direct_qk[layer])
            for layer, edit_kind in selected:
                edited = support_qk if edit_kind == "support" else signflip_qk
                set_layer_qk_weights(model, int(layer), edited[int(layer)])
            selected_layers = [int(layer) for layer, _ in selected]
            selected_edit_kinds = {str(int(layer)): edit_kind for layer, edit_kind in selected}
        elif kind == "random_joint":
            rep, selected = selected
            log(f"evaluating patch set {name} layer_edits={selected}")
            for layer in layers:
                set_layer_qk_weights(model, layer, direct_qk[layer])
            for layer, edit_kind in selected:
                edited = random_support_qk[rep] if edit_kind == "random_support" else random_signflip_qk[rep]
                set_layer_qk_weights(model, int(layer), edited[int(layer)])
            selected_layers = [int(layer) for layer, _ in selected]
            selected_edit_kinds = {str(int(layer)): edit_kind for layer, edit_kind in selected}
        elif kind in ("random_support", "random_signflip"):
            rep, selected = selected
            edited = random_support_qk[rep] if kind == "random_support" else random_signflip_qk[rep]
            log(f"evaluating patch set {name} layers={selected}")
            patch_set(model, layers, direct_qk, edited, selected)
            selected_layers = selected
            selected_edit_kinds = None
        else:
            edited = support_qk if kind == "support" else signflip_qk
            log(f"evaluating patch set {name} layers={selected}")
            patch_set(model, layers, direct_qk, edited, selected)
            selected_layers = selected
            selected_edit_kinds = None
        scores = {
            "val": evaluate_nll(model, val, device),
            "untouched_w": evaluate_nll(model, untouched, device),
        }
        if c4_untouched is not None:
            scores["untouched_c4"] = evaluate_nll(model, c4_untouched, device)
        patch_results[name] = {
            "layers": selected_layers,
            "edit_kinds": selected_edit_kinds,
            "n_layers": len(selected_layers),
            "nll": scores,
            "cloze": (
                evaluate_last_token_cloze(model, cloze_examples, device)
                if cloze_examples and (not cloze_prefixes or name.startswith(cloze_prefixes))
                else None
            ),
            "delta_vs_direct_val": scores["val"] - direct_nll["val"],
            "delta_vs_direct_untouched_w": scores["untouched_w"] - direct_nll["untouched_w"],
            "delta_vs_direct_untouched_c4": (
                scores["untouched_c4"] - direct_nll["untouched_c4"]
                if c4_untouched is not None
                else None
            ),
        }
    patch_set(model, layers, direct_qk, support_qk, [])
    timing["patch_set_nll_eval_sec"] = time.time() - patch_t0

    result = {
        "run_id": args.run_id,
        "model": args.model,
        "architecture_adapter": {
            "family": architecture,
            "layout": adapter_layout,
            "target_projection": "qk",
        },
        "config": vars(args),
        "validation_version": {
            "name": "CEGSP-07A-ternary-specificity-compatible",
            "primary_question": "Do CE gradients at deployed ternary weights expose a ternary-specific zero-support relocation signal beyond nonzero-only signflip controls?",
            "gate": {
                "primary": "CE joint top-k should improve untouched NLL versus direct ternary and random joint controls",
                "matched_control": "compare support relocation and nonzero-only signflip on the same selected layers",
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
            "uses_path_barrier_or_tdbt_transport": False,
            "uses_ce_gradient_at_quantized_weights": True,
            "uses_optimizer_steps": False,
        },
        "data": {
            "source": data_source,
            "fit_batches": len(fit),
            "val_batches": len(val),
            "untouched_w_batches": len(untouched),
            "untouched_c4_batches": len(c4_untouched) if c4_untouched is not None else 0,
            "cloze_source": cloze_data.get("source"),
            "cloze_examples": len(cloze_examples),
            "cloze_errors": cloze_data.get("errors", []),
            "split": "Wikitext-2 train fit / Wikitext-2 validation for selection and untouched; optional C4 validation is report-only transfer",
        },
        "nll": {
            "fp": fp_nll,
            "direct_ternary": direct_nll,
            "patch_sets": patch_results,
        },
        "cloze": {
            "fp": fp_cloze,
            "direct_ternary": direct_cloze,
        },
        "quant_counts": quant_counts,
        "per_layer": per_layer,
        "timing": timing,
        "status": "complete",
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    log(f"wrote {out_dir / 'result.json'} elapsed={result['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
