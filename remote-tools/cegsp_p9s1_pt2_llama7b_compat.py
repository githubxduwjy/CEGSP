#!/usr/bin/env python3
"""P9-S1: instrumented official PT2 Llama-2-7B + frozen affine CEGSP.

The script re-runs the official PT2 ATQ+SSR pipeline, captures the real
ternary state inside the quantizer, reconstructs the SSR coordinate system,
and only then evaluates frozen CEGSP support relocation.  No state is fitted
from the final checkpoint and no hyperparameter search is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import set_seed

from cegsp_p7_a100_scaling import (
    AffineCode,
    AffineEdit,
    audit_all,
    build_top_candidates,
    changed_coordinates,
    collect_grads,
    get_decoder_layers,
    metric_delta,
    target_qk,
)


REFERENCE_W2_PPL = 11.6425
REFERENCE_C4_PPL = 24.3239
INSTRUMENTATION_PPL_TOL = 0.05


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/Llama-2-7b-hf")
    parser.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    parser.add_argument("--pt2-data-root", default="/root/PT2-data")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", default="/root/tqgsp-runs")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--calib-seq-len", type=int, default=2048)
    parser.add_argument("--ppl-seq-len", type=int, default=2048)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--num-p", type=int, default=1)
    parser.add_argument("--threshold-factor", type=float, default=0.75)
    parser.add_argument("--layer-probe-edits", type=int, default=8)
    parser.add_argument("--edits-per-layer", type=int, default=64)
    parser.add_argument("--primary-layer-budget", type=int, default=6)
    parser.add_argument("--grad-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260837)
    return parser.parse_args()


def module_by_name(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    current = layer
    for part in name.split("."):
        current = getattr(current, part)
    return current


def module_specs(model: torch.nn.Module) -> List[Tuple[int, str, torch.nn.Module]]:
    names = [
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    ]
    return [
        (layer_idx, name, module_by_name(layer, name))
        for layer_idx, layer in enumerate(get_decoder_layers(model))
        for name in names
    ]


def set_module_weight(module: torch.nn.Module, weight: torch.Tensor) -> None:
    module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))


def snapshot_qk(model: torch.nn.Module, layers: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    return {
        # Preserve the deployed dtype for the two snapshots.  Consumers cast
        # to FP32 when doing arithmetic; keeping the snapshots in FP16/BF16
        # avoids an unnecessary multi-GB host-memory allocation on 7B models.
        layer: {key: ref.module.weight.detach().cpu().clone() for key, ref in target_qk(model, layer).items()}
        for layer in layers
    }


def restore_qk(model: torch.nn.Module, snapshot: Dict[int, Dict[str, torch.Tensor]]) -> None:
    for layer, values in snapshot.items():
        refs = target_qk(model, layer)
        for key, weight in values.items():
            set_module_weight(refs[key].module, weight)


def affine_from_q_and_t(q: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
    q = q.detach().float()
    t = t.detach().float()
    t_mean = t.mean(dim=1, keepdim=True)
    q_mean = q.mean(dim=1, keepdim=True)
    tc = t - t_mean
    qc = q - q_mean
    denom = (tc * tc).sum(dim=1, keepdim=True)
    alpha = torch.where(denom > 1e-12, (tc * qc).sum(dim=1, keepdim=True) / denom, torch.zeros_like(denom))
    mu = q_mean - alpha * t_mean
    residual = float((q - (mu + alpha * t)).abs().max().item())
    return mu, alpha, residual


def compose_ssr_perm(columns: int, blocksize: int, local_perms: Sequence[torch.Tensor]) -> torch.Tensor:
    perm = torch.arange(columns, dtype=torch.long)
    expected = (columns + blocksize - 1) // blocksize
    if len(local_perms) != expected:
        raise RuntimeError(f"SSR permutation trace mismatch: got={len(local_perms)} expected={expected}")
    for block_idx, local in enumerate(local_perms):
        col_start = block_idx * blocksize
        remaining = perm[col_start:]
        local = local.to(dtype=torch.long, device=perm.device)
        perm[col_start:] = remaining[local]
    if sorted(perm.tolist()) != list(range(columns)):
        raise RuntimeError("SSR permutation is not a bijection")
    return perm


class Capture:
    def __init__(self) -> None:
        self.initial_t: torch.Tensor | None = None
        self.final_t: torch.Tensor | None = None
        self.records: List[Dict[str, torch.Tensor]] = []
        self.modules: List[Dict[str, object]] = []
        self.active_ssr: object | None = None
        self.local_perms: List[torch.Tensor] = []
        self.total_quantizer_calls = 0

    def reset_quantizer(self) -> None:
        self.initial_t = None
        self.final_t = None


def install_capture(qmod: object, gptqmod: object, ssrmod: object, capture: Capture, group_size: int):
    original_init = qmod.ternary_init
    original_update = qmod.update_ternary
    original_quantize = qmod.TernaryQuantizer.quantize
    original_regular_fasterquant = gptqmod.GPTQ.fasterquant
    original_ssr_fasterquant = ssrmod.GPTQ_SSR.fasterquant
    original_topk = ssrmod.topk_similar_columns

    def init_capture(x, *args, **kwargs):
        result = original_init(x, *args, **kwargs)
        capture.initial_t = result[2].detach().float().cpu().clone()
        return result

    def update_capture(x, alpha, mean):
        result = original_update(x, alpha, mean)
        capture.final_t = result.detach().float().cpu().clone()
        return result

    def quantize_capture(self, w, *args, **kwargs):
        capture.reset_quantizer()
        q, placeholder_t = original_quantize(self, w, *args, **kwargs)
        ternary = capture.final_t if capture.final_t is not None else capture.initial_t
        if ternary is None:
            raise RuntimeError("PT2 quantizer did not expose a ternary state")
        q_cpu = q.detach().float().cpu().clone()
        # T is a discrete index state; retain it compactly while the current
        # PT2 module is being assembled.
        t_cpu = ternary.detach().to(torch.int8).cpu().clone()
        if q_cpu.shape != t_cpu.shape:
            raise RuntimeError(f"quantizer state shape mismatch q={q_cpu.shape} T={t_cpu.shape}")
        capture.records.append({"q": q_cpu, "T": t_cpu})
        capture.total_quantizer_calls += 1
        return q, placeholder_t

    def topk_capture(w_left, *args, **kwargs):
        result = original_topk(w_left, *args, **kwargs)
        if capture.active_ssr is not None:
            capture.local_perms.append(result.detach().cpu().clone())
        return result

    def regular_capture(self, *args, **kwargs):
        # Keep only the current module's block captures.  The previous version
        # retained q/T for all 224 modules and hit the 128 GiB cgroup limit.
        capture.records = []
        result = original_regular_fasterquant(self, *args, **kwargs)
        q_blocks = capture.records
        if not q_blocks:
            raise RuntimeError("regular PT2 module produced no quantizer records")
        module_index = len(capture.modules)
        is_target = module_index % 7 in (0, 2)
        q_cat = torch.cat([row["q"] for row in q_blocks], dim=1)
        t_cat = torch.cat([row["T"] for row in q_blocks], dim=1)
        columns = int(q_cat.shape[1])
        capture.modules.append({
            "q": q_cat if is_target else None,
            "T": t_cat if is_target else None,
            "perm": torch.arange(columns, dtype=torch.long),
            "ssr": False,
            "shape": [int(q_cat.shape[0]), columns],
        })
        capture.records = []
        return result

    def ssr_capture(self, *args, **kwargs):
        capture.records = []
        capture.active_ssr = self
        capture.local_perms = []
        try:
            result = original_ssr_fasterquant(self, *args, **kwargs)
        finally:
            capture.active_ssr = None
        q_blocks = capture.records
        if not q_blocks:
            raise RuntimeError("SSR PT2 module produced no quantizer records")
        q_cat = torch.cat([row["q"] for row in q_blocks], dim=1)
        t_cat = torch.cat([row["T"] for row in q_blocks], dim=1)
        blocksize = int(kwargs.get("blocksize", group_size))
        perm = compose_ssr_perm(int(q_cat.shape[1]), blocksize, capture.local_perms)
        module_index = len(capture.modules)
        is_target = module_index % 7 in (0, 2)
        capture.modules.append({
            "q": q_cat if is_target else None,
            "T": t_cat if is_target else None,
            "perm": perm,
            "ssr": True,
            "shape": [int(q_cat.shape[0]), int(q_cat.shape[1])],
        })
        capture.records = []
        return result

    qmod.ternary_init = init_capture
    qmod.update_ternary = update_capture
    qmod.TernaryQuantizer.quantize = quantize_capture
    gptqmod.GPTQ.fasterquant = regular_capture
    ssrmod.GPTQ_SSR.fasterquant = ssr_capture
    ssrmod.topk_similar_columns = topk_capture

    def restore() -> None:
        qmod.ternary_init = original_init
        qmod.update_ternary = original_update
        qmod.TernaryQuantizer.quantize = original_quantize
        gptqmod.GPTQ.fasterquant = original_regular_fasterquant
        ssrmod.GPTQ_SSR.fasterquant = original_ssr_fasterquant
        ssrmod.topk_similar_columns = original_topk

    return restore


def pad_permuted(weight: torch.Tensor, group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, columns = weight.shape
    blocks = (columns + group_size - 1) // group_size
    padded = torch.zeros((rows, blocks * group_size), dtype=torch.float32)
    valid = torch.zeros_like(padded, dtype=torch.bool)
    padded[:, :columns] = weight.float()
    valid[:, :columns] = True
    return padded.view(rows, blocks, group_size), valid.view(rows, blocks, group_size)


def build_codes(
    model: torch.nn.Module,
    fp_qk: Dict[int, Dict[str, torch.Tensor]],
    captured_modules: Sequence[Dict[str, object]],
    group_size: int,
) -> Tuple[Dict[int, Dict[str, AffineCode]], Dict[int, Dict[str, torch.Tensor]], Dict[str, object]]:
    specs = module_specs(model)
    if len(specs) != len(captured_modules):
        raise RuntimeError(f"module capture mismatch specs={len(specs)} captures={len(captured_modules)}")
    codes: Dict[int, Dict[str, AffineCode]] = {}
    perms: Dict[int, Dict[str, torch.Tensor]] = {}
    parity_rows: List[Dict[str, object]] = []
    for (_, (layer, name, module)) in enumerate(specs):
        capture = captured_modules[_]
        if name not in {"self_attn.q_proj", "self_attn.k_proj"}:
            continue
        key = "q" if name.endswith("q_proj") else "k"
        q_perm = capture["q"].float()
        t_perm = capture["T"].float()
        perm = capture["perm"].long()
        if q_perm.shape != t_perm.shape or q_perm.shape[1] != module.weight.shape[1]:
            raise RuntimeError(f"capture shape mismatch layer={layer} name={name} capture={q_perm.shape} module={module.weight.shape}")
        fp_original = fp_qk[layer][key]
        fp_perm = fp_original[:, perm]
        fp_padded, valid = pad_permuted(fp_perm, group_size)
        rows, columns = q_perm.shape
        blocks = (columns + group_size - 1) // group_size
        t_padded = torch.zeros((rows, blocks * group_size), dtype=torch.float32)
        q_padded = torch.zeros_like(t_padded)
        t_padded[:, :columns] = t_perm
        q_padded[:, :columns] = q_perm
        t_group = t_padded.view(rows, blocks, group_size).round().to(torch.int8)
        q_group = q_padded.view(rows, blocks, group_size)
        mu_rows: List[torch.Tensor] = []
        alpha_rows: List[torch.Tensor] = []
        residuals: List[float] = []
        for block in range(blocks):
            width = min(group_size, columns - block * group_size)
            mu, alpha, residual = affine_from_q_and_t(q_group[:, block, :width], t_group[:, block, :width])
            mu_rows.append(mu)
            alpha_rows.append(alpha)
            residuals.append(residual)
        mu = torch.stack([x.squeeze(1) for x in mu_rows], dim=1).unsqueeze(-1)
        alpha = torch.stack([x.squeeze(1) for x in alpha_rows], dim=1).unsqueeze(-1)
        code = AffineCode(
            mu=mu,
            alpha=alpha,
            T=t_group,
            valid=valid,
            original_shape=(int(rows), int(columns)),
            group_size=group_size,
            fp_padded=fp_padded,
        )
        codes.setdefault(layer, {})[key] = code
        perms.setdefault(layer, {})[key] = perm
        inverse = torch.argsort(perm)
        q_original = q_perm[:, inverse]
        deployed = module.weight.detach().float().cpu()
        captured_q = q_original.to(deployed.dtype).float()
        parity_rows.append({
            "layer": layer,
            "module": name,
            "shape": [int(rows), int(columns)],
            "groups": blocks,
            "ssr": bool(capture["ssr"]),
            "permutation_bijection": sorted(perm.tolist()) == list(range(columns)),
            "capture_codebook_residual": max(residuals),
            "illegal_T": int(((t_group != -1) & (t_group != 0) & (t_group != 1)).sum().item()),
            "nonfinite_T": int((~torch.isfinite(t_group.float())).sum().item()),
            "final_vs_capture_q_max_abs": float((deployed - captured_q).abs().max().item()),
            "capture_q_abs_max": float(q_original.abs().max().item()),
            "permutation_sha256": hashlib.sha256(perm.numpy().tobytes()).hexdigest(),
        })
    return codes, perms, {"rows": parity_rows}


def parity_gate(
    codes: Dict[int, Dict[str, AffineCode]],
    parity_detail: Dict[str, object],
    perms: Dict[int, Dict[str, torch.Tensor]],
    group_size: int,
) -> Dict[str, object]:
    rows = parity_detail["rows"]
    code_audit = audit_all(codes)
    max_capture_residual = max((float(row["capture_codebook_residual"]) for row in rows), default=float("inf"))
    max_deployed_residual = max((float(row["final_vs_capture_q_max_abs"]) for row in rows), default=float("inf"))
    illegal = sum(int(row["illegal_T"]) for row in rows)
    nonfinite = sum(int(row["nonfinite_T"]) for row in rows)
    all_bijection = all(bool(row["permutation_bijection"]) for row in rows)
    passed = (
        len(rows) == 64
        and group_size == 128
        and illegal == 0
        and nonfinite == 0
        and all_bijection
        and max_capture_residual < 1e-3
        and max_deployed_residual < 1e-3
        and code_audit["total_illegal_states"] == 0
        and all(len(perms[layer]) == 2 for layer in perms)
    )
    return {
        "pass": passed,
        "qk_module_count": len(rows),
        "expected_qk_module_count": 64,
        "group_size": group_size,
        "scale_granularity": "per-row per-SSR-group",
        "ssr_permutation_recorded": all(bool(row["ssr"]) for row in rows),
        "permutation_bijection": all_bijection,
        "illegal_T_count": illegal,
        "nonfinite_T_count": nonfinite,
        "max_capture_codebook_residual": max_capture_residual,
        "max_final_vs_capture_q_residual": max_deployed_residual,
        "code_audit": code_audit,
        "module_rows": rows,
    }


def apply_ssr_codes(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    states: Dict[int, Dict[str, torch.Tensor]] | None = None,
) -> None:
    for layer, layer_codes in codes.items():
        refs = target_qk(model, layer)
        for key, code in layer_codes.items():
            state = code.T if states is None else states[layer][key]
            q_perm = (code.mu + code.alpha * state.float()).view(code.original_shape[0], -1)[:, : code.original_shape[1]]
            inverse = torch.argsort(perms[layer][key])
            q_original = q_perm[:, inverse]
            set_module_weight(refs[key].module, q_original)


def cardinality_violations(codes: Dict[int, Dict[str, AffineCode]], states: Dict[int, Dict[str, torch.Tensor]]) -> int:
    count = 0
    for layer, layer_codes in codes.items():
        for key, code in layer_codes.items():
            before = code.T.abs().sum(dim=-1)
            after = states[layer][key].abs().sum(dim=-1)
            count += int((before != after).sum().item())
    return count


def finite_metrics(metrics: Dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def ppls_to_nll(metrics: Dict[str, float]) -> Dict[str, float]:
    return {key: float(math.log(float(value))) for key, value in metrics.items()}


def official_metrics(model: torch.nn.Module, model_path: str, device: torch.device, pt2_data_root: str, seqlen: int) -> Dict[str, float]:
    data = importlib.import_module("pt2_llm.data")
    evaluator = importlib.import_module("pt2_llm.eval_ppl")
    _, w2_test = data.get_loaders("wikitext2", seed=0, seqlen=seqlen, model=model_path)
    _, c4_test = data.get_loaders("c4", seed=0, seqlen=seqlen, model=model_path)
    w2 = float(evaluator.llama_eval(model, w2_test, device, "wikitext2", False, seqlen))
    c4 = float(evaluator.llama_eval(model, c4_test, device, "c4", False, seqlen))
    result = {"wikitext2_ppl": w2, "c4_ppl": c4}
    if not finite_metrics(result):
        raise RuntimeError(f"official evaluator returned nonfinite metrics: {result}")
    return result


def random_layerwise_edits(codes: Dict[int, Dict[str, AffineCode]], selected_layers: Sequence[int], count: int, seed: int) -> List[AffineEdit]:
    import random

    rng = random.Random(seed)
    selected: List[AffineEdit] = []
    for layer in selected_layers:
        working = {key: code.T.clone() for key, code in codes[layer].items()}
        used = set()
        attempts = 0
        while sum(1 for edit in selected if edit.layer == layer) < count and attempts < 200000:
            attempts += 1
            key = rng.choice(["q", "k"])
            code = codes[layer][key]
            state = working[key]
            rows, blocks, group = state.shape
            row = rng.randrange(rows)
            block = rng.randrange(blocks)
            valid = code.valid[row, block]
            donors = [int(x) for x in torch.where((state[row, block] != 0) & valid)[0].tolist() if (key, row, block, int(x)) not in used]
            receivers = [int(x) for x in torch.where((state[row, block] == 0) & valid)[0].tolist() if (key, row, block, int(x)) not in used]
            if not donors or not receivers:
                continue
            donor = rng.choice(donors)
            receiver = rng.choice(receivers)
            donor_sign = int(state[row, block, donor].item())
            centered = float(code.fp_padded[row, block, receiver] - code.mu[row, block, 0])
            receiver_sign = 1 if centered >= 0 else -1
            selected.append(AffineEdit(layer, key, row, block, donor, receiver, donor_sign, receiver_sign, 0.0))
            used.add((key, row, block, donor))
            used.add((key, row, block, receiver))
            state[row, block, donor] = 0
            state[row, block, receiver] = receiver_sign
    return selected


def layer_counts(edits: Sequence[AffineEdit]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for edit in edits:
        counts[str(edit.layer)] = counts.get(str(edit.layer), 0) + 1
    return counts


def make_pt2_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.nsamples,
        percdamp=args.percdamp,
        blocksize=args.group_size,
        num_p=args.num_p,
        salient_metric="hessian",
        device="cuda:0",
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=args.calib_seq_len,
        ppl_seqlen=args.ppl_seq_len,
        quant_only="",
        invert=False,
        ssr=True,
        log_wandb=False,
        tasks="",
        experiment=args.run_id,
        num_fewshot=0,
        limit=-1,
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.group_size != 128 or args.nsamples != 128 or args.calib_seq_len != 2048 or args.ppl_seq_len != 2048:
        raise ValueError("P9-S1 is frozen to group=128, nsamples=128, calibration/eval seqlen=2048")
    if not torch.cuda.is_available():
        raise RuntimeError("P9-S1 requires CUDA")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sys.path.insert(0, args.pt2_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pt2_quantize = importlib.import_module("quantize")
    qmod = importlib.import_module("pt2_llm.quantizer")
    gptqmod = importlib.import_module("pt2_llm.gptq")
    ssrmod = importlib.import_module("pt2_llm.gptq_ssr")
    data = importlib.import_module("pt2_llm.data")
    pt2_quantize.args = make_pt2_args(args)
    pt2_quantize.groupsize = args.group_size

    calib_loader, _ = data.get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=args.calib_seq_len, model=args.model)
    if len(calib_loader) != args.nsamples:
        raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.nsamples}")

    log(f"loading official PT2 model={args.model} method=atq ssr=True")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    model.seqlen = args.calib_seq_len
    model.eval()
    layers = list(range(len(get_decoder_layers(model))))
    fp_qk = snapshot_qk(model, layers)
    capture = Capture()
    restore_capture = install_capture(qmod, gptqmod, ssrmod, capture, args.group_size)
    quant_started = time.time()
    try:
        pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
    finally:
        restore_capture()
    quant_sec = time.time() - quant_started
    model.to(device)
    model.config.use_cache = False
    pt2_qk = snapshot_qk(model, layers)

    codes, perms, parity_detail = build_codes(model, fp_qk, capture.modules, args.group_size)
    parity = parity_gate(codes, parity_detail, perms, args.group_size)
    result: Dict[str, object] = {
        "run_id": args.run_id,
        "experiment": "P9-S1 official PT2 Llama-2-7B + affine-index CEGSP",
        "status": "parity_passed" if parity["pass"] else "parity_failed",
        "config": vars(args),
        "protocol": {
            "pt2_method": "atq",
            "ssr": True,
            "scope": "all 32 decoder layers, Q/K only",
            "layer_rule": "top-6 by sum of top-8 legal CE candidates",
            "edits_per_layer": args.edits_per_layer,
            "expected_relocations": args.primary_layer_budget * args.edits_per_layer,
            "expected_changed_coordinates": 2 * args.primary_layer_budget * args.edits_per_layer,
            "mu_alpha_refit": False,
            "teacher_or_qat": False,
            "selection_uses_untouched": False,
            "one_quantized_point_backward": True,
            "instrumentation_ppl_abs_tolerance": INSTRUMENTATION_PPL_TOL,
            "random_seed": args.random_seed,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "bf16": torch.cuda.is_bf16_supported(),
            "max_memory_gb_after_quant": torch.cuda.max_memory_allocated() / (1024**3),
        },
        "capture": {
            "quantizer_calls": capture.total_quantizer_calls,
            "module_calls": len(capture.modules),
            "expected_module_calls": len(module_specs(model)),
        },
        "state_parity": parity,
        "quantization_sec": quant_sec,
        "elapsed_until_parity_sec": time.time() - started,
    }

    if not parity["pass"]:
        result["performance_gate"] = "NOT_RUN_STATE_PARITY_FAILED"
        result["elapsed_sec"] = time.time() - started
        path = out_dir / "p9s1_result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"state parity failed; wrote {path}")
        return

    log("collecting exactly one quantized-point CE gradient batch")
    grads_original = collect_grads(model, [calib_loader[0][0]], layers, device, args.grad_batches)
    grads_ssr: Dict[int, Dict[str, torch.Tensor]] = {
        layer: {key: grads_original[layer][key][:, perms[layer][key]] for key in ("q", "k")}
        for layer in layers
    }
    candidates_by_layer: Dict[int, List[AffineEdit]] = {}
    layer_ranking: List[Dict[str, object]] = []
    for layer in layers:
        candidates = build_top_candidates(codes, grads_ssr, layer, max(args.edits_per_layer * 2, args.layer_probe_edits * 2, 128))
        candidates_by_layer[layer] = candidates
        probe = candidates[: args.layer_probe_edits]
        layer_ranking.append({
            "layer": layer,
            "num_candidates": len(candidates),
            "probe_count": len(probe),
            "probe_score_sum": float(sum(float(edit.score) for edit in probe)),
            "top_scores": [float(edit.score) for edit in probe[:5]],
        })
    layer_ranking.sort(key=lambda row: (-float(row["probe_score_sum"]), int(row["layer"])))
    selected_layers = [int(row["layer"]) for row in layer_ranking[: args.primary_layer_budget]]
    ce_edits: List[AffineEdit] = []
    for layer in selected_layers:
        used = set()
        for edit in candidates_by_layer[layer]:
            coords = {(edit.key, edit.row, edit.block, edit.donor), (edit.key, edit.row, edit.block, edit.receiver)}
            if used.intersection(coords):
                continue
            ce_edits.append(edit)
            used.update(coords)
            if sum(1 for x in ce_edits if x.layer == layer) >= args.edits_per_layer:
                break
    ce_states = {layer: {key: code.T.clone() for key, code in layer_codes.items()} for layer, layer_codes in codes.items()}
    for edit in ce_edits:
        ce_states[edit.layer][edit.key][edit.row, edit.block, edit.donor] = 0
        ce_states[edit.layer][edit.key][edit.row, edit.block, edit.receiver] = edit.receiver_sign
    random_edits = random_layerwise_edits(codes, selected_layers, args.edits_per_layer, args.random_seed)
    random_states = {layer: {key: code.T.clone() for key, code in layer_codes.items()} for layer, layer_codes in codes.items()}
    for edit in random_edits:
        random_states[edit.layer][edit.key][edit.row, edit.block, edit.donor] = 0
        random_states[edit.layer][edit.key][edit.row, edit.block, edit.receiver] = edit.receiver_sign

    ce_audit = audit_all(codes, ce_states)
    random_audit = audit_all(codes, random_states)
    legality_pass = (
        ce_audit["total_illegal_states"] == 0
        and random_audit["total_illegal_states"] == 0
        and cardinality_violations(codes, ce_states) == 0
        and cardinality_violations(codes, random_states) == 0
    )
    if not legality_pass:
        raise RuntimeError("patch legality failed before official evaluation")

    log("evaluating instrumented PT2 baseline with official W2/C4 evaluator")
    baseline_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.ppl_seq_len)
    instrumentation_pass = (
        abs(baseline_metrics["wikitext2_ppl"] - REFERENCE_W2_PPL) <= INSTRUMENTATION_PPL_TOL
        and abs(baseline_metrics["c4_ppl"] - REFERENCE_C4_PPL) <= INSTRUMENTATION_PPL_TOL
    )
    result["instrumented_baseline_metrics"] = baseline_metrics
    result["instrumentation_gate"] = {
        "reference": {"wikitext2_ppl": REFERENCE_W2_PPL, "c4_ppl": REFERENCE_C4_PPL},
        "absolute_tolerance": INSTRUMENTATION_PPL_TOL,
        "pass": instrumentation_pass,
    }
    if not instrumentation_pass:
        result["performance_gate"] = "STOP_INSTRUMENTATION_PARITY_FAIL"
        result["elapsed_sec"] = time.time() - started
        path = out_dir / "p9s1_result.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log(f"instrumentation parity failed; wrote {path}")
        return

    apply_ssr_codes(model, codes, perms, None)
    baseline_state_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.ppl_seq_len)
    apply_ssr_codes(model, codes, perms, ce_states)
    ce_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.ppl_seq_len)
    apply_ssr_codes(model, codes, perms, random_states)
    random_metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.ppl_seq_len)
    restore_qk(model, pt2_qk)

    baseline_nll = ppls_to_nll(baseline_state_metrics)
    ce_nll = ppls_to_nll(ce_metrics)
    random_nll = ppls_to_nll(random_metrics)
    finite_pass = finite_metrics(baseline_state_metrics) and finite_metrics(ce_metrics) and finite_metrics(random_metrics)
    ce_vs_pt2_w2 = ce_nll["wikitext2_ppl"] < baseline_nll["wikitext2_ppl"]
    ce_vs_pt2_c4 = ce_nll["c4_ppl"] < baseline_nll["c4_ppl"]
    ce_vs_random_w2 = ce_nll["wikitext2_ppl"] < random_nll["wikitext2_ppl"]
    ce_vs_random_c4 = ce_nll["c4_ppl"] < random_nll["c4_ppl"]
    strong_pass = finite_pass and legality_pass and ce_vs_pt2_w2 and ce_vs_pt2_c4 and ce_vs_random_w2 and ce_vs_random_c4
    result["selected_layers"] = selected_layers
    result["layer_ranking"] = layer_ranking
    result["variants"] = {
        "pt2": {"metrics": baseline_state_metrics, "nll": baseline_nll, "num_edits": 0, "changed_coordinates": 0},
        "pt2_plus_affine_cegsp_top6": {
            "metrics": ce_metrics,
            "nll": ce_nll,
            "delta_vs_pt2_nll": metric_delta(ce_nll, baseline_nll),
            "num_edits": len(ce_edits),
            "changed_coordinates": changed_coordinates(codes, ce_states),
            "edits_per_layer": layer_counts(ce_edits),
            "audit": ce_audit,
            "cardinality_violations": cardinality_violations(codes, ce_states),
        },
        "pt2_plus_matched_random_top6": {
            "metrics": random_metrics,
            "nll": random_nll,
            "delta_vs_pt2_nll": metric_delta(random_nll, baseline_nll),
            "num_edits": len(random_edits),
            "changed_coordinates": changed_coordinates(codes, random_states),
            "edits_per_layer": layer_counts(random_edits),
            "audit": random_audit,
            "cardinality_violations": cardinality_violations(codes, random_states),
        },
    }
    result["performance_gate"] = {
        "legality_pass": legality_pass,
        "finite_pass": finite_pass,
        "ce_improves_pt2_w2": ce_vs_pt2_w2,
        "ce_improves_pt2_c4": ce_vs_pt2_c4,
        "ce_beats_random_w2": ce_vs_random_w2,
        "ce_beats_random_c4": ce_vs_random_c4,
        "strong_compatibility_pass": strong_pass,
        "classification": "STRONG_PASS" if strong_pass else "FAIL_OR_WEAK_REQUIRES_PRESPECIFIED_INTERPRETATION",
    }
    result["elapsed_sec"] = time.time() - started
    path = out_dir / "p9s1_result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    log(f"wrote {path}")
    log(json.dumps(result["performance_gate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
