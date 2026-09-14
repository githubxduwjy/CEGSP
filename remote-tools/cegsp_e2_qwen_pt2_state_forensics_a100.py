#!/usr/bin/env python3
"""E2-A0-F: forensic localization for Qwen3 PT2 sidecar parity residual.

This run intentionally does not run TernRefine/HBA.  It repeats the minimal
official PT2 Qwen health run and records where the ~1e-3 state residual appears:
capture-time quantizer output, final deployed state, affine T/mu/alpha
reconstruction, and save/reload reconstruction.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from transformers import set_seed

from cegsp_e2_qwen_pt2_health_a100 import apply_qwen_pt2_layer_adapter
from cegsp_p7_a100_scaling import get_decoder_layers, target_qk
from cegsp_p9s2_detached_pt2_plugin import (
    Capture,
    affine_from_q_and_t,
    build_codes,
    install_capture,
    module_specs,
    pad_permuted,
    snapshot_qk,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model/bitahub-model/pice35408784b54431987c4d13c457b9cd/Qwen3-8B")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--nsamples", type=int, default=2)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--ppl-seq-len", type=int, default=2048)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--num-p", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--topk", type=int, default=12)
    return p.parse_args()


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


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def tensor_stats(diff: torch.Tensor) -> Dict[str, object]:
    diff = diff.detach().float()
    absdiff = diff.abs()
    flat_idx = int(absdiff.reshape(-1).argmax().item())
    unraveled = list(torch.unravel_index(torch.tensor(flat_idx), absdiff.shape))
    return {
        "max_abs": float(absdiff.max().item()),
        "mean_abs": float(absdiff.mean().item()),
        "rms": float(torch.sqrt((diff * diff).mean()).item()),
        "argmax": [int(x.item()) for x in unraveled],
    }


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Approximate BF16 ULP spacing at x in FP32."""
    x = x.detach().float().abs()
    _, exponent = torch.frexp(x.clamp_min(torch.finfo(torch.float32).tiny))
    return torch.ldexp(torch.ones_like(x), exponent - 8)


def ratio_stats(absdiff: torch.Tensor, denom: torch.Tensor) -> Dict[str, float]:
    ratio = (absdiff.detach().float() / denom.detach().float().clamp_min(1e-30)).reshape(-1)
    return {
        "max": float(ratio.max().item()),
        "mean": float(ratio.mean().item()),
        "p99": float(torch.quantile(ratio, 0.99).item()),
        "frac_le_1_ulp": float((ratio <= 1.0 + 1e-12).float().mean().item()),
        "frac_le_2_ulp": float((ratio <= 2.0 + 1e-12).float().mean().item()),
    }


def discrete_code_parity(
    q_deployed_original: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    t_group: torch.Tensor,
    valid: torch.Tensor,
    perm: torch.Tensor,
    group_size: int,
) -> Dict[str, object]:
    rows, columns = q_deployed_original.shape
    blocks = (columns + group_size - 1) // group_size
    q_perm = q_deployed_original.float()[:, perm]
    q_pad = torch.zeros((rows, blocks * group_size), dtype=torch.float32)
    q_pad[:, :columns] = q_perm
    q_group = q_pad.view(rows, blocks, group_size)
    states = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32).view(1, 1, 1, 3)
    centers = mu.float().unsqueeze(-1) + alpha.float().unsqueeze(-1) * states
    dist = (q_group.unsqueeze(-1) - centers).abs()
    nearest_idx = dist.argmin(dim=-1).to(torch.int8)
    t_deploy = nearest_idx - 1
    valid = valid.bool()
    mismatch = ((t_deploy != t_group.to(torch.int8)) & valid)
    valid_count = int(valid.sum().item())
    mismatch_count = int(mismatch.sum().item())
    margin = torch.sort(dist, dim=-1).values
    valid_margin = (margin[..., 1] - margin[..., 0])[valid]
    return {
        "valid_count": valid_count,
        "mismatch_count": mismatch_count,
        "mismatch_fraction": float(mismatch_count / max(valid_count, 1)),
        "exact_T_match": mismatch_count == 0,
        "min_nearest_margin": float(valid_margin.min().item()) if valid_margin.numel() else None,
        "p01_nearest_margin": float(torch.quantile(valid_margin.float(), 0.01).item()) if valid_margin.numel() else None,
    }


def module_key(name: str) -> str:
    if name.endswith("q_proj"):
        return "q"
    if name.endswith("k_proj"):
        return "k"
    raise ValueError(name)


def forensic_rows(
    model: torch.nn.Module,
    captured_modules: List[Dict[str, object]],
    group_size: int,
) -> List[Dict[str, object]]:
    specs = module_specs(model)
    rows: List[Dict[str, object]] = []
    for module_idx, (layer, name, module) in enumerate(specs):
        if name not in {"self_attn.q_proj", "self_attn.k_proj"}:
            continue
        capture = captured_modules[module_idx]
        key = module_key(name)
        q1_perm = capture["q"].float()
        t_perm = capture["T"].float()
        perm = capture["perm"].long()
        inverse = torch.argsort(perm)
        rows_n, cols = q1_perm.shape
        blocks = (cols + group_size - 1) // group_size
        t_pad = torch.zeros((rows_n, blocks * group_size), dtype=torch.float32)
        q_pad = torch.zeros_like(t_pad)
        t_pad[:, :cols] = t_perm
        q_pad[:, :cols] = q1_perm
        t_group = t_pad.view(rows_n, blocks, group_size).round().to(torch.int8)
        q_group = q_pad.view(rows_n, blocks, group_size)
        valid = torch.ones((rows_n, blocks, group_size), dtype=torch.bool)
        if blocks * group_size != cols:
            valid[:, -1, cols - (blocks - 1) * group_size :] = False
        mu_rows, alpha_rows, residuals = [], [], []
        for block in range(blocks):
            width = min(group_size, cols - block * group_size)
            mu, alpha, residual = affine_from_q_and_t(q_group[:, block, :width], t_group[:, block, :width])
            mu_rows.append(mu)
            alpha_rows.append(alpha)
            residuals.append(residual)
        mu = torch.stack([x.squeeze(1) for x in mu_rows], dim=1).unsqueeze(-1)
        alpha = torch.stack([x.squeeze(1) for x in alpha_rows], dim=1).unsqueeze(-1)
        qhat_perm = (mu + alpha * t_group.float()).view(rows_n, -1)[:, :cols]
        q1_original = q1_perm[:, inverse]
        qhat_original = qhat_perm[:, inverse]
        q2_deployed = module.weight.detach().cpu()
        q2_float = q2_deployed.float()
        q1_cast = q1_original.to(q2_deployed.dtype).float()
        qhat_cast = qhat_original.to(q2_deployed.dtype).float()
        qhat_bf16 = qhat_original.to(torch.bfloat16).float()
        q1_bf16 = q1_original.to(torch.bfloat16).float()
        ulp_reference = bf16_ulp(q2_float)

        tmp = {
            "layer": int(layer),
            "key": key,
            "module": name,
            "module_index": int(module_idx),
            "shape": [int(rows_n), int(cols)],
            "groups": int(blocks),
            "ssr": bool(capture["ssr"]),
            "deployed_dtype": str(q2_deployed.dtype),
            "capture_q_dtype_after_hook": str(q1_perm.dtype),
            "permutation_bijection": sorted(perm.tolist()) == list(range(cols)),
            "illegal_T": int(((t_group != -1) & (t_group != 0) & (t_group != 1)).sum().item()),
            "nonfinite_T": int((~torch.isfinite(t_group.float())).sum().item()),
            "q1_capture_perm_vs_qhat_perm": tensor_stats(q1_perm - qhat_perm),
            "q1_capture_original_vs_q2_deployed_float": tensor_stats(q1_original - q2_float),
            "q1_capture_original_cast_to_deployed_vs_q2_deployed": tensor_stats(q1_cast - q2_float),
            "qhat_original_vs_q2_deployed_float": tensor_stats(qhat_original - q2_float),
            "qhat_original_cast_to_deployed_vs_q2_deployed": tensor_stats(qhat_cast - q2_float),
            "qhat_bf16_vs_q2_deployed": tensor_stats(qhat_bf16 - q2_float),
            "q1_bf16_vs_q2_deployed": tensor_stats(q1_bf16 - q2_float),
            "qhat_bf16_residual_in_bf16_ulp": ratio_stats((qhat_bf16 - q2_float).abs(), ulp_reference),
            "q1_bf16_residual_in_bf16_ulp": ratio_stats((q1_bf16 - q2_float).abs(), ulp_reference),
            "deployed_inferred_T_vs_sidecar_T": discrete_code_parity(
                q2_float,
                mu,
                alpha,
                t_group,
                valid,
                perm,
                group_size,
            ),
        }
        # Serialize only this module's compact state and reload it to isolate
        # save/load dtype effects without writing multi-GB diagnostic tensors.
        tmp_payload = {
            "T": t_group.cpu(),
            "mu": mu.cpu(),
            "alpha": alpha.cpu(),
            "perm": perm.cpu(),
            "shape": (int(rows_n), int(cols)),
            "group_size": int(group_size),
        }
        rows.append(tmp | {"_tmp_payload": tmp_payload})
    return rows


def reload_compare_for_rows(rows: List[Dict[str, object]], out_dir: Path) -> List[Dict[str, object]]:
    final_rows = []
    tmp_dir = out_dir / "forensic_module_payloads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        payload = row.pop("_tmp_payload")
        path = tmp_dir / f"layer{row['layer']:02d}_{row['key']}.pt"
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu")
        rows_n, cols = loaded["shape"]
        q_reload_perm = (loaded["mu"].float() + loaded["alpha"].float() * loaded["T"].float()).view(rows_n, -1)[:, :cols]
        # Since deployed tensors are not serialized here, this step only checks
        # whether T/mu/alpha/perm survive save/load exactly enough.
        q_payload_perm = (payload["mu"].float() + payload["alpha"].float() * payload["T"].float()).view(rows_n, -1)[:, :cols]
        row["qhat_payload_vs_qhat_reload"] = tensor_stats(q_payload_perm - q_reload_perm)
        row["forensic_payload_path"] = str(path)
        final_rows.append(row)
    return final_rows


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "e2_qwen_pt2_forensics_result.json"
    partial_path = out_dir / "e2_qwen_pt2_forensics_partial.json"
    result: Dict[str, object] = {
        "run_id": args.run_id,
        "experiment": "E2-A0-F Qwen3-8B PT2 sidecar residual forensics",
        "status": "started",
        "config": vars(args),
    }
    write_json(partial_path, result)
    try:
        if args.group_size != 128 or args.calib_seq_len != 2048:
            raise ValueError("forensics run is frozen to group=128 and calibration seq_len=2048")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        sys.path.insert(0, args.pt2_root)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        pt2_quantize = importlib.import_module("quantize")
        qmod = importlib.import_module("pt2_llm.quantizer")
        gptqmod = importlib.import_module("pt2_llm.gptq")
        ssrmod = importlib.import_module("pt2_llm.gptq_ssr")
        data = importlib.import_module("pt2_llm.data")
        if hasattr(data, "DATA_ROOT"):
            data.DATA_ROOT = args.pt2_data_root
        pt2_quantize.args = make_pt2_args(args)
        pt2_quantize.groupsize = args.group_size

        log(f"loading calibration nsamples={args.nsamples}")
        calib_loader, _ = data.get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=args.calib_seq_len, model=args.model)
        log("loading official PT2 model")
        model = pt2_quantize.get_model(args.model, args.calib_seq_len)
        adapter = apply_qwen_pt2_layer_adapter(model, args.model)
        model.seqlen = args.calib_seq_len
        model.eval()
        expected_layers = len(get_decoder_layers(model))
        specs = module_specs(model)
        result.update({
            "status": "model_loaded",
            "model_type": getattr(model.config, "model_type", None),
            "decoder_layers": expected_layers,
            "expected_qk_modules": expected_layers * 2,
            "expected_quantized_modules": len(specs),
            "architecture_adapter": adapter,
        })
        write_json(partial_path, result)

        capture = Capture()
        restore_capture = install_capture(qmod, gptqmod, ssrmod, capture, args.group_size)
        quant_started = time.time()
        log("starting official PT2 quant_sequential")
        try:
            pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
        finally:
            restore_capture()
        quant_sec = time.time() - quant_started
        log("building baseline parity codes")
        layers = list(range(expected_layers))
        fp_qk = snapshot_qk(model, layers)
        _codes, _perms, parity_detail = build_codes(model, fp_qk, capture.modules, args.group_size)
        log("computing forensic residual rows")
        rows = forensic_rows(model, capture.modules, args.group_size)
        rows = reload_compare_for_rows(rows, out_dir)
        rows_sorted = sorted(rows, key=lambda x: x["q1_capture_original_cast_to_deployed_vs_q2_deployed"]["max_abs"], reverse=True)
        fail_rows = [
            row for row in rows_sorted
            if row["q1_capture_original_cast_to_deployed_vs_q2_deployed"]["max_abs"] >= 1e-3
        ]
        summary = {
            "qk_module_count": len(rows),
            "expected_qk_module_count": expected_layers * 2,
            "max_capture_codebook_residual": max(row["q1_capture_perm_vs_qhat_perm"]["max_abs"] for row in rows),
            "max_capture_original_vs_deployed_float": max(row["q1_capture_original_vs_q2_deployed_float"]["max_abs"] for row in rows),
            "max_capture_original_cast_vs_deployed": max(row["q1_capture_original_cast_to_deployed_vs_q2_deployed"]["max_abs"] for row in rows),
            "max_qhat_original_vs_deployed_float": max(row["qhat_original_vs_q2_deployed_float"]["max_abs"] for row in rows),
            "max_qhat_original_cast_vs_deployed": max(row["qhat_original_cast_to_deployed_vs_q2_deployed"]["max_abs"] for row in rows),
            "max_qhat_bf16_vs_deployed": max(row["qhat_bf16_vs_q2_deployed"]["max_abs"] for row in rows),
            "max_q1_bf16_vs_deployed": max(row["q1_bf16_vs_q2_deployed"]["max_abs"] for row in rows),
            "max_qhat_bf16_residual_ulp_ratio": max(row["qhat_bf16_residual_in_bf16_ulp"]["max"] for row in rows),
            "worst_p99_qhat_bf16_residual_ulp_ratio": max(row["qhat_bf16_residual_in_bf16_ulp"]["p99"] for row in rows),
            "max_payload_vs_reload": max(row["qhat_payload_vs_qhat_reload"]["max_abs"] for row in rows),
            "rows_over_1e_3_cast_vs_deployed": len(fail_rows),
            "deployed_T_exact_match_all": all(bool(row["deployed_inferred_T_vs_sidecar_T"]["exact_T_match"]) for row in rows),
            "deployed_T_mismatch_count_total": sum(int(row["deployed_inferred_T_vs_sidecar_T"]["mismatch_count"]) for row in rows),
            "deployed_T_valid_count_total": sum(int(row["deployed_inferred_T_vs_sidecar_T"]["valid_count"]) for row in rows),
            "all_permutation_bijection": all(bool(row["permutation_bijection"]) for row in rows),
            "illegal_T_count": sum(int(row["illegal_T"]) for row in rows),
            "nonfinite_T_count": sum(int(row["nonfinite_T"]) for row in rows),
        }
        result.update({
            "status": "complete",
            "classification": "DONE_QWEN_PT2_RESIDUAL_FORENSICS",
            "summary": summary,
            "top_offenders": rows_sorted[: args.topk],
            "all_rows_path": str(out_dir / "e2_qwen_pt2_forensics_rows.json"),
            "baseline_parity_rows": parity_detail["rows"],
            "quantization_sec": quant_sec,
            "elapsed_sec": time.time() - started,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            },
        })
        write_json(out_dir / "e2_qwen_pt2_forensics_rows.json", {"rows": rows_sorted})
        write_json(result_path, result)
        log(f"wrote {result_path}")
        log(json.dumps({"classification": result["classification"], "summary": summary}, indent=2, ensure_ascii=False))
    except Exception as exc:
        result.update({
            "status": "failed",
            "classification": "FAIL_QWEN_PT2_RESIDUAL_FORENSICS_EXCEPTION",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        write_json(result_path, result)
        log(f"failed; wrote {result_path}")
        raise


if __name__ == "__main__":
    main()
