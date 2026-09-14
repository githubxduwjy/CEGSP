#!/usr/bin/env python3
"""E2-A0: Qwen3-8B official PT2 health and ternary-state export gate.

This is the Qwen analogue of the Llama PT2 sidecar health check.  It reuses
the audited Llama capture helpers, but all layer/module counts are inferred
from the loaded model so the gate can test whether the official PT2 pipeline
actually supports Qwen3-8B before any TernRefine claim is attempted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

import torch
from transformers import set_seed

from cegsp_p7_a100_scaling import AffineCode, audit_all, get_decoder_layers, target_qk
from cegsp_p9s2_detached_pt2_plugin import (
    Capture,
    apply_ssr_codes,
    build_codes,
    finite_metrics,
    install_capture,
    module_specs,
    official_metrics,
    save_detached_artifacts,
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
    p.add_argument("--sidecar-dir-name", default="pt2_qwen3_8b_sidecar")
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


def dynamic_parity_gate(
    codes: Dict[int, Dict[str, AffineCode]],
    parity_detail: Dict[str, object],
    perms: Dict[int, Dict[str, torch.Tensor]],
    group_size: int,
    expected_layers: int,
) -> Dict[str, object]:
    rows = parity_detail["rows"]
    expected_qk = expected_layers * 2
    code_audit = audit_all(codes)
    max_capture_residual = max((float(row["capture_codebook_residual"]) for row in rows), default=float("inf"))
    max_deployed_residual = max((float(row["final_vs_capture_q_max_abs"]) for row in rows), default=float("inf"))
    illegal = sum(int(row["illegal_T"]) for row in rows)
    nonfinite = sum(int(row["nonfinite_T"]) for row in rows)
    all_bijection = all(bool(row["permutation_bijection"]) for row in rows)
    layer_ok = sorted(codes) == list(range(expected_layers)) and all(sorted(v) == ["k", "q"] for v in codes.values())
    passed = (
        len(rows) == expected_qk
        and group_size == 128
        and illegal == 0
        and nonfinite == 0
        and all_bijection
        and max_capture_residual < 1e-3
        and max_deployed_residual < 1e-3
        and code_audit["total_illegal_states"] == 0
        and layer_ok
    )
    return {
        "pass": passed,
        "qk_module_count": len(rows),
        "expected_qk_module_count": expected_qk,
        "expected_layers": expected_layers,
        "captured_layers": sorted(int(x) for x in codes),
        "layer_scope_pass": layer_ok,
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


def dynamic_detached_reload_gate(
    model: torch.nn.Module,
    codes: Dict[int, Dict[str, AffineCode]],
    perms: Dict[int, Dict[str, torch.Tensor]],
    qk_checkpoint: Dict[int, Dict[str, torch.Tensor]],
    expected_layers: int,
) -> Dict[str, object]:
    apply_ssr_codes(model, codes, perms, None)
    rows = []
    for layer, layer_codes in codes.items():
        refs = target_qk(model, layer)
        for key, code in layer_codes.items():
            q_perm = (code.mu + code.alpha * code.T.float()).view(code.original_shape[0], -1)[:, : code.original_shape[1]]
            q_original = q_perm[:, torch.argsort(perms[layer][key])]
            saved = qk_checkpoint[layer][key].float()
            deployed = refs[key].module.weight.detach().float().cpu()
            rows.append({
                "layer": int(layer),
                "key": key,
                "sidecar_vs_saved_max_abs": float((q_original - saved).abs().max().item()),
                "deployed_vs_saved_max_abs": float((deployed - saved).abs().max().item()),
                "ssr_bijection": sorted(perms[layer][key].tolist()) == list(range(code.original_shape[1])),
            })
    expected_qk = expected_layers * 2
    max_sidecar = max(float(row["sidecar_vs_saved_max_abs"]) for row in rows)
    max_deployed = max(float(row["deployed_vs_saved_max_abs"]) for row in rows)
    return {
        "pass": len(rows) == expected_qk and max_sidecar < 1e-3 and max_deployed < 1e-3 and all(bool(row["ssr_bijection"]) for row in rows),
        "qk_module_count": len(rows),
        "expected_qk_module_count": expected_qk,
        "max_sidecar_vs_saved_q_residual": max_sidecar,
        "max_deployed_vs_saved_q_residual": max_deployed,
        "module_rows": rows,
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def apply_qwen_pt2_layer_adapter(model: torch.nn.Module, model_path: str) -> Dict[str, object]:
    """Add the minimal metadata expected by this PT2 fork's Qwen branch.

    The fork already contains Qwen traversal logic, but assumes decoder layers
    expose ``attention_type``.  Newer Qwen3 HF layers do not.  Setting a passive
    attribute keeps the official PT2 quantization/evaluation path unchanged
    while making the interface explicit for this health gate.
    """
    if "qwen" not in model_path.lower():
        return {"applied": False, "reason": "non-qwen model"}
    layers = get_decoder_layers(model)
    patched = 0
    for layer in layers:
        if not hasattr(layer, "attention_type"):
            setattr(layer, "attention_type", None)
            patched += 1
    return {
        "applied": True,
        "patched_layers": patched,
        "total_layers": len(layers),
        "value": None,
        "reason": "PT2 qwen branch expects layer.attention_type",
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    result_path = out_dir / "e2_qwen_pt2_health_result.json"

    result: Dict[str, object] = {
        "run_id": args.run_id,
        "experiment": "E2-A0 Qwen3-8B official PT2 health and sidecar export gate",
        "status": "started",
        "config": vars(args),
    }
    write_json(out_dir / "e2_qwen_pt2_health_partial.json", result)

    try:
        if args.group_size != 128 or args.calib_seq_len != 2048 or args.ppl_seq_len != 2048:
            raise ValueError("E2-A0 is frozen to group=128 and seq_len=2048")
        if not torch.cuda.is_available():
            raise RuntimeError("E2-A0 requires CUDA")
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device("cuda")

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

        log(f"loading calibration via official PT2 loader nsamples={args.nsamples}")
        calib_loader, _ = data.get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=args.calib_seq_len, model=args.model)
        if len(calib_loader) != args.nsamples:
            raise RuntimeError(f"calibration sample mismatch {len(calib_loader)} != {args.nsamples}")

        log(f"loading official PT2 model={args.model} method=atq ssr=True")
        model = pt2_quantize.get_model(args.model, args.calib_seq_len)
        architecture_adapter = apply_qwen_pt2_layer_adapter(model, args.model)
        model.seqlen = args.calib_seq_len
        model.eval()
        expected_layers = len(get_decoder_layers(model))
        layers = list(range(expected_layers))
        specs = module_specs(model)
        fp_qk = snapshot_qk(model, layers)
        result.update({
            "status": "model_loaded",
            "model_type": getattr(model.config, "model_type", None),
            "decoder_layers": expected_layers,
            "expected_qk_modules": expected_layers * 2,
            "expected_quantized_modules": len(specs),
            "architecture_adapter": architecture_adapter,
        })
        write_json(out_dir / "e2_qwen_pt2_health_partial.json", result)

        capture = Capture()
        restore_capture = install_capture(qmod, gptqmod, ssrmod, capture, args.group_size)
        quant_started = time.time()
        log("starting official PT2 quant_sequential")
        try:
            pt2_quantize.quant_sequential(model, calib_loader, "cuda:0")
        finally:
            restore_capture()
        quant_sec = time.time() - quant_started
        model.to(device)
        model.config.use_cache = False
        pt2_qk = snapshot_qk(model, layers)

        codes, perms, parity_detail = build_codes(model, fp_qk, capture.modules, args.group_size)
        parity = dynamic_parity_gate(codes, parity_detail, perms, args.group_size, expected_layers)
        sidecar_metadata = save_detached_artifacts(out_dir, args, codes, perms, fp_qk, pt2_qk, parity)
        detached_reload = dynamic_detached_reload_gate(model, codes, perms, pt2_qk, expected_layers)

        metrics = None
        metrics_finite = False
        if parity["pass"] and detached_reload["pass"]:
            log("evaluating official W2/C4 finite health")
            data = importlib.import_module("pt2_llm.data")
            if hasattr(data, "DATA_ROOT"):
                data.DATA_ROOT = args.pt2_data_root
            metrics = official_metrics(model, args.model, device, args.pt2_data_root, args.ppl_seq_len)
            metrics_finite = finite_metrics(metrics)

        health_pass = bool(parity["pass"] and detached_reload["pass"] and metrics_finite)
        result.update({
            "status": "complete",
            "classification": "PASS_QWEN_PT2_HEALTH" if health_pass else "FAIL_QWEN_PT2_HEALTH",
            "protocol": {
                "pt2_method": "official quant_sequential ATQ+SSR",
                "health_budget_only": True,
                "not_quality_baseline": True,
                "nsamples": args.nsamples,
                "seq_len": args.calib_seq_len,
                "group_size": args.group_size,
                "scope": "all decoder layers, Q/K state export audited",
                "ternrefine_not_run": True,
            },
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "bf16": torch.cuda.is_bf16_supported(),
                "max_memory_gb": torch.cuda.max_memory_allocated() / (1024**3),
            },
            "capture": {
                "quantizer_calls": capture.total_quantizer_calls,
                "module_calls": len(capture.modules),
                "expected_module_calls": len(specs),
            },
            "state_parity": parity,
            "detached_reload_gate": detached_reload,
            "official_metrics": metrics,
            "official_metrics_finite": metrics_finite,
            "detached_artifacts": sidecar_metadata["artifacts"],
            "quantization_sec": quant_sec,
            "elapsed_sec": time.time() - started,
        })
        write_json(result_path, result)
        log(f"wrote {result_path}")
        log(json.dumps({"classification": result["classification"], "metrics": metrics}, indent=2, ensure_ascii=False))
    except Exception as exc:
        result.update({
            "status": "failed",
            "classification": "FAIL_QWEN_PT2_HEALTH_EXCEPTION",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.time() - started,
        })
        write_json(result_path, result)
        log(f"failed; wrote {result_path}")
        raise


if __name__ == "__main__":
    main()
