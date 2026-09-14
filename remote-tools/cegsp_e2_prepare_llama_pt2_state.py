#!/usr/bin/env python3
"""Prepare a reusable Llama-2-7B PT2 full-state checkpoint for E2 lmeval.

This is a preparation step, not a downstream result.  It runs the official PT2
ATQ+SSR path once, exports the detached Q/K sidecar used by TernRefine patches,
and saves a compact full-module checkpoint so downstream endpoints can be
evaluated without re-running PT2 quantization.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import set_seed

from cegsp_e2_qwen_pt2_hba_skipgate_a100 import build_full_pt2_state, save_full_pt2_state
from cegsp_p7_a100_scaling import get_decoder_layers
from cegsp_p9s2_detached_pt2_plugin import (
    Capture,
    build_codes,
    detached_reload_gate,
    install_capture,
    make_pt2_args,
    parity_gate,
    save_detached_artifacts,
    snapshot_qk,
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/Llama-2-7b-hf")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--pt2-data-root", default="/root/PT2-data")
    p.add_argument("--run-id", required=True)
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--calib-seq-len", type=int, default=2048)
    p.add_argument("--ppl-seq-len", type=int, default=2048)
    p.add_argument("--percdamp", type=float, default=0.01)
    p.add_argument("--num-p", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sidecar-dir-name", default="pt2_llama7b_sidecar")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.group_size != 128 or args.nsamples != 128 or args.calib_seq_len != 2048:
        raise ValueError("E2 Llama PT2 preparation is frozen to group=128, nsamples=128, seq_len=2048")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    out = Path(args.out_dir) / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "prepare_partial.json"
    result_path = out / "prepare_result.json"
    write_json(partial_path, {"status": "started", "config": vars(args)})

    sys.path.insert(0, args.pt2_root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    os.environ["PT2_DATA_ROOT"] = args.pt2_data_root
    data = importlib.import_module("pt2_llm.data")
    if hasattr(data, "DATA_ROOT"):
        data.DATA_ROOT = args.pt2_data_root
    pt2_quantize = importlib.import_module("quantize")
    qmod = importlib.import_module("pt2_llm.quantizer")
    gptqmod = importlib.import_module("pt2_llm.gptq")
    ssrmod = importlib.import_module("pt2_llm.gptq_ssr")
    pt2_quantize.args = make_pt2_args(args)
    pt2_quantize.groupsize = args.group_size

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    log(f"loading calibration nsamples={args.nsamples}")
    calib_loader, _ = data.get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.calib_seq_len,
        model=args.model,
    )
    log(f"loading Llama model {args.model}")
    model = pt2_quantize.get_model(args.model, args.calib_seq_len)
    model.seqlen = args.calib_seq_len
    model.eval()
    layers = list(range(len(get_decoder_layers(model))))
    fp_qk = snapshot_qk(model, layers)
    write_json(partial_path, {"status": "model_loaded", "decoder_layers": len(layers), "elapsed_sec": time.time() - started})

    capture = Capture()
    capture.store_all_t = True
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
    model.eval()

    pt2_qk = snapshot_qk(model, layers)
    codes, perms, parity_detail = build_codes(model, fp_qk, capture.modules, args.group_size)
    parity = parity_gate(codes, parity_detail, perms, args.group_size)
    sidecar_metadata = save_detached_artifacts(out, args, codes, perms, fp_qk, pt2_qk, parity)
    reload_gate = detached_reload_gate(model, codes, perms, pt2_qk)
    full_state = build_full_pt2_state(model, capture.modules, args.group_size)
    checkpoint_manifest = save_full_pt2_state(full_state, out)

    result = {
        "status": "complete",
        "run_id": args.run_id,
        "config": vars(args),
        "protocol": {
            "preparation_only": True,
            "official_pt2": "ATQ+SSR",
            "dataset": "wikitext2",
            "nsamples": args.nsamples,
            "seq_len": args.calib_seq_len,
            "group_size": args.group_size,
            "downstream_evaluated": False,
        },
        "decoder_layers": len(layers),
        "qk_modules": sum(len(v) for v in codes.values()),
        "captured_quantized_modules": len(capture.modules),
        "state_parity": parity,
        "detached_reload_gate": reload_gate,
        "sidecar": sidecar_metadata["artifacts"],
        "checkpoint": checkpoint_manifest,
        "timing": {"quant_sec": quant_sec, "elapsed_sec": time.time() - started},
    }
    write_json(result_path, result)
    write_json(partial_path, {"status": "complete", "result": str(result_path), "elapsed_sec": time.time() - started})
    log(f"wrote {result_path}")


if __name__ == "__main__":
    main()
