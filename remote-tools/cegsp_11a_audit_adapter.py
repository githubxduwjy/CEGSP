#!/usr/bin/env python3
"""CEGSP-11A: matched-split audit for direct ternary vs official PT2 ATQ.

This adapter does not modify PT2. It imports the official PT2 quantization
loop, quantizes OPT-family models, then evaluates the quantized model with the
same compact NLL evaluator and Wikitext/C4 batches used by the CEGSP scripts.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import sys
import time
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default="CEGSP-11A-AUDIT-OPT350M")
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--cegsp-root", default="/root/tqgsp-work")
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--fit-batches", type=int, default=8)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--untouched-batches", type=int, default=32)
    p.add_argument("--c4-untouched-batches", type=int, default=32)
    p.add_argument("--fit-token-offset", type=int, default=0)
    p.add_argument("--val-token-offset", type=int, default=0)
    p.add_argument("--c4-token-offset", type=int, default=0)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--threshold-factor", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    p.add_argument("--pt2-methods", default="atq")
    p.add_argument("--pt2-ssr-methods", default="")
    p.add_argument("--pt2-calib-seq-len", type=int, default=0)
    p.add_argument("--pt2-calib-samples", type=int, default=0)
    p.add_argument("--skip-direct", action="store_true")
    return p.parse_args()


def as_ppl(nll: float) -> float:
    if not isinstance(nll, (int, float)) or not math.isfinite(nll):
        return float("nan")
    return float(math.exp(nll))


def evaluate_pack(model, batches, c4_batches, evaluate_nll, device) -> Dict[str, float]:
    result = {
        "val": evaluate_nll(model, batches["val"], device),
        "untouched_w": evaluate_nll(model, batches["untouched_w"], device),
    }
    if c4_batches is not None:
        result["untouched_c4"] = evaluate_nll(model, c4_batches, device)
    return result


def make_pt2_args(args: argparse.Namespace, method: str, ssr: bool, device: str) -> SimpleNamespace:
    calib_seq_len = args.pt2_calib_seq_len or args.seq_len
    calib_samples = args.pt2_calib_samples or (args.fit_batches * args.batch_size)
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method=method,
        nsamples=calib_samples,
        percdamp=0.01,
        blocksize=args.group_size,
        num_p=1,
        salient_metric="hessian",
        device=device,
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=calib_seq_len,
        ppl_seqlen=args.seq_len,
        quant_only="",
        invert=False,
        ssr=ssr,
        log_wandb=False,
        tasks="",
        experiment=args.run_id,
        num_fewshot=0,
        limit=-1,
    )


def make_pt2_calib_loader(fit_batches: List[torch.Tensor]) -> List[tuple[torch.Tensor]]:
    loader: List[tuple[torch.Tensor]] = []
    for batch in fit_batches:
        inputs = batch[:, :-1].contiguous()
        for row_idx in range(inputs.shape[0]):
            loader.append((inputs[row_idx : row_idx + 1].contiguous(),))
    return loader


def build_long_wikitext_calib_loader(tokenizer, args: argparse.Namespace) -> List[tuple[torch.Tensor]]:
    calib_seq_len = args.pt2_calib_seq_len or args.seq_len
    calib_samples = args.pt2_calib_samples or (args.fit_batches * args.batch_size)
    if calib_seq_len == args.seq_len and calib_samples == args.fit_batches * args.batch_size:
        return []
    try:
        from datasets import load_dataset

        train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        train_text = "\n".join(x["text"] for x in train if x["text"].strip())
    except Exception:
        train_text, _ = importlib.import_module("tqgsp_support_projection_4090").read_wikitext_arrow_cache()
    ids = tokenizer(train_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    needed = args.fit_token_offset + calib_samples * calib_seq_len
    if ids.numel() < needed:
        raise RuntimeError(f"not enough PT2 calibration tokens: have={ids.numel()} need={needed}")
    rows = []
    start = args.fit_token_offset
    for idx in range(calib_samples):
        lo = start + idx * calib_seq_len
        rows.append((ids[lo : lo + calib_seq_len].view(1, calib_seq_len).contiguous(),))
    return rows


def patch_opt_position_embeddings_compat(model) -> bool:
    """Drop PT2's extra OPT kwarg when the installed transformers OPT lacks it."""
    if "opt" not in str(getattr(getattr(model, "config", None), "model_type", "")).lower():
        return False
    layers = getattr(getattr(model, "model", None), "decoder", None).layers
    patched = False
    for layer in layers:
        forward = layer.forward
        if "position_embeddings" in inspect.signature(forward).parameters:
            continue

        def wrapped_forward(*f_args, __forward=forward, **f_kwargs):
            f_kwargs.pop("position_embeddings", None)
            return __forward(*f_args, **f_kwargs)

        layer.forward = wrapped_forward
        patched = True
    return patched


@torch.no_grad()
def run_official_pt2(args, method: str, ssr: bool, calib_loader, eval_batches, c4_batches, evaluate_nll, device):
    pt2_root = Path(args.pt2_root)
    sys.path.insert(0, str(pt2_root))
    pt2_quantize = importlib.import_module("quantize")
    pt2_quantize.args = make_pt2_args(args, method, ssr, str(device))
    pt2_quantize.groupsize = args.group_size

    started = time.time()
    calib_seq_len = args.pt2_calib_seq_len or args.seq_len
    model = pt2_quantize.get_model(args.model, calib_seq_len)
    model.seqlen = calib_seq_len
    model.eval()
    compat_patched = patch_opt_position_embeddings_compat(model)
    quant_started = time.time()
    pt2_quantize.quant_sequential(model, calib_loader, str(device))
    quant_sec = time.time() - quant_started
    model.to(device)
    model.config.use_cache = False
    nll = evaluate_pack(model, eval_batches, c4_batches, evaluate_nll, device)
    total_sec = time.time() - started
    del model
    torch.cuda.empty_cache()
    return {
        "method": method,
        "ssr": ssr,
        "nll": nll,
        "ppl": {k: as_ppl(v) for k, v in nll.items()},
        "quant_sec": quant_sec,
        "total_sec": total_sec,
        "compat": {"opt_position_embeddings_kwarg_dropped": compat_patched},
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    sys.path.insert(0, args.cegsp_root)
    cegsp = importlib.import_module("cegsp_ce_gradient_4090")
    cegsp.set_seed(args.seed)

    logging.info("loading tokenizer and matched eval batches")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, data_source = cegsp.build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.fit_batches,
        args.val_batches,
        args.untouched_batches,
        args.fit_token_offset,
        args.val_token_offset,
    )
    c4_batches = cegsp.build_c4_untouched_batches(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.c4_untouched_batches,
        args.c4_token_offset,
    )
    eval_batches = {"fit": fit, "val": val, "untouched_w": untouched}
    compact_pt2_calib_loader = make_pt2_calib_loader(fit)
    long_pt2_calib_loader = build_long_wikitext_calib_loader(tokenizer, args)
    pt2_calib_loader = long_pt2_calib_loader or compact_pt2_calib_loader

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    systems: Dict[str, Dict[str, object]] = {}

    logging.info("evaluating FP reference")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    fp_nll = evaluate_pack(model, eval_batches, c4_batches, cegsp.evaluate_nll, device)
    systems["fp"] = {
        "nll": fp_nll,
        "ppl": {k: as_ppl(v) for k, v in fp_nll.items()},
        "total_sec": time.time() - t0,
    }

    if not args.skip_direct:
        logging.info("evaluating canonical direct ternary")
        t0 = time.time()
        quant_counts = cegsp.apply_direct_ptq_local(model, args.group_size, args.threshold_factor)
        direct_nll = evaluate_pack(model, eval_batches, c4_batches, cegsp.evaluate_nll, device)
        systems["direct_ternary"] = {
            "nll": direct_nll,
            "ppl": {k: as_ppl(v) for k, v in direct_nll.items()},
            "quant_counts": quant_counts,
            "total_sec": time.time() - t0,
        }
    del model
    torch.cuda.empty_cache()

    pt2_results: List[Dict[str, object]] = []
    for method in [x.strip() for x in args.pt2_methods.split(",") if x.strip()]:
        logging.info("running official PT2 method=%s ssr=False", method)
        pt2_results.append(
            run_official_pt2(args, method, False, pt2_calib_loader, eval_batches, c4_batches, cegsp.evaluate_nll, device)
        )
    for method in [x.strip() for x in args.pt2_ssr_methods.split(",") if x.strip()]:
        logging.info("running official PT2 method=%s ssr=True", method)
        pt2_results.append(
            run_official_pt2(args, method, True, pt2_calib_loader, eval_batches, c4_batches, cegsp.evaluate_nll, device)
        )
    for row in pt2_results:
        key = f"pt2_{row['method']}_ssr_{row['ssr']}"
        systems[key] = row

    result = {
        "run_id": args.run_id,
        "config": vars(args),
        "data": {
            "source": data_source,
            "split": "CEGSP matched Wikitext-2 train/validation plus report-only C4 validation batches",
            "fit_batches": args.fit_batches,
            "pt2_calib_seq_len": args.pt2_calib_seq_len or args.seq_len,
            "pt2_calib_samples": args.pt2_calib_samples or (args.fit_batches * args.batch_size),
            "val_batches": args.val_batches,
            "untouched_w_batches": args.untouched_batches,
            "untouched_c4_batches": args.c4_untouched_batches,
        },
        "systems": systems,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "elapsed_sec": time.time() - started,
        "clean_room_invariants": {
            "uses_qat_checkpoint": False,
            "uses_qat_logits": False,
            "uses_qat_latent_weights": False,
            "uses_optimizer_steps": False,
            "uses_official_pt2_quantizer": True,
            "uses_matched_cegsp_eval_batches": True,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    logging.info("wrote %s", out_dir / "result.json")


if __name__ == "__main__":
    main()
