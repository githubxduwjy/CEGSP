#!/usr/bin/env python3
"""P5-C0: official PT2 numerical-health and evaluator-parity audit.

This is deliberately a baseline-only experiment.  It does not import or run
CEGSP, does not search a CEGSP budget, and does not change PT2's quantization
code.  It runs the released OPT PT2 quantization loop with the preregistered
ATQ settings (and the released SSR variant), records layer/block numerical
statistics, and evaluates the same checkpoint with both the official and the
compact evaluator.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--cegsp-root", default="/root/tqgsp-work")
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--compact-fit-batches", type=int, default=8)
    p.add_argument("--compact-val-batches", type=int, default=8)
    p.add_argument("--compact-untouched-batches", type=int, default=16)
    p.add_argument("--compact-c4-batches", type=int, default=16)
    p.add_argument("--health-ratio-threshold", type=float, default=10.0)
    return p.parse_args()


def finite_float(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def safe_exp(value: Any) -> Optional[float]:
    value = finite_float(value)
    if value is None or value > 700:
        return None
    return finite_float(math.exp(value))


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    x = x.detach().float()
    finite = torch.isfinite(x)
    if not bool(finite.all()):
        return {
            "finite": False,
            "count": int(x.numel()),
            "nonfinite": int((~finite).sum().item()),
        }
    a = x.abs().reshape(-1)
    q = torch.quantile(a, torch.tensor([0.5, 0.99], device=a.device)) if a.numel() else torch.zeros(2, device=a.device)
    return {
        "finite": True,
        "count": int(x.numel()),
        "nonfinite": 0,
        "min": finite_float(x.min().item()) if x.numel() else 0.0,
        "max": finite_float(x.max().item()) if x.numel() else 0.0,
        "abs_max": finite_float(a.max().item()) if a.numel() else 0.0,
        "abs_mean": finite_float(a.mean().item()) if a.numel() else 0.0,
        "abs_p50": finite_float(q[0].item()),
        "abs_p99": finite_float(q[1].item()),
    }


class AuditCapture:
    def __init__(self) -> None:
        self.method = ""
        self.ssr = False
        self.current_t: Optional[torch.Tensor] = None
        self.blocks: List[Dict[str, Any]] = []
        self.modules: List[Dict[str, Any]] = []

    def reset(self, method: str, ssr: bool) -> None:
        self.method = method
        self.ssr = ssr
        self.current_t = None
        self.blocks = []
        self.modules = []

    def record_block(self, q: torch.Tensor, t: Optional[torch.Tensor]) -> None:
        q_cpu = q.detach().float().cpu()
        t_cpu = t.detach().float().cpu() if t is not None else None
        row: Dict[str, Any] = {
            "q_stats": tensor_stats(q_cpu),
            "t_stats": None,
        }
        if t_cpu is not None and t_cpu.shape == q_cpu.shape:
            active = t_cpu.ne(0)
            row["t_stats"] = {
                "finite": bool(torch.isfinite(t_cpu).all()),
                "neg_frac": finite_float(t_cpu.eq(-1).float().mean().item()),
                "zero_frac": finite_float(t_cpu.eq(0).float().mean().item()),
                "pos_frac": finite_float(t_cpu.eq(1).float().mean().item()),
                "nonzero_frac": finite_float(active.float().mean().item()),
                "illegal_frac": finite_float((~t_cpu.eq(-1) & ~t_cpu.eq(0) & ~t_cpu.eq(1)).float().mean().item()),
            }
            denom = t_cpu.pow(2).sum(dim=1).clamp_min(1.0)
            mu = q_cpu.mean(dim=1)
            alpha = ((q_cpu - mu[:, None]) * t_cpu).sum(dim=1) / denom
            row["mu_stats"] = tensor_stats(mu)
            row["alpha_stats"] = tensor_stats(alpha)
        else:
            row["mu_stats"] = None
            row["alpha_stats"] = None
        self.blocks.append(row)


def make_pt2_args(args: argparse.Namespace, method: str, ssr: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method=method,
        nsamples=128,
        percdamp=0.01,
        blocksize=128,
        num_p=1,
        salient_metric="hessian",
        device=args.device,
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=2048,
        ppl_seqlen=2048,
        quant_only="",
        invert=False,
        ssr=ssr,
        log_wandb=False,
        tasks="",
        experiment=args.run_id,
        num_fewshot=0,
        limit=-1,
    )


def patch_opt_position_embeddings_compat(model: torch.nn.Module) -> bool:
    decoder = getattr(getattr(model, "model", None), "decoder", None)
    layers = getattr(decoder, "layers", [])
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


def build_compact_c4_batches(
    tokenizer: Any,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    token_offset: int = 0,
) -> Optional[List[torch.Tensor]]:
    """Build fixed compact C4 validation batches without importing CEGSP."""
    if n_batches <= 0:
        return None
    from datasets import load_dataset

    needed = token_offset + n_batches * batch_size * (seq_len + 1)
    texts: List[str] = []
    rough_tokens = 0
    stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    for row in stream:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        texts.append(text)
        rough_tokens += max(1, len(text) // 4)
        if rough_tokens >= needed * 2:
            ids = tokenizer("\n".join(texts), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if ids.numel() >= needed:
                break
    ids = tokenizer("\n".join(texts), add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() < needed:
        raise RuntimeError(f"not enough C4 validation tokens: have={ids.numel()} need={needed}")
    sliced = ids[token_offset : token_offset + n_batches * batch_size * (seq_len + 1)]
    return [x.clone() for x in sliced.view(n_batches, batch_size, seq_len + 1)]


def install_capture_wrappers(pt2_quantize: Any, gptq_module: Any, gptq_ssr_module: Any, capture: AuditCapture) -> None:
    quantizer_module = importlib.import_module("pt2_llm.quantizer")

    if getattr(install_capture_wrappers, "installed", False):
        return

    original_update = quantizer_module.update_ternary

    @torch.no_grad()
    def update_wrapper(*u_args, **u_kwargs):
        out = original_update(*u_args, **u_kwargs)
        capture.current_t = out.detach().clone()
        return out

    quantizer_module.update_ternary = update_wrapper
    original_quantize = quantizer_module.TernaryQuantizer.quantize

    @torch.no_grad()
    def quantize_wrapper(self, w, *q_args, **q_kwargs):
        capture.current_t = None
        out = original_quantize(self, w, *q_args, **q_kwargs)
        q = out[0] if isinstance(out, tuple) else out
        returned_t = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        t = returned_t
        if t is None or not bool(torch.isfinite(t).all()) or bool(t.abs().sum() == 0):
            t = capture.current_t
        capture.record_block(q, t)
        return out

    quantizer_module.TernaryQuantizer.quantize = quantize_wrapper

    def wrap_fasterquant(cls: Any) -> None:
        original = cls.fasterquant

        @torch.no_grad()
        def fasterquant_wrapper(self, *f_args, **f_kwargs):
            start = len(capture.blocks)
            result = original(self, *f_args, **f_kwargs)
            new_blocks = capture.blocks[start:]
            layer_name = str(getattr(self.layer, "global_name", "unknown"))
            module_row: Dict[str, Any] = {
                "name": layer_name,
                "method": capture.method,
                "ssr": capture.ssr,
                "num_blocks": len(new_blocks),
                "weight_dtype": str(self.layer.weight.dtype),
                "weight_stats": tensor_stats(self.layer.weight.data),
                "block_stats": [],
            }
            for block in new_blocks:
                compact = {k: v for k, v in block.items() if k != "q"}
                module_row["block_stats"].append(compact)
            inp = getattr(self, "inp1", None)
            out = getattr(self, "out1", None)
            if isinstance(inp, torch.Tensor) and isinstance(out, torch.Tensor):
                try:
                    pred = self.layer(inp)
                    delta = (pred.float() - out.float()).detach()
                    module_row["output_reconstruction"] = {
                        "finite": bool(torch.isfinite(delta).all()),
                        "mse": finite_float(delta.pow(2).mean().item()),
                        "rmse": finite_float(delta.pow(2).mean().sqrt().item()),
                        "max_abs": finite_float(delta.abs().max().item()),
                        "sample_shape": list(delta.shape),
                    }
                except Exception as exc:
                    module_row["output_reconstruction"] = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                module_row["output_reconstruction"] = {"available": False}
            capture.modules.append(module_row)
            return result

        cls.fasterquant = fasterquant_wrapper

    wrap_fasterquant(gptq_module.GPTQ)
    wrap_fasterquant(gptq_ssr_module.GPTQ_SSR)
    install_capture_wrappers.installed = True


def compact_evaluate(model: torch.nn.Module, batches: List[torch.Tensor], device: torch.device, evaluator: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # The released official evaluator may offload decoder modules while it
    # iterates.  Restore a single-device state before the compact evaluator;
    # this is evaluator plumbing, not a quantization change.
    model.to(device)
    model.eval()
    for name, rows in batches.items():
        value = evaluator(model, rows, device)
        out[name] = {"nll": finite_float(value), "ppl": safe_exp(value)}
    return out


def official_evaluate(model: torch.nn.Module, loaders: Dict[str, Any], opt_eval: Any, device: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    model.to(device)
    model.eval()
    for dataset, loader in loaders.items():
        value = opt_eval(model, loader, device, dataset, False)
        out[dataset] = {"ppl": finite_float(value), "nll": finite_float(math.log(value)) if finite_float(value) and value > 0 else None}
    return out


def summarize_health(modules: List[Dict[str, Any]], ratio_threshold: float) -> Dict[str, Any]:
    q_abs_max: List[float] = []
    q_p99: List[float] = []
    t_illegal = 0.0
    t_nonfinite = 0
    output_mse: List[float] = []
    for module in modules:
        ws = module.get("weight_stats", {})
        if ws.get("finite"):
            q_abs_max.append(float(ws.get("abs_max", 0.0)))
        for block in module.get("block_stats", []):
            qs = block.get("q_stats", {})
            if qs.get("finite") and qs.get("abs_p99") is not None:
                q_p99.append(float(qs["abs_p99"]))
            ts = block.get("t_stats") or {}
            t_illegal = max(t_illegal, float(ts.get("illegal_frac", 0.0) or 0.0))
            if not ts.get("finite", True):
                t_nonfinite += 1
        rec = module.get("output_reconstruction", {})
        if rec.get("mse") is not None and math.isfinite(float(rec["mse"])):
            output_mse.append(float(rec["mse"]))
    median_p99 = float(torch.tensor(q_p99).median().item()) if q_p99 else None
    max_p99 = max(q_p99) if q_p99 else None
    ratio = (max_p99 / median_p99) if median_p99 and max_p99 is not None else None
    return {
        "all_recorded_values_finite": all(
            module.get("weight_stats", {}).get("finite", False)
            and all(block.get("q_stats", {}).get("finite", False) for block in module.get("block_stats", []))
            for module in modules
        ),
        "num_modules": len(modules),
        "expected_module_count": 24 * 6,
        "expected_module_count_ok": len(modules) == 24 * 6,
        "num_blocks": sum(int(m.get("num_blocks", 0)) for m in modules),
        "q_abs_max_global": max(q_abs_max) if q_abs_max else None,
        "q_abs_p99_median": median_p99,
        "q_abs_p99_max": max_p99,
        "q_abs_p99_max_over_median": ratio,
        "q_outlier_ratio_threshold": ratio_threshold,
        "q_outlier_flag": bool(ratio is not None and ratio > ratio_threshold),
        "t_max_illegal_frac": t_illegal,
        "t_nonfinite_block_count": t_nonfinite,
        "max_output_reconstruction_mse": max(output_mse) if output_mse else None,
        "mean_output_reconstruction_mse": (sum(output_mse) / len(output_mse)) if output_mse else None,
    }


def strip_tensors(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, dict):
        return {str(k): strip_tensors(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_tensors(v) for v in value]
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for P5-C0")
    device = torch.device(args.device)
    out_dir = Path(args.out_dir) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    running = out_dir / "result.running.json"
    running.write_text(json.dumps({"run_id": args.run_id, "status": "running", "config": vars(args)}, indent=2) + "\n")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

    pt2_root = Path(args.pt2_root)
    sys.path.insert(0, str(pt2_root))
    sys.path.insert(0, str(Path(args.cegsp_root)))
    pt2_quantize = importlib.import_module("quantize")
    gptq_module = importlib.import_module("pt2_llm.gptq")
    gptq_ssr_module = importlib.import_module("pt2_llm.gptq_ssr")
    data_module = importlib.import_module("pt2_llm.data")
    eval_module = importlib.import_module("pt2_llm.eval_ppl")
    compact_module = importlib.import_module("tqgsp_support_projection_4090")
    from transformers import AutoTokenizer

    capture = AuditCapture()
    install_capture_wrappers(pt2_quantize, gptq_module, gptq_ssr_module, capture)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit, val, untouched, source = compact_module.build_wikitext_splits(
        tokenizer,
        args.seq_len,
        args.batch_size,
        args.compact_fit_batches,
        args.compact_val_batches,
        args.compact_untouched_batches,
        0,
        0,
    )
    c4 = build_compact_c4_batches(
        tokenizer, args.seq_len, args.batch_size, args.compact_c4_batches, 0
    )
    compact_batches = {"val": val, "untouched_w": untouched}
    if c4 is not None:
        compact_batches["untouched_c4"] = c4

    loader_cache: Dict[str, Any] = {}
    for dataset in ("wikitext2", "c4"):
        _, loader_cache[dataset] = data_module.get_loaders(
            dataset, seed=args.seed, seqlen=2048, model=args.model
        )

    systems: Dict[str, Any] = {}
    logging.info("evaluating clean FP model with official and compact evaluators")
    clean = pt2_quantize.get_model(args.model, 2048)
    clean.eval()
    clean.config.use_cache = False
    systems["fp16_clean"] = {
        "official": official_evaluate(clean, loader_cache, eval_module.opt_eval, str(device)),
        "compact": compact_evaluate(clean, compact_batches, device, compact_module.evaluate_nll),
        "dtype": str(next(clean.parameters()).dtype),
    }
    del clean
    torch.cuda.empty_cache()

    for method, ssr in (("atq", False), ("atq", True)):
        label = f"pt2_{method}_ssr_{str(ssr).lower()}"
        logging.info("running official PT2 configuration: %s", label)
        capture.reset(method, ssr)
        pt2_quantize.args = make_pt2_args(args, method, ssr)
        pt2_quantize.groupsize = 128
        model = pt2_quantize.get_model(args.model, 2048)
        model.eval()
        compat = patch_opt_position_embeddings_compat(model)
        calib_loader, _ = data_module.get_loaders(
            "wikitext2", nsamples=128, seed=args.seed, model=args.model, seqlen=2048
        )
        quant_started = time.time()
        pt2_quantize.quant_sequential(model, calib_loader, str(device))
        quant_sec = time.time() - quant_started
        model.eval()
        official = official_evaluate(model, loader_cache, eval_module.opt_eval, str(device))
        compact = compact_evaluate(model, compact_batches, device, compact_module.evaluate_nll)
        health = summarize_health(capture.modules, args.health_ratio_threshold)
        systems[label] = {
            "config": {
                "method": method,
                "ssr": ssr,
                "nsamples": 128,
                "calib_seqlen": 2048,
                "ppl_seqlen": 2048,
                "group_size": 128,
                "percdamp": 0.01,
                "num_p": 1,
                "salient_metric": "hessian",
                "dtype": str(next(model.parameters()).dtype),
                "quantization_order": ["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
            },
            "compat": {"opt_position_embeddings_kwarg_dropped": compat},
            "quant_sec": quant_sec,
            "official": official,
            "compact": compact,
            "health": health,
            "layerwise_modules": capture.modules,
        }
        del model
        torch.cuda.empty_cache()

    fp_official = systems["fp16_clean"]["official"]
    fp_compact = systems["fp16_clean"]["compact"]
    for label, row in systems.items():
        if label == "fp16_clean":
            continue
        row["evaluator_parity"] = {
            "official_wikitext2_finite": row["official"].get("wikitext2", {}).get("ppl") is not None,
            "official_c4_finite": row["official"].get("c4", {}).get("ppl") is not None,
            "compact_wikitext2_finite": row["compact"].get("untouched_w", {}).get("nll") is not None,
            "compact_c4_finite": row["compact"].get("untouched_c4", {}).get("nll") is not None,
            "official_degrades_vs_fp": {
                d: row["official"].get(d, {}).get("ppl") is not None and fp_official.get(d, {}).get("ppl") is not None and row["official"][d]["ppl"] >= fp_official[d]["ppl"]
                for d in ("wikitext2", "c4")
            },
            "compact_degrades_vs_fp": {
                key: row["compact"].get(key, {}).get("nll") is not None and fp_compact.get(key, {}).get("nll") is not None and row["compact"][key]["nll"] >= fp_compact[key]["nll"]
                for key in ("untouched_w", "untouched_c4") if key in row["compact"] and key in fp_compact
            },
        }

    health_rows = [systems[k]["health"] for k in systems if k != "fp16_clean"]
    result = {
        "run_id": args.run_id,
        "status": "complete",
        "config": vars(args),
        "protocol": {
            "purpose": "P5-C0 PT2 numerical health audit",
            "cegsp_called": False,
            "qat_checkpoint_or_logits_used": False,
            "official_pt2_quantizer": True,
            "official_config": {
                "nsamples": 128,
                "calib_seqlen": 2048,
                "ppl_seqlen": 2048,
                "group_size": 128,
                "percdamp": 0.01,
                "num_p": 1,
                "salient_metric": "hessian",
                "method": "atq",
                "ssr_variant": "atq+ssr",
            },
            "official_quantization_order": ["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2"],
            "compact_eval": {"seq_len": args.seq_len, "source": source},
        },
        "data": {
            "official_eval_datasets": ["wikitext2", "c4"],
            "official_eval_seqlen": 2048,
            "compact_eval": {
                "val_batches": args.compact_val_batches,
                "untouched_w_batches": args.compact_untouched_batches,
                "untouched_c4_batches": args.compact_c4_batches,
            },
        },
        "systems": strip_tensors(systems),
        "health_summary": {
            "all_finite_recorded": all(row.get("all_recorded_values_finite", False) and row.get("expected_module_count_ok", False) for row in health_rows),
            "all_official_and_compact_metrics_finite": all(
                systems[k]["evaluator_parity"]["official_wikitext2_finite"]
                and systems[k]["evaluator_parity"]["official_c4_finite"]
                and systems[k]["evaluator_parity"]["compact_wikitext2_finite"]
                and systems[k]["evaluator_parity"]["compact_c4_finite"]
                for k in systems if k != "fp16_clean"
            ),
            "outlier_ratio_threshold": args.health_ratio_threshold,
            "any_q_outlier_flag": any(row.get("q_outlier_flag", False) for row in health_rows),
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": __import__("transformers").__version__,
            "max_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if running.exists():
        running.unlink()
    logging.info("wrote %s", out_dir / "result.json")


if __name__ == "__main__":
    main()
