#!/usr/bin/env python3
"""R045: test whether held-out trajectory metrics reject a bad hard-T update."""

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import gated_t_gptq_quantize as gated_integration
from pt2_llm import model_utils
from pt2_llm.data import get_loaders
from pt2_llm.model_utils import FPInputsCache, find_layers
from pt2_llm.quantizer import TernaryQuantizer
from pt2_llm.gptq import GPTQ
import quantize as pt2_quantize


SEQUENTIAL = [
    ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
    ["self_attn.o_proj"],
    ["mlp.up_proj", "mlp.gate_proj"],
    ["mlp.down_proj"],
]


def make_qargs(args, device):
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.candidate_nsamples,
        percdamp=0.01,
        blocksize=args.blocksize,
        num_p=1,
        salient_metric="hessian",
        device=device,
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=args.seqlen,
        ppl_seqlen=args.seqlen,
        quant_only="",
        invert=False,
        ssr=False,
        log_wandb=False,
        tasks="",
        experiment="r045_trajectory_gate",
        num_fewshot=0,
        limit=-1,
    )


def set_global_names(model, model_name):
    for name, module in model.named_modules():
        module.global_name = model_name + name


def move_llama_front(model, device):
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    model.model.norm = model.model.norm.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)


def move_llama_front_cpu(model):
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()


@torch.no_grad()
def capture_layer0_inputs(model, dataloader, nsamples, device, dtype):
    layers = model.model.layers
    layers[0] = layers[0].to(device)
    move_llama_front(model, device)
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    cache = {
        "i": 0,
        "attention_mask": None,
        "position_ids": None,
        "position_embeddings": None,
    }

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader[:nsamples]:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    move_llama_front_cpu(model)
    torch.cuda.empty_cache()
    return inps, cache


@torch.no_grad()
def quantize_layer0(model, dataloader, args, variant, device):
    pt2_quantize.args = make_qargs(args, device)
    pt2_quantize.groupsize = args.blocksize
    set_global_names(model, args.model)
    gated_integration.VALIDATION_FRACTION = args.validation_fraction
    gated_integration.MAX_STEPS = args.max_steps
    gated_integration.GATED_STATS.clear()

    dtype = next(iter(model.parameters())).dtype
    inps, cache = capture_layer0_inputs(
        model, dataloader, args.candidate_nsamples, device, dtype
    )
    layer = model.model.layers[0].to(device)
    full = find_layers(layer)

    fp_inputs_cache = FPInputsCache(SEQUENTIAL)
    fp_inps = inps.clone()
    fp_inputs_cache.add_hook(full)
    for j in range(args.candidate_nsamples):
        fp_inps[j] = layer(
            fp_inps[j].unsqueeze(0),
            attention_mask=cache["attention_mask"],
            position_ids=cache["position_ids"],
            position_embeddings=cache["position_embeddings"],
        )[0]
    fp_inputs_cache.clear_hook()

    gptq_cls = GPTQ
    if variant == "hard_t_layer0":
        gptq_cls = gated_integration.GPTQValidationGated

    for names in SEQUENTIAL:
        subset = {name: full[name] for name in names}
        gptq = {}
        for name, module in subset.items():
            quantizer = TernaryQuantizer(
                module.weight,
                method=args.low_quant_method,
                groupsize=args.blocksize,
            )
            gptq[name] = gptq_cls(
                module,
                quantizer,
                salient_metric="hessian",
                disable_gptq=False,
                method=args.low_quant_method,
                gptaq=True,
                reorder=False,
            )
            gptq[name].fp_inp = fp_inputs_cache.fp_cache[name]

        def add_batch(name):
            def hook(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return hook

        handles = []
        for name, module in subset.items():
            handles.append(module.register_forward_hook(add_batch(name)))
        for j in range(args.candidate_nsamples):
            layer(
                inps[j].unsqueeze(0),
                attention_mask=cache["attention_mask"],
                position_ids=cache["position_ids"],
                position_embeddings=cache["position_embeddings"],
            )[0]
        for handle in handles:
            handle.remove()

        for name in names:
            gptq[name].fasterquant(
                percdamp=0.01,
                blocksize=args.blocksize,
                num_p=1,
                disable_mask=True,
            )
            gptq[name].free()

    fp_inputs_cache.clear_cache()
    layer.cpu()
    torch.cuda.empty_cache()
    stats = deepcopy(gated_integration.GATED_STATS)
    return stats


def token_nll(logits, input_ids):
    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = input_ids[:, 1:]
    return F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        reduction="none",
    )


def rms(x):
    return torch.sqrt(torch.mean(x.float().square()).clamp_min(1e-30))


@torch.no_grad()
def score_model(model, score_batches, device, fp_reference=None):
    model.to(device)
    model.eval()
    rows = []
    captured = {}

    def hook(_, __, out):
        value = out[0] if isinstance(out, tuple) else out
        captured["layer0_out"] = value.detach()

    handle = model.model.layers[0].register_forward_hook(hook)
    try:
        for seq_id, batch in enumerate(score_batches):
            input_ids = batch[0].to(device)
            captured.clear()
            outputs = model(input_ids, use_cache=False)
            layer0_out = captured["layer0_out"]
            nll = token_nll(outputs.logits, input_ids).detach().cpu()
            row = {
                "sequence": seq_id,
                "mean_token_nll": float(nll.mean().item()),
                "nonfinite_count": int((~torch.isfinite(outputs.logits)).sum().item())
                + int((~torch.isfinite(layer0_out)).sum().item()),
            }
            if fp_reference is None:
                row["token_nll"] = nll
                row["layer0_out"] = layer0_out.detach().cpu().float()
            else:
                fp_nll = fp_reference[seq_id]["token_nll"]
                fp_layer0 = fp_reference[seq_id]["layer0_out"].to(device)
                diff = layer0_out.float() - fp_layer0.float()
                denom = fp_layer0.float().square().sum().clamp_min(1e-30)
                row["layer0_nmse"] = float((diff.square().sum() / denom).item())
                row["layer0_cosine_drift"] = float(
                    1.0
                    - F.cosine_similarity(
                        layer0_out.float().flatten(),
                        fp_layer0.float().flatten(),
                        dim=0,
                    ).item()
                )
                delta = nll - fp_nll
                k = max(1, math.ceil(delta.numel() * 0.10))
                row["mean_nll_increase"] = float(delta.mean().item())
                row["cvar10_nll_increase"] = float(torch.topk(delta, k).values.mean().item())
                ratio = float((rms(layer0_out) / rms(fp_layer0)).item())
                row["layer0_rms_ratio"] = ratio
                row["layer0_rms_abs_log_ratio"] = float(abs(math.log(max(ratio, 1e-30))))
            rows.append(row)
            del outputs, layer0_out
            torch.cuda.empty_cache()
    finally:
        handle.remove()
        model.cpu()
        torch.cuda.empty_cache()
    return rows


def strip_reference(rows):
    stripped = []
    for row in rows:
        stripped.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"token_nll", "layer0_out"}
            }
        )
    return stripped


def build_gate(fixed_rows, hard_rows):
    lower_is_better = [
        "layer0_nmse",
        "layer0_cosine_drift",
        "mean_token_nll",
        "mean_nll_increase",
        "cvar10_nll_increase",
        "layer0_rms_abs_log_ratio",
        "nonfinite_count",
    ]
    metrics = {}
    for metric in lower_is_better:
        per_sequence = []
        for fixed, hard in zip(fixed_rows, hard_rows):
            per_sequence.append(
                {
                    "sequence": fixed["sequence"],
                    "fixed": fixed[metric],
                    "hard": hard[metric],
                    "rejects_hard": fixed[metric] < hard[metric],
                }
            )
        metrics[metric] = {
            "rejects_hard_all_sequences": all(
                item["rejects_hard"] for item in per_sequence
            ),
            "per_sequence": per_sequence,
        }
    successful = [
        metric
        for metric, value in metrics.items()
        if value["rejects_hard_all_sequences"]
    ]
    return {
        "pass": bool(successful),
        "successful_metrics": successful,
        "metrics": metrics,
    }


def summarize_variant(rows):
    numeric_keys = [key for key in rows[0] if key != "sequence"]
    summary = {}
    for key in numeric_keys:
        values = [row[key] for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-nsamples", type=int, default=8)
    parser.add_argument("--score-nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--low-quant-method", default="atq")
    args = parser.parse_args()

    started = time.time()
    device = "cuda:0"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    model = pt2_quantize.get_model(args.model, args.seqlen)
    model.eval()
    model.config.use_cache = False
    model.seqlen = args.seqlen

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.candidate_nsamples + args.score_nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
    )
    score_batches = dataloader[args.candidate_nsamples :]
    original_layer0 = deepcopy(model.model.layers[0].state_dict())

    fp_rows = score_model(model, score_batches, device, fp_reference=None)

    model.model.layers[0].load_state_dict(original_layer0)
    fixed_stats = quantize_layer0(model, dataloader, args, "fixed_t_layer0", device)
    fixed_rows = score_model(model, score_batches, device, fp_reference=fp_rows)

    model.model.layers[0].load_state_dict(original_layer0)
    hard_stats = quantize_layer0(model, dataloader, args, "hard_t_layer0", device)
    hard_rows = score_model(model, score_batches, device, fp_reference=fp_rows)

    gate = build_gate(fixed_rows, hard_rows)
    result = {
        "config": vars(args),
        "split_samples": {
            "candidate": args.candidate_nsamples,
            "score": args.score_nsamples,
        },
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "fp16": strip_reference(fp_rows),
        "fixed_t_layer0": fixed_rows,
        "hard_t_layer0": hard_rows,
        "summary": {
            "fixed_t_layer0": summarize_variant(fixed_rows),
            "hard_t_layer0": summarize_variant(hard_rows),
        },
        "gated_stats": hard_stats,
        "fixed_stats": fixed_stats,
        "gate": gate,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "gate": gate}, indent=2), flush=True)
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
