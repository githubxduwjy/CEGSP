#!/usr/bin/env python3
"""R047: screen adjacent hard-T interactions with local and joint metrics."""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import gated_t_gptq_quantize as gated_integration
import quantize as pt2_quantize
from pt2_llm import model_utils
from pt2_llm.data import get_loaders


def parse_layers(text):
    if not text:
        return set()
    return {int(value.strip()) for value in text.split(",") if value.strip()}


def make_qargs(args, variant, device):
    return SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.calib_nsamples,
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
        experiment=variant,
        num_fewshot=0,
        limit=-1,
    )


def configure_quantization(args, variant, gated_layers, device):
    pt2_quantize.args = make_qargs(args, variant, device)
    pt2_quantize.groupsize = args.blocksize
    gated_integration.GATED_STATS.clear()
    gated_integration.GATED_LAYERS = set(gated_layers)
    gated_integration.GATED_PROJECTIONS = set()
    gated_integration.VALIDATION_FRACTION = args.validation_fraction
    gated_integration.MAX_STEPS = args.max_steps
    if gated_layers:
        pt2_quantize.GPTQ = gated_integration.GPTQSelectiveValidationGated


@torch.no_grad()
def quantize_model(model, dataloader, args, variant, gated_layers, device):
    configure_quantization(args, variant, gated_layers, device)
    started = time.time()
    pt2_quantize.quant_sequential(model, dataloader, device)
    return {
        "quant_seconds": time.time() - started,
        "gated_layers": sorted(gated_layers),
        "gated_stats": list(gated_integration.GATED_STATS),
    }


def nmse(value, reference):
    diff = value.float() - reference.float()
    denom = reference.float().square().sum().clamp_min(1e-30)
    return float((diff.square().sum() / denom).item())


def cosine_drift(value, reference):
    return float(
        1.0
        - F.cosine_similarity(
            value.float().flatten(),
            reference.float().flatten(),
            dim=0,
        ).item()
    )


@torch.no_grad()
def score_dataset(model, testenc, args, device, fp_reference=None):
    model.seqlen = args.seqlen
    token_ids = testenc.input_ids[:, : args.score_nsamples * args.seqlen]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.score_nsamples, args.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    cache = {"i": 0, "attention_mask": None, "position_embeddings": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs.get("position_embeddings", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(args.score_nsamples):
        batch = token_ids[:, i * args.seqlen : (i + 1) * args.seqlen].to(device)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    checkpoint_layers = set(args.window_layers)
    hidden_refs = None
    if fp_reference is not None:
        hidden_refs = {
            row["sequence"]: {
                int(key): value.to(device)
                for key, value in row["hidden_checkpoints"].items()
            }
            for row in fp_reference
        }

    outs = torch.zeros_like(inps)
    hidden_checkpoints = {sample_idx: {} for sample_idx in range(args.score_nsamples)}
    hidden_metrics = {sample_idx: {} for sample_idx in range(args.score_nsamples)}
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        for sample_idx in range(args.score_nsamples):
            outs[sample_idx] = layer(
                inps[sample_idx].unsqueeze(0),
                attention_mask=cache["attention_mask"],
                position_embeddings=cache["position_embeddings"],
            )[0]
        if layer_idx in checkpoint_layers:
            for sample_idx in range(args.score_nsamples):
                if fp_reference is None:
                    hidden_checkpoints[sample_idx][layer_idx] = (
                        outs[sample_idx].detach().cpu()
                    )
                else:
                    ref = hidden_refs[sample_idx][layer_idx]
                    hidden_metrics[sample_idx][f"layer{layer_idx}_nmse"] = nmse(
                        outs[sample_idx], ref
                    )
                    hidden_metrics[sample_idx][f"layer{layer_idx}_cosine_drift"] = (
                        cosine_drift(outs[sample_idx], ref)
                    )
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    rows = []
    token_ids = token_ids.to(device)
    for sample_idx in range(args.score_nsamples):
        hidden_states = inps[sample_idx].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        labels = token_ids[:, sample_idx * args.seqlen : (sample_idx + 1) * args.seqlen][
            :, 1:
        ]
        nll = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).detach().cpu()
        row = {
            "sequence": sample_idx,
            "mean_token_nll": float(nll.mean().item()),
            "nonfinite_count": int((~torch.isfinite(logits)).sum().item()),
        }
        if fp_reference is None:
            row["token_nll"] = nll
            row["hidden_checkpoints"] = hidden_checkpoints[sample_idx]
        else:
            delta = nll - fp_reference[sample_idx]["token_nll"]
            k = max(1, math.ceil(delta.numel() * 0.10))
            row["mean_nll_increase"] = float(delta.mean().item())
            row["cvar10_nll_increase"] = float(torch.topk(delta, k).values.mean().item())
            row.update(hidden_metrics[sample_idx])
        rows.append(row)

    model.config.use_cache = use_cache
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    torch.cuda.empty_cache()
    return rows


def strip_reference(rows):
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"token_nll", "hidden_checkpoints"}
        }
        for row in rows
    ]


def summarize(rows):
    keys = [key for key in rows[0] if key != "sequence"]
    out = {}
    for key in keys:
        values = [row[key] for row in rows]
        out[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return out


def compare_to_official(official, variants):
    lower_is_better = [
        "mean_token_nll",
        "mean_nll_increase",
        "cvar10_nll_increase",
        "nonfinite_count",
    ]
    for layer_idx in variants["__window_layers__"]:
        lower_is_better.extend(
            [f"layer{layer_idx}_nmse", f"layer{layer_idx}_cosine_drift"]
        )

    out = {}
    for variant, datasets in variants.items():
        if variant == "__window_layers__":
            continue
        out[variant] = {}
        for dataset, rows in datasets.items():
            out[variant][dataset] = {}
            for metric in lower_is_better:
                hard_values = [row[metric] for row in rows]
                official_values = [row[metric] for row in official[dataset]]
                out[variant][dataset][metric] = {
                    "mean_delta_vs_official": (
                        sum(hard_values) - sum(official_values)
                    )
                    / len(hard_values),
                    "wins": sum(
                        1 for hard, base in zip(hard_values, official_values) if hard < base
                    ),
                    "n": len(hard_values),
                }
    return out


def load_model(args):
    model = pt2_quantize.get_model(args.model, args.seqlen)
    model.eval()
    model.config.use_cache = False
    model.seqlen = args.seqlen
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calib-nsamples", type=int, default=8)
    parser.add_argument("--score-nsamples", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--window-layers", default="0,1")
    args = parser.parse_args()
    args.window_layers = sorted(parse_layers(args.window_layers))

    started = time.time()
    device = "cuda:0"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    calib_loader, _ = get_loaders(
        "wikitext2",
        nsamples=args.calib_nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=args.seqlen,
    )
    eval_sets = {}
    for dataset in ("wikitext2", "c4"):
        _, testenc = get_loaders(
            dataset,
            nsamples=args.calib_nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=args.seqlen,
        )
        eval_sets[dataset] = testenc

    fp_model = load_model(args)
    fp_rows = {
        dataset: score_dataset(fp_model, testenc, args, device)
        for dataset, testenc in eval_sets.items()
    }
    del fp_model
    model_utils.cleanup_memory(verbos=False)

    variant_specs = {
        "official": set(),
        "hard_l0": {0},
        "hard_l1": {1},
        "hard_l0_l1": {0, 1},
    }
    quantization = {}
    scored = {}
    for variant, gated_layers in variant_specs.items():
        model = load_model(args)
        quantization[variant] = quantize_model(
            model, calib_loader, args, variant, gated_layers, device
        )
        scored[variant] = {
            dataset: score_dataset(model, testenc, args, device, fp_rows[dataset])
            for dataset, testenc in eval_sets.items()
        }
        del model
        model_utils.cleanup_memory(verbos=False)

    variants_for_compare = {"__window_layers__": args.window_layers}
    variants_for_compare.update(
        {key: value for key, value in scored.items() if key != "official"}
    )
    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "fp16": {dataset: strip_reference(rows) for dataset, rows in fp_rows.items()},
        "scores": scored,
        "quantization": quantization,
        "summary": {
            variant: {dataset: summarize(rows) for dataset, rows in datasets.items()}
            for variant, datasets in scored.items()
        },
        "delta_vs_official": compare_to_official(scored["official"], variants_for_compare),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "delta_vs_official": result["delta_vs_official"],
            },
            indent=2,
        ),
        flush=True,
    )
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
