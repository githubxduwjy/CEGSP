#!/usr/bin/env python3
"""R046: score layer-0 hard-T inside the matched fully quantized context."""

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


def configure_quantization(args, variant, device):
    pt2_quantize.args = make_qargs(args, variant, device)
    pt2_quantize.groupsize = args.blocksize
    gated_integration.GATED_STATS.clear()
    gated_integration.GATED_LAYERS = {0}
    gated_integration.GATED_PROJECTIONS = set()
    gated_integration.VALIDATION_FRACTION = args.validation_fraction
    gated_integration.MAX_STEPS = args.max_steps
    if variant == "hard_t_layer0_full_context":
        pt2_quantize.GPTQ = gated_integration.GPTQSelectiveValidationGated


@torch.no_grad()
def quantize_model(model, dataloader, args, variant, device):
    configure_quantization(args, variant, device)
    started = time.time()
    pt2_quantize.quant_sequential(model, dataloader, device)
    return {
        "quant_seconds": time.time() - started,
        "gated_stats": list(gated_integration.GATED_STATS),
    }


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

    outs = torch.zeros_like(inps)
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        for sample_idx in range(args.score_nsamples):
            outs[sample_idx] = layer(
                inps[sample_idx].unsqueeze(0),
                attention_mask=cache["attention_mask"],
                position_embeddings=cache["position_embeddings"],
            )[0]
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
        else:
            delta = nll - fp_reference[sample_idx]["token_nll"]
            k = max(1, math.ceil(delta.numel() * 0.10))
            row["mean_nll_increase"] = float(delta.mean().item())
            row["cvar10_nll_increase"] = float(torch.topk(delta, k).values.mean().item())
        rows.append(row)

    model.config.use_cache = use_cache
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    torch.cuda.empty_cache()
    return rows


def strip_reference(rows):
    return [{key: value for key, value in row.items() if key != "token_nll"} for row in rows]


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


def build_gate(official, hard):
    metrics = ("mean_token_nll", "mean_nll_increase", "cvar10_nll_increase", "nonfinite_count")
    out = {}
    for dataset in official:
        out[dataset] = {}
        for metric in metrics:
            per_sequence = []
            for fixed_row, hard_row in zip(official[dataset], hard[dataset]):
                per_sequence.append(
                    {
                        "sequence": fixed_row["sequence"],
                        "official": fixed_row[metric],
                        "hard": hard_row[metric],
                        "rejects_hard": fixed_row[metric] < hard_row[metric],
                    }
                )
            out[dataset][metric] = {
                "rejects_hard_all_sequences": all(
                    item["rejects_hard"] for item in per_sequence
                ),
                "per_sequence": per_sequence,
            }
    successful = [
        f"{dataset}:{metric}"
        for dataset, dataset_metrics in out.items()
        for metric, value in dataset_metrics.items()
        if value["rejects_hard_all_sequences"]
    ]
    return {"pass": bool(successful), "successful_metrics": successful, "metrics": out}


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
    args = parser.parse_args()

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

    official_model = load_model(args)
    official_quant = quantize_model(
        official_model, calib_loader, args, "official_full_context", device
    )
    official_rows = {
        dataset: score_dataset(official_model, testenc, args, device, fp_rows[dataset])
        for dataset, testenc in eval_sets.items()
    }
    del official_model
    model_utils.cleanup_memory(verbos=False)

    hard_model = load_model(args)
    hard_quant = quantize_model(
        hard_model, calib_loader, args, "hard_t_layer0_full_context", device
    )
    hard_rows = {
        dataset: score_dataset(hard_model, testenc, args, device, fp_rows[dataset])
        for dataset, testenc in eval_sets.items()
    }

    gate = build_gate(official_rows, hard_rows)
    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "fp16": {dataset: strip_reference(rows) for dataset, rows in fp_rows.items()},
        "official_full_context": official_rows,
        "hard_t_layer0_full_context": hard_rows,
        "quantization": {
            "official_full_context": official_quant,
            "hard_t_layer0_full_context": hard_quant,
        },
        "summary": {
            "official_full_context": {
                dataset: summarize(rows) for dataset, rows in official_rows.items()
            },
            "hard_t_layer0_full_context": {
                dataset: summarize(rows) for dataset, rows in hard_rows.items()
            },
        },
        "gate": gate,
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "gate": gate}, indent=2), flush=True)
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
