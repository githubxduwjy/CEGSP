#!/usr/bin/env python3
"""R048: held-out generalization test for a conservative cross-layer gate.

The first half of each dataset's scored windows is used only to choose among
official, hard_l0, hard_l1, and hard_l0_l1.  The second half is untouched until
the choice is frozen and is used to test whether the gate generalizes.
"""

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
    pt2_quantize.GPTQ = (
        gated_integration.GPTQSelectiveValidationGated
        if gated_layers
        else gated_integration.GPTQ
    )


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
            value.float().flatten(), reference.float().flatten(), dim=0
        ).item()
    )


@torch.no_grad()
def score_dataset(
    model,
    testenc,
    args,
    device,
    fp_reference=None,
    return_boundary_hidden=False,
):
    score_nsamples = args.gate_nsamples + args.test_nsamples
    model.seqlen = args.seqlen
    token_start = args.score_start * args.seqlen
    token_end = token_start + score_nsamples * args.seqlen
    token_ids = testenc.input_ids[:, token_start:token_end]
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (score_nsamples, args.seqlen, model.config.hidden_size),
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
    for sample_idx in range(score_nsamples):
        batch = token_ids[:, sample_idx * args.seqlen : (sample_idx + 1) * args.seqlen]
        try:
            model(batch.to(device))
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
    hidden_checkpoints = {idx: {} for idx in range(score_nsamples)}
    hidden_metrics = {idx: {} for idx in range(score_nsamples)}
    boundary_hidden = {}
    boundary_layer = max(checkpoint_layers)
    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(device)
        for sample_idx in range(score_nsamples):
            outs[sample_idx] = layer(
                inps[sample_idx].unsqueeze(0),
                attention_mask=cache["attention_mask"],
                position_embeddings=cache["position_embeddings"],
            )[0]
        if layer_idx in checkpoint_layers:
            for sample_idx in range(score_nsamples):
                if fp_reference is None:
                    hidden_checkpoints[sample_idx][layer_idx] = (
                        outs[sample_idx].detach().cpu()
                    )
                else:
                    source_sequence = args.score_start + sample_idx
                    ref = hidden_refs[source_sequence][layer_idx]
                    hidden_metrics[sample_idx][f"layer{layer_idx}_nmse"] = nmse(
                        outs[sample_idx], ref
                    )
                    hidden_metrics[sample_idx][
                        f"layer{layer_idx}_cosine_drift"
                    ] = cosine_drift(outs[sample_idx], ref)
                if return_boundary_hidden and layer_idx == boundary_layer:
                    boundary_hidden[sample_idx] = outs[sample_idx].detach().cpu()
        layers[layer_idx] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(device)
    model.lm_head = model.lm_head.to(device)

    rows = []
    token_ids = token_ids.to(device)
    for sample_idx in range(score_nsamples):
        hidden_states = inps[sample_idx].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        logits = model.lm_head(hidden_states)
        labels = token_ids[
            :, sample_idx * args.seqlen : (sample_idx + 1) * args.seqlen
        ][:, 1:]
        nll = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).detach().cpu()
        row = {
            "sequence": args.score_start + sample_idx,
            "split": "gate" if sample_idx < args.gate_nsamples else "test",
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
            row["cvar10_nll_increase"] = float(
                torch.topk(delta, k).values.mean().item()
            )
            row.update(hidden_metrics[sample_idx])
        rows.append(row)

    model.config.use_cache = use_cache
    model.model.norm = model.model.norm.cpu()
    model.lm_head = model.lm_head.cpu()
    torch.cuda.empty_cache()
    if return_boundary_hidden:
        return rows, boundary_hidden
    return rows


def cancellation_rows(scored, hidden_outputs, args):
    first_layer, second_layer = args.window_layers
    first_name = f"hard_l{first_layer}"
    second_name = f"hard_l{second_layer}"
    joint_name = f"hard_l{first_layer}_l{second_layer}"
    output = {}
    for dataset in ("wikitext2", "c4"):
        rows = []
        for sample_idx in range(args.gate_nsamples + args.test_nsamples):
            base_h = hidden_outputs["official"][dataset][sample_idx].float()
            first_delta = hidden_outputs[first_name][dataset][sample_idx].float() - base_h
            second_delta = hidden_outputs[second_name][dataset][sample_idx].float() - base_h
            joint_delta = hidden_outputs[joint_name][dataset][sample_idx].float() - base_h
            first_energy = float(first_delta.square().sum().item())
            second_energy = float(second_delta.square().sum().item())
            joint_energy = float(joint_delta.square().sum().item())
            denominator = max(first_energy + second_energy, 1e-30)
            joint_row = scored[joint_name][dataset][sample_idx]
            official_row = scored["official"][dataset][sample_idx]
            rows.append(
                {
                    "sequence": joint_row["sequence"],
                    "split": joint_row["split"],
                    "cancellation_index": (
                        first_energy + second_energy - joint_energy
                    )
                    / denominator,
                    "first_update_energy": first_energy,
                    "second_update_energy": second_energy,
                    "joint_update_energy": joint_energy,
                    "first_checkpoint_nmse_delta": (
                        joint_row[f"layer{first_layer}_nmse"]
                        - official_row[f"layer{first_layer}_nmse"]
                    ),
                    "boundary_checkpoint_nmse_delta": (
                        joint_row[f"layer{second_layer}_nmse"]
                        - official_row[f"layer{second_layer}_nmse"]
                    ),
                    "mean_token_nll_delta": (
                        joint_row["mean_token_nll"]
                        - official_row["mean_token_nll"]
                    ),
                    "cvar10_nll_increase_delta": (
                        joint_row["cvar10_nll_increase"]
                        - official_row["cvar10_nll_increase"]
                    ),
                    "nonfinite_count_delta": (
                        joint_row["nonfinite_count"]
                        - official_row["nonfinite_count"]
                    ),
                }
            )
        output[dataset] = rows
    return output


def strip_reference(rows):
    return [
        {
            key: value
            for key, value in row.items()
            if key not in {"token_nll", "hidden_checkpoints"}
        }
        for row in rows
    ]


def rows_for_split(rows, split):
    return [row for row in rows if row["split"] == split]


def summarize(rows):
    keys = [key for key in rows[0] if key not in {"sequence", "split"}]
    return {
        key: {
            "mean": sum(row[key] for row in rows) / len(rows),
            "min": min(row[key] for row in rows),
            "max": max(row[key] for row in rows),
        }
        for key in keys
    }


def metric_delta(candidate_rows, official_rows, metric):
    return sum(
        candidate[metric] - official[metric]
        for candidate, official in zip(candidate_rows, official_rows)
    ) / len(candidate_rows)


def evaluate_gate(scored, args):
    decision = {}
    for variant in scored:
        checks = {}
        objective_terms = []
        eligible = True
        for dataset in ("wikitext2", "c4"):
            candidate = rows_for_split(scored[variant][dataset], "gate")
            official = rows_for_split(scored["official"][dataset], "gate")
            mean_delta = metric_delta(candidate, official, "mean_token_nll")
            cvar_delta = metric_delta(candidate, official, "cvar10_nll_increase")
            nonfinite_delta = metric_delta(candidate, official, "nonfinite_count")
            checks[dataset] = {
                "mean_token_nll_delta": mean_delta,
                "cvar10_nll_increase_delta": cvar_delta,
                "nonfinite_count_delta": nonfinite_delta,
                "passes": (
                    mean_delta <= args.mean_epsilon
                    and cvar_delta <= args.cvar_epsilon
                    and nonfinite_delta <= 0
                ),
            }
            eligible = eligible and checks[dataset]["passes"]
            objective_terms.extend([mean_delta, cvar_delta])
        decision[variant] = {
            "eligible": eligible,
            "gate_objective": sum(objective_terms) / len(objective_terms),
            "checks": checks,
        }

    eligible = [variant for variant, row in decision.items() if row["eligible"]]
    selected = min(eligible, key=lambda name: (decision[name]["gate_objective"], name))
    return selected, decision


def compare_test(scored, selected):
    output = {}
    for dataset in ("wikitext2", "c4"):
        candidate = rows_for_split(scored[selected][dataset], "test")
        official = rows_for_split(scored["official"][dataset], "test")
        output[dataset] = {
            metric: {
                "mean_delta_vs_official": metric_delta(candidate, official, metric),
                "wins": sum(
                    1
                    for cand, base in zip(candidate, official)
                    if cand[metric] < base[metric]
                ),
                "n": len(candidate),
            }
            for metric in (
                "mean_token_nll",
                "mean_nll_increase",
                "cvar10_nll_increase",
                "nonfinite_count",
            )
        }
    return output


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
    parser.add_argument("--gate-nsamples", type=int, default=4)
    parser.add_argument("--test-nsamples", type=int, default=4)
    parser.add_argument("--score-start", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--window-layers", default="0,1")
    parser.add_argument("--mean-epsilon", type=float, default=0.0)
    parser.add_argument("--cvar-epsilon", type=float, default=0.0)
    parser.add_argument("--compute-cancellation", action="store_true")
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

    if len(args.window_layers) != 2:
        raise ValueError("window-layers must contain exactly two adjacent layers")
    first_layer, second_layer = args.window_layers
    if second_layer != first_layer + 1:
        raise ValueError("window-layers must be adjacent")
    variant_specs = {
        "official": set(),
        f"hard_l{first_layer}": {first_layer},
        f"hard_l{second_layer}": {second_layer},
        f"hard_l{first_layer}_l{second_layer}": {first_layer, second_layer},
    }
    quantization = {}
    scored = {}
    hidden_outputs = {}
    for variant, gated_layers in variant_specs.items():
        model = load_model(args)
        quantization[variant] = quantize_model(
            model, calib_loader, args, variant, gated_layers, device
        )
        if args.compute_cancellation:
            scored[variant] = {}
            hidden_outputs[variant] = {}
            for dataset, testenc in eval_sets.items():
                rows, boundary_hidden = score_dataset(
                    model,
                    testenc,
                    args,
                    device,
                    fp_rows[dataset],
                    return_boundary_hidden=True,
                )
                scored[variant][dataset] = rows
                hidden_outputs[variant][dataset] = boundary_hidden
        else:
            scored[variant] = {
                dataset: score_dataset(model, testenc, args, device, fp_rows[dataset])
                for dataset, testenc in eval_sets.items()
            }
        del model
        model_utils.cleanup_memory(verbos=False)

    selected, gate_decision = evaluate_gate(scored, args)
    test_result = compare_test(scored, selected)
    result = {
        "config": vars(args),
        "elapsed_seconds": time.time() - started,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "fp16": {dataset: strip_reference(rows) for dataset, rows in fp_rows.items()},
        "scores": scored,
        "quantization": quantization,
        "summary": {
            variant: {
                dataset: {
                    split: summarize(rows_for_split(rows, split))
                    for split in ("gate", "test")
                }
                for dataset, rows in datasets.items()
            }
            for variant, datasets in scored.items()
        },
        "gate_decision": gate_decision,
        "selected_variant": selected,
        "untouched_test_result": test_result,
    }
    if args.compute_cancellation:
        result["cancellation"] = cancellation_rows(
            scored, hidden_outputs, args
        )
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_variant": selected,
                "gate_decision": gate_decision,
                "untouched_test_result": test_result,
            },
            indent=2,
        ),
        flush=True,
    )
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
