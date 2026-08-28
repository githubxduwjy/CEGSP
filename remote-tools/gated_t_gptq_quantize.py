#!/usr/bin/env python3
"""Integrate validation-gated discrete ternary refinement into official GPTQ."""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import quantize as pt2_quantize
from hessian_gated_ternary_diagnostics import (
    activation_grid,
    reconstruct,
    refine_ternary,
)
from pt2_llm import model_utils
from pt2_llm.data import get_loaders
from pt2_llm.eval_ppl import llama_eval
from pt2_llm.gptq import GPTQ
from pt2_llm.quantizer import (
    solve_closed_form_alpha_mu,
    ternary_init,
    update_ternary,
)


GATED_LAYERS = set()
GATED_PROJECTIONS = set()
VALIDATION_FRACTION = 0.25
MAX_STEPS = 4
GATED_STATS = []


@torch.no_grad()
def stable_weight_itf(weight, max_iter=100):
    """ITF with finite-value inheritance for degenerate constant rows."""
    mean, alpha, ternary = ternary_init(weight)
    mean = mean.float()
    alpha = alpha.float()
    ternary = ternary.float()
    fallback_rows = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    for _ in range(max_iter):
        previous = ternary.clone()
        candidate_alpha, candidate_mean = solve_closed_form_alpha_mu(
            ternary, weight
        )
        valid = torch.isfinite(candidate_alpha) & torch.isfinite(candidate_mean)
        fallback_rows |= ~valid
        alpha = torch.where(valid, candidate_alpha.float(), alpha)
        mean = torch.where(valid, candidate_mean.float(), mean)
        ternary = update_ternary(weight, alpha, mean).float()
        if torch.equal(ternary, previous):
            break
    if not torch.isfinite(alpha).all() or not torch.isfinite(mean).all():
        raise FloatingPointError("stable ITF produced non-finite grid parameters")
    return ternary, alpha, mean, int(fallback_rows.sum().item())


def parse_csv_set(text: str, cast=str):
    if not text:
        return set()
    return {cast(value.strip()) for value in text.split(",") if value.strip()}


def layer_projection(global_name: str):
    match = re.search(r"\.layers\.(\d+)\.", global_name)
    layer_idx = int(match.group(1)) if match else None
    projection = global_name.rsplit(".", 1)[-1]
    return layer_idx, projection


def use_gated_refinement(layer):
    layer_idx, projection = layer_projection(getattr(layer, "global_name", ""))
    if GATED_LAYERS and layer_idx not in GATED_LAYERS:
        return False
    if GATED_PROJECTIONS and projection not in GATED_PROJECTIONS:
        return False
    return True


class ValidationGatedQuantizerProxy:
    """Drop-in quantizer used by official GPTQ's unchanged error propagation."""

    def __init__(self, owner, original, blocksize):
        self.owner = owner
        self.original = original
        self.groupsize = original.groupsize
        self.blocksize = blocksize
        self.block_index = 0

    @torch.no_grad()
    def quantize(self, weight, **kwargs):
        start = self.block_index * self.blocksize
        end = start + weight.shape[1]
        self.block_index += 1

        if torch.all(weight == 0):
            return torch.zeros_like(weight), torch.zeros_like(weight)

        activation = self.owner.inp[:, :, start:end].float()
        nsamples = activation.shape[0]
        n_validation = max(1, int(round(nsamples * VALIDATION_FRACTION)))
        n_fit = nsamples - n_validation
        if n_fit < 1:
            raise ValueError("Validation split leaves no fit sample")

        fit_x = activation[:n_fit]
        validation_x = activation[n_fit:]
        fit_s = torch.matmul(fit_x.transpose(1, 2), fit_x).mean(dim=0)
        validation_s = torch.matmul(
            validation_x.transpose(1, 2), validation_x
        ).mean(dim=0)
        all_s = (
            fit_s * n_fit + validation_s * n_validation
        ) / nsamples

        ternary, _, _, itf_fallback_rows = stable_weight_itf(weight)
        _, gated_t, info = refine_ternary(
            weight,
            ternary,
            fit_s,
            validation_s,
            MAX_STEPS,
        )
        fallback_alpha, fallback_mean = solve_closed_form_alpha_mu(gated_t, weight)
        alpha, mean, fallback_rows = activation_grid(
            gated_t,
            weight,
            all_s,
            fallback_alpha,
            fallback_mean,
        )
        quantized = reconstruct(gated_t, alpha, mean)
        if not torch.isfinite(quantized).all():
            layer_idx, projection = layer_projection(
                getattr(self.owner.layer, "global_name", "")
            )
            raise FloatingPointError(
                f"non-finite quantized block at layer={layer_idx} "
                f"projection={projection} block_start={start}"
            )

        layer_idx, projection = layer_projection(
            getattr(self.owner.layer, "global_name", "")
        )
        GATED_STATS.append(
            {
                "layer": layer_idx,
                "projection": projection,
                "block_start": start,
                "blocksize": weight.shape[1],
                "fit_samples": n_fit,
                "validation_samples": n_validation,
                "changed_fraction": info["changed_fraction"],
                "acceptance_rate": info["acceptance_rate"],
                "fallback_rows": fallback_rows,
                "itf_fallback_rows": itf_fallback_rows,
            }
        )
        return quantized, gated_t.float()


class GPTQValidationGated(GPTQ):
    def fasterquant(self, blocksize=128, **kwargs):
        original = self.braq_quantizer
        self.braq_quantizer = ValidationGatedQuantizerProxy(
            self, original, blocksize
        )
        try:
            return super().fasterquant(blocksize=blocksize, **kwargs)
        finally:
            self.braq_quantizer = original


class GPTQSelectiveValidationGated(GPTQValidationGated):
    def fasterquant(self, blocksize=128, **kwargs):
        if use_gated_refinement(self.layer):
            return super().fasterquant(blocksize=blocksize, **kwargs)
        return GPTQ.fasterquant(self, blocksize=blocksize, **kwargs)


def evaluate_ppl(model, model_path, datasets, seed, seqlen, device):
    results = {}
    for dataset in datasets:
        _, testloader = get_loaders(
            dataset, seed=seed, seqlen=seqlen, model=model_path
        )
        results[dataset] = llama_eval(
            model, testloader, device, dataset, False, seqlen
        )
        print(f"{dataset} ppl {results[dataset]:.4f}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=("official_gptq", "gated_t_gptq", "selective_gated_t_gptq"),
        required=True,
    )
    parser.add_argument("--gated-layers", default="")
    parser.add_argument("--gated-projections", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-datasets", default="wikitext2,c4")
    args = parser.parse_args()

    global GATED_LAYERS, GATED_PROJECTIONS, VALIDATION_FRACTION, MAX_STEPS
    GATED_LAYERS = parse_csv_set(args.gated_layers, int)
    GATED_PROJECTIONS = parse_csv_set(args.gated_projections)
    VALIDATION_FRACTION = args.validation_fraction
    MAX_STEPS = args.max_steps
    if not 0.0 < VALIDATION_FRACTION < 1.0:
        raise ValueError("validation-fraction must lie strictly between zero and one")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = "cuda:0"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    qargs = SimpleNamespace(
        model=args.model,
        dataset="wikitext2",
        low_quant_method="atq",
        nsamples=args.nsamples,
        percdamp=0.01,
        blocksize=args.blocksize,
        num_p=1,
        salient_metric="hessian",
        device=device,
        disable_gptq=False,
        minlayer=-1,
        maxlayer=1000,
        calib_seqlen=args.calib_seqlen,
        ppl_seqlen=args.ppl_seqlen,
        quant_only="",
        invert=False,
        ssr=False,
        log_wandb=False,
        tasks="",
        experiment=args.variant,
        num_fewshot=0,
        limit=-1,
    )
    pt2_quantize.args = qargs
    pt2_quantize.groupsize = args.blocksize
    if args.variant == "gated_t_gptq":
        pt2_quantize.GPTQ = GPTQValidationGated
    elif args.variant == "selective_gated_t_gptq":
        pt2_quantize.GPTQ = GPTQSelectiveValidationGated

    model = pt2_quantize.get_model(args.model, args.calib_seqlen)
    model.eval()
    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    quant_started = time.time()
    pt2_quantize.quant_sequential(model, dataloader, device)
    quant_seconds = time.time() - quant_started

    ppl = evaluate_ppl(
        model,
        args.model,
        [value for value in args.eval_datasets.split(",") if value],
        args.seed,
        args.ppl_seqlen,
        device,
    )
    result = {
        "config": vars(args),
        "variant": args.variant,
        "quant_seconds": quant_seconds,
        "elapsed_seconds": time.time() - started,
        "ppl": ppl,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "gated_stats": GATED_STATS,
    }
    (out_dir / f"{args.variant}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
