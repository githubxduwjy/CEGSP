#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from haar_mechanism_diagnostics import shared_atq
from pt2_llm.data import get_loaders
from pt2_llm.eval_ppl import llama_eval


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones((1, 1), device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(n)


def parse_layers(text: str):
    if text == "all":
        return None
    return [int(x) for x in text.split(",") if x]


@torch.no_grad()
def quantize_block_identity(weight: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    covariance = acts.float().T @ acts.float()
    covariance /= max(1, acts.shape[0])
    quantized, _, _ = shared_atq(weight.float(), covariance)
    return quantized


@torch.no_grad()
def quantize_block_activation_hadamard(weight: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
    n = weight.shape[1]
    score = acts.float().square().mean(dim=0)
    order = torch.argsort(score, descending=True)
    perm = torch.eye(n, device=weight.device, dtype=torch.float32)[order]
    rot = perm.T @ hadamard(n, weight.device)
    coeff = weight.float() @ rot
    transformed_acts = acts.float() @ rot
    covariance = transformed_acts.T @ transformed_acts
    covariance /= max(1, transformed_acts.shape[0])
    quantized, _, _ = shared_atq(coeff, covariance)
    return quantized @ rot.T


@torch.no_grad()
def quantize_linear(module, acts_cpu: torch.Tensor, variant: str, blocksize: int, device: str):
    weight_cpu = module.weight.detach().cpu()
    quantized_cpu = torch.empty_like(weight_cpu, dtype=torch.float32)
    in_features = weight_cpu.shape[1]
    if in_features % blocksize:
        raise ValueError(f"{module} input features {in_features} not divisible by blocksize {blocksize}")
    for start in range(0, in_features, blocksize):
        end = start + blocksize
        weight = weight_cpu[:, start:end].to(device, dtype=torch.float32)
        acts = acts_cpu[:, start:end].to(device, dtype=torch.float32)
        if variant == "identity":
            q = quantize_block_identity(weight, acts)
        elif variant == "activation_rms_hadamard":
            q = quantize_block_activation_hadamard(weight, acts)
        else:
            raise ValueError(variant)
        quantized_cpu[:, start:end] = q.cpu()
        del weight, acts, q
        torch.cuda.empty_cache()
    module.weight.data.copy_(quantized_cpu.to(dtype=module.weight.dtype))


def selected_modules(model, layers):
    projection_names = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
    )
    result = {}
    layer_ids = range(len(model.model.layers)) if layers is None else layers
    for layer_idx in layer_ids:
        named = dict(model.model.layers[layer_idx].named_modules())
        for projection in projection_names:
            result[f"layer_{layer_idx}.{projection}"] = named[projection]
    return result


@torch.no_grad()
def capture_activations(model, targets, model_path: str, nsamples: int, tokens_per_sample: int, device: str):
    activations = {name: [] for name in targets}
    handles = []
    for name, module in targets.items():
        def hook(_, inputs, __, key=name):
            flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(tokens_per_sample, flat.shape[0])
            index = torch.linspace(0, flat.shape[0] - 1, count, device=flat.device).long()
            activations[key].append(flat[index].float().cpu())
        handles.append(module.register_forward_hook(hook))

    dataloader, _ = get_loaders(
        "wikitext2",
        nsamples=nsamples,
        seed=0,
        model=model_path,
        seqlen=2048,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    with torch.inference_mode():
        for sample_idx, batch in enumerate(dataloader):
            model(batch[0].to(device), use_cache=False)
            print(f"Captured calibration sample {sample_idx + 1}/{nsamples}", flush=True)
    for handle in handles:
        handle.remove()
    model.cpu()
    torch.cuda.empty_cache()
    return {name: torch.cat(values, dim=0) for name, values in activations.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", choices=("identity", "activation_rms_hadamard"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="0,10,20,31")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--tokens-per-sample", type=int, default=128)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--eval-datasets", default="wikitext2,c4")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    layers = parse_layers(args.layers)
    device = "cuda:0"

    print(f"Loading model {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.seqlen = 2048
    model.eval()
    model.config.use_cache = False

    targets = selected_modules(model, layers)
    print(f"Capturing activations for {len(targets)} modules", flush=True)
    activations = capture_activations(model, targets, args.model, args.nsamples, args.tokens_per_sample, device)

    print(f"Applying fake quantization variant={args.variant}", flush=True)
    quant_started = time.time()
    for idx, name in enumerate(sorted(targets), start=1):
        quantize_linear(targets[name], activations[name], args.variant, args.blocksize, device)
        if idx % 7 == 0 or idx == len(targets):
            print(f"Quantized {idx}/{len(targets)} modules", flush=True)
    quant_seconds = time.time() - quant_started
    del activations, targets
    torch.cuda.empty_cache()

    ppl = {}
    eval_started = time.time()
    for dataset in [x for x in args.eval_datasets.split(",") if x]:
        _, testloader = get_loaders(dataset, seed=0, seqlen=args.ppl_seqlen, model=args.model)
        ppl[dataset] = llama_eval(model, testloader, device, dataset, False, args.ppl_seqlen)
        print(f"{dataset} ppl {ppl[dataset]:.4f}", flush=True)
    eval_seconds = time.time() - eval_started

    result = {
        "config": vars(args),
        "variant": args.variant,
        "num_quantized_modules": 0 if layers == [] else len(selected_modules(model, layers)),
        "quant_seconds": quant_seconds,
        "eval_seconds": eval_seconds,
        "elapsed_seconds": time.time() - started,
        "ppl": ppl,
        "peak_gpu_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }
    (out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
