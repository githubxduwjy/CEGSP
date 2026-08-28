#!/usr/bin/env python3
import argparse
import json
import logging
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import quantize as pt2_quantize
from pt2_llm.data import get_loaders
from pt2_llm.eval_ppl import llama_eval
from pt2_llm.gptq import GPTQ, error_computing_x_all_accelerate
from pt2_llm.gptq_ssr import GPTQ_SSR, topk_similar_columns
from pt2_llm import model_utils


SELECTIVE_LAYERS = set()
SELECTIVE_PROJECTIONS = set()


def hadamard(n: int, device: torch.device) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard size must be a power of two, got {n}")
    h = torch.ones((1, 1), device=device)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h / math.sqrt(n)


def activation_rms_hadamard_from_cov(s: torch.Tensor) -> torch.Tensor:
    n = s.shape[0]
    score = torch.diag(s.float())
    order = torch.argsort(score, descending=True)
    perm = torch.eye(n, device=s.device, dtype=torch.float32)[order]
    return perm.T @ hadamard(n, s.device)


def parse_csv_set(text: str, cast=str):
    if not text:
        return set()
    return {cast(x.strip()) for x in text.split(",") if x.strip()}


def layer_projection_from_global_name(global_name: str):
    layer_match = re.search(r"\.layers\.(\d+)\.", global_name)
    projection = global_name.rsplit(".", 1)[-1]
    layer_idx = int(layer_match.group(1)) if layer_match else None
    return layer_idx, projection


def use_activation_hadamard_for_layer(layer) -> bool:
    global_name = getattr(layer, "global_name", "")
    layer_idx, projection = layer_projection_from_global_name(global_name)
    if SELECTIVE_LAYERS and layer_idx not in SELECTIVE_LAYERS:
        return False
    if SELECTIVE_PROJECTIONS and projection not in SELECTIVE_PROJECTIONS:
        return False
    return True


class GPTQActivationHadamard(GPTQ):
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        orders=(1, 1, 2),
        num_p=1,
        disable_mask=False,
        no_mask_order=1,
        alpha=0.25,
    ):
        W = self.layer.weight.data.clone()
        W_ori = self.layer.weight.data.clone().float()
        W = W.float()
        tick = time.time()
        T = torch.zeros_like(W)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if self.gptaq:
            self.dXXT[:, dead] = 0

        self.inp = torch.concat(self.inp)
        self.fp_inp1 = torch.concat(self.fp_inp1)

        if self.reorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            self.dXXT = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)
            self.inp = self.inp[:, :, perm]

        Losses = torch.zeros(self.rows, device=self.dev)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        if self.gptaq:
            P = alpha * ((self.dXXT @ Hinv.T).triu_(diagonal=1)) @ Hinv
            del self.dXXT

        for col_st in range(0, self.columns, blocksize):
            col_ed = min(col_st + blocksize, self.columns)
            st = col_st
            ed = col_ed
            inp_block = self.inp[:, :, st:ed].to(torch.float32)
            S = torch.matmul(inp_block.transpose(1, 2), inp_block).mean(dim=0)

            W1 = W[:, col_st:col_ed].clone()
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]
            rot = activation_rms_hadamard_from_cov(S)
            W1_rot = W1 @ rot
            S_rot = rot.T @ S @ rot
            q_rot, ternary = self.braq_quantizer.quantize(W1_rot, S=S_rot, logger=logging.getLogger())
            q_all = q_rot @ rot.T

            diff = W1 - q_all
            d_vec = torch.diag(Hinv1).view(1, -1)
            Q[:, col_st:col_ed] = q_all
            Losses1 = (diff ** 2) / (d_vec ** 2)
            Err1 = diff / d_vec

            W[:, col_st:col_ed] = q_all
            T[:, col_st:col_ed] = ternary
            Losses += torch.sum(Losses1, 1) / 2

            if self.gptaq:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:]) - W1.matmul(P[col_st:col_ed, col_ed:])
            else:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

        if self.reorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        mse_loss = torch.norm(W_ori - Q, p="fro") ** 2 / W_ori.numel()
        torch.cuda.synchronize()
        logging.getLogger().debug(
            f"activation_hadamard_gptq time {time.time() - tick:.2f}, mse {mse_loss.item():.6f}, loss {torch.sum(Losses).item():.6f}"
        )
        if not self.disable_gptq:
            del W1, W1_rot, q_rot, q_all, W, Err1, Losses1, Hinv1, W_ori, mse_loss
        del H, Hinv, self.inp, S
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}


class GPTQSelectiveActivationHadamard(GPTQActivationHadamard):
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        orders=(1, 1, 2),
        num_p=1,
        disable_mask=False,
        no_mask_order=1,
        alpha=0.25,
    ):
        if use_activation_hadamard_for_layer(self.layer):
            return super().fasterquant(
                blocksize=blocksize,
                percdamp=percdamp,
                orders=orders,
                num_p=num_p,
                disable_mask=disable_mask,
                no_mask_order=no_mask_order,
                alpha=alpha,
            )
        return GPTQ.fasterquant(
            self,
            blocksize=blocksize,
            percdamp=percdamp,
            orders=orders,
            num_p=num_p,
            disable_mask=disable_mask,
            no_mask_order=no_mask_order,
            alpha=alpha,
        )


class GPTQSSRActivationHadamard(GPTQ_SSR):
    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        orders=(1, 1, 2),
        num_p=1,
        disable_mask=False,
        no_mask_order=1,
        alpha=0.25,
    ):
        W = self.layer.weight.data.clone()
        W_ori = self.layer.weight.data.clone().float()
        W = W.float()
        tick = time.time()
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if self.gptaq:
            self.dXXT[:, dead] = 0

        self.inp = torch.concat(self.inp)
        Losses = torch.zeros(self.rows, device=self.dev)
        Q = torch.zeros_like(W)
        T = torch.zeros_like(W)
        x_loss_all = 0.0
        perm = torch.arange(self.columns, device=self.dev)
        invperm = torch.arange(self.columns, device=self.dev)

        for blocki, col_st in enumerate(range(0, self.columns, blocksize)):
            col_ed = min(col_st + blocksize, self.columns)
            st = col_st
            ed = col_ed

            if self.reorder:
                remaining = perm[col_st:]
                W_left = W[:, remaining]
                perm_local = topk_similar_columns(W_left, blocksize=blocksize)
                perm[col_st:] = remaining[perm_local]

            W = W[:, perm]
            H_perm = H[perm][:, perm]
            dXXT_perm = self.dXXT[perm][:, perm]
            invperm = torch.argsort(perm)
            inp_perm = self.inp[:, :, perm]

            damp = percdamp * torch.mean(torch.diag(H_perm))
            diag = torch.arange(self.columns, device=self.dev)
            H_perm[diag, diag] += damp
            H_perm = torch.linalg.cholesky(H_perm)
            H_perm = torch.cholesky_inverse(H_perm)
            H_perm = torch.linalg.cholesky(H_perm, upper=True)
            Hinv = H_perm
            P = alpha * ((dXXT_perm @ Hinv.T).triu_(diagonal=1)) @ Hinv

            inp_block = inp_perm[:, :, st:ed].to(torch.float32)
            S = torch.matmul(inp_block.transpose(1, 2), inp_block).mean(dim=0)
            W1 = W[:, col_st:col_ed].clone()
            Hinv1 = Hinv[col_st:col_ed, col_st:col_ed]

            rot = activation_rms_hadamard_from_cov(S)
            W1_rot = W1 @ rot
            S_rot = rot.T @ S @ rot
            q_rot, ternary = self.braq_quantizer.quantize(W1_rot, S=S_rot, logger=logging.getLogger())
            q_all = q_rot @ rot.T

            diff = W1 - q_all
            d_vec = torch.diag(Hinv1).view(1, -1)
            Q[:, col_st:col_ed] = q_all
            x_loss_all += float(error_computing_x_all_accelerate(W1, q_all, S))
            Losses1 = (diff ** 2) / (d_vec ** 2)
            Err1 = diff / d_vec

            W[:, col_st:col_ed] = q_all
            T[:, col_st:col_ed] = ternary
            Losses += torch.sum(Losses1, 1) / 2

            if self.gptaq:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:]) - W1.matmul(P[col_st:col_ed, col_ed:])
            else:
                W[:, col_ed:] -= Err1.matmul(Hinv[col_st:col_ed, col_ed:])

            W = W[:, invperm]

        Q = Q[:, invperm]
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        mse_loss = torch.norm(W_ori - Q, p="fro") ** 2 / W_ori.numel()
        torch.cuda.synchronize()
        logging.getLogger().debug(
            f"activation_hadamard_ssr time {time.time() - tick:.2f}, mse {mse_loss.item():.6f}, "
            f"loss {torch.sum(Losses).item():.6f}, x_loss {x_loss_all:.6f}"
        )
        del W1, W1_rot, q_rot, q_all, W, Err1, Losses1, Hinv1, W_ori, mse_loss
        del H, Hinv, self.inp, S
        torch.cuda.empty_cache()
        return {"error": torch.sum(Losses).item()}


def parse_ppl(model, model_path, datasets, seed, ppl_seqlen, device):
    ppl = {}
    for dataset in datasets:
        _, testloader = get_loaders(dataset, seed=seed, seqlen=ppl_seqlen, model=model_path)
        ppl[dataset] = llama_eval(model, testloader, device, dataset, False, ppl_seqlen)
        print(f"{dataset} ppl {ppl[dataset]:.4f}", flush=True)
    return ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--variant",
        choices=(
            "official_gptq",
            "activation_hadamard_gptq",
            "selective_activation_hadamard_gptq",
            "official_ssr",
            "activation_hadamard_ssr",
        ),
        required=True,
    )
    parser.add_argument("--activation-layers", default="")
    parser.add_argument("--activation-projections", default="")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--ppl-seqlen", type=int, default=2048)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-datasets", default="wikitext2,c4")
    args = parser.parse_args()

    global SELECTIVE_LAYERS, SELECTIVE_PROJECTIONS
    SELECTIVE_LAYERS = parse_csv_set(args.activation_layers, int)
    SELECTIVE_PROJECTIONS = parse_csv_set(args.activation_projections, str)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device = "cuda:0"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

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
        ssr=args.variant.endswith("_ssr"),
        log_wandb=False,
        tasks="",
        experiment=args.variant,
        num_fewshot=0,
        limit=-1,
    )
    pt2_quantize.args = qargs
    pt2_quantize.groupsize = args.blocksize
    if args.variant == "activation_hadamard_gptq":
        pt2_quantize.GPTQ = GPTQActivationHadamard
    elif args.variant == "selective_activation_hadamard_gptq":
        pt2_quantize.GPTQ = GPTQSelectiveActivationHadamard
    elif args.variant == "activation_hadamard_ssr":
        pt2_quantize.GPTQ_SSR = GPTQSSRActivationHadamard

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

    ppl = parse_ppl(
        model,
        args.model,
        [x for x in args.eval_datasets.split(",") if x],
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
    }
    (out_dir / f"{args.variant}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    model_utils.cleanup_memory(verbos=False)


if __name__ == "__main__":
    main()
