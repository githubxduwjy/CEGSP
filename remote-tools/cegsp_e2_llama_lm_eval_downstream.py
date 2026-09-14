#!/usr/bin/env python3
"""Standard lm-eval downstream runner for Llama-2-7B PT2/TernRefine endpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer, set_seed


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def patch_lm_eval_text_only_imports() -> None:
    try:
        import transformers

        class _UnavailableVisionModel:
            pass

        if not hasattr(transformers, "AutoModelForVision2Seq"):
            transformers.AutoModelForVision2Seq = _UnavailableVisionModel  # type: ignore[attr-defined]
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--model", default="/root/Llama-2-7b-hf")
    p.add_argument("--pt2-checkpoint", required=True)
    p.add_argument("--sidecar-dir", required=True)
    p.add_argument("--patch-json", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pt2-root", default="/root/PT2-LLM-full")
    p.add_argument("--tool-root", default="/root/CEGSP-code/remote-tools")
    p.add_argument("--tasks", default="hellaswag,piqa,arc_easy,arc_challenge,winogrande,mmlu")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--batch-size", default="1")
    p.add_argument("--limit", default="")
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260911)
    p.add_argument("--log-samples", action="store_true")
    return p.parse_args()


def edit_from_row(row: List[Any], score: float = 0.0):
    from cegsp_p7_a100_scaling import AffineEdit

    return AffineEdit(
        layer=int(row[0]),
        key=str(row[1]),
        row=int(row[2]),
        block=int(row[3]),
        donor=int(row[4]),
        receiver=int(row[5]),
        donor_sign=0,
        receiver_sign=int(row[6]),
        score=float(score),
    )


def apply_optional_patch(model: torch.nn.Module, patch_path: str, codes: Dict[int, Dict[str, Any]], perms: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    if not patch_path:
        return {"endpoint": "pt2_baseline", "patch_applied": False}
    from cegsp_e2_qwen_pt2_hba_skipgate_a100 import apply_edit_list
    from cegsp_p7_a100_scaling import audit_all, changed_coordinates
    from cegsp_p9s2_detached_pt2_plugin import apply_ssr_codes, cardinality_violations

    payload = json.loads(Path(patch_path).read_text(encoding="utf-8"))
    edits = [edit_from_row(row) for row in payload["edit_ids"]]
    states = apply_edit_list(codes, edits)
    apply_ssr_codes(model, codes, perms, states)
    audit = audit_all(codes, states)
    return {
        "endpoint": "ternrefine",
        "patch_applied": True,
        "patch_json": patch_path,
        "k_per_layer": payload.get("k_per_layer"),
        "num_relocations": len(edits),
        "changed_coordinates": changed_coordinates(codes, states),
        "cardinality_violations": int(cardinality_violations(codes, states)),
        "audit": audit,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "lm_eval_result.json"
    samples_path = out / "lm_eval_samples.json"
    partial_path = out / "lm_eval_partial.json"
    write_json(partial_path, {"status": "started", "config": vars(args)})

    sys.path.insert(0, args.tool_root)
    sys.path.insert(0, args.pt2_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")

    from cegsp_e2_qwen_pt2_hba_skipgate_a100 import load_full_pt2_state
    from cegsp_p7_a100_scaling import get_decoder_layers
    from cegsp_p9s2_detached_pt2_plugin import load_detached_artifacts, restore_qk
    import quantize as pt2_quantize

    log(f"loading tokenizer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log(f"loading Llama model and PT2 checkpoint {args.pt2_checkpoint}")
    model = pt2_quantize.get_model(args.model, args.max_length)
    model.seqlen = args.max_length
    model.to(device)
    load_info = load_full_pt2_state(model, Path(args.pt2_checkpoint))
    model.config.use_cache = False
    model.eval()

    log(f"loading sidecar {args.sidecar_dir}")
    codes, perms, qk_checkpoint = load_detached_artifacts(Path(args.sidecar_dir))
    restore_qk(model, qk_checkpoint)
    patch_info = apply_optional_patch(model, args.patch_json, codes, perms)

    layers = get_decoder_layers(model)
    write_json(
        partial_path,
        {
            "status": "model_ready",
            "endpoint_name": args.endpoint_name,
            "load_info": load_info,
            "decoder_layers": len(layers),
            "qk_modules": sum(len(v) for v in codes.values()),
            "patch_info": patch_info,
            "elapsed_sec": time.time() - started,
        },
    )

    patch_lm_eval_text_only_imports()
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    task_list = [x.strip() for x in args.tasks.split(",") if x.strip()]
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        add_bos_token=False,
    )
    limit = None if args.limit == "" else float(args.limit)
    log(f"running lm-eval tasks={task_list} fewshot={args.num_fewshot} batch_size={args.batch_size} limit={limit} log_samples={args.log_samples}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=task_list,
            num_fewshot=args.num_fewshot,
            limit=limit,
            log_samples=args.log_samples,
        )
    try:
        print(make_table(results), flush=True)
    except Exception as exc:
        log(f"make_table failed non-fatally: {type(exc).__name__}: {exc}")

    samples = results.get("samples", {})
    if args.log_samples:
        write_json(samples_path, {"endpoint_name": args.endpoint_name, "samples": samples})
    payload = {
        "status": "complete",
        "endpoint_name": args.endpoint_name,
        "config": vars(args),
        "protocol": {
            "evaluator": "lm-evaluation-harness",
            "num_fewshot": args.num_fewshot,
            "tasks": task_list,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "log_samples": bool(args.log_samples),
            "custom_compact_nll": False,
        },
        "state": {
            "pt2_checkpoint": load_info,
            "sidecar_dir": args.sidecar_dir,
            "strict_llama_state_parity": "passed_in_source_pt2_hba_replication",
            "decoder_layers": len(layers),
            "qk_modules": sum(len(v) for v in codes.values()),
            "patch": patch_info,
        },
        "results": results.get("results", {}),
        "versions": results.get("versions", {}),
        "sample_file": str(samples_path) if args.log_samples else "",
        "elapsed_sec": time.time() - started,
    }
    write_json(result_path, payload)
    write_json(partial_path, {"status": "complete", "result": str(result_path), "sample_file": payload["sample_file"], "elapsed_sec": time.time() - started})
    log(f"wrote {result_path}")


if __name__ == "__main__":
    main()
