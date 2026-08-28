#!/usr/bin/env python3
"""P6-B: preregistered seed/offset replication of P6-A.

This wrapper runs the unchanged P6-A protocol three times sequentially.  Only
the seed and the fit/validation/C4 token offsets differ between replicates.
No result from an earlier replicate is used to alter a later configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPLICATES = (
    {"name": "r0_seed20260829_offset0", "seed": 20260829, "offset": 0},
    {"name": "r1_seed20260830_offset512", "seed": 20260830, "offset": 512},
    {"name": "r2_seed20260831_offset1024", "seed": 20260831, "offset": 1024},
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-prefix", required=True)
    p.add_argument("--p6a-script", default="/root/tqgsp-work/cegsp_p6a_score_validity_4090.py")
    p.add_argument("--model", default="facebook/opt-350m")
    p.add_argument("--out-dir", default="/root/tqgsp-runs")
    p.add_argument("--cegsp-root", default="/root/tqgsp-work")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    common = [
        sys.executable,
        args.p6a_script,
        "--model",
        args.model,
        "--out-dir",
        args.out_dir,
        "--cegsp-root",
        args.cegsp_root,
        "--layers",
        ",".join(str(i) for i in range(24)),
        "--seq-len",
        "128",
        "--batch-size",
        "2",
        "--fit-batches",
        "8",
        "--val-batches",
        "8",
        "--untouched-batches",
        "16",
        "--c4-batches",
        "16",
        "--group-size",
        "128",
        "--centered-threshold",
        "0.70",
        "--affine-threshold",
        "0.75",
        "--candidate-pool",
        "32",
        "--evaluated-per-layer",
        "8",
        "--random-per-layer",
        "8",
        "--grad-batches",
        "1",
        "--dtype",
        "bf16",
    ]
    records: List[Dict[str, Any]] = []
    for rep in REPLICATES:
        run_id = f"{args.run_prefix}_{rep['name']}"
        offset = str(rep["offset"])
        cmd = common + [
            "--run-id",
            run_id,
            "--seed",
            str(rep["seed"]),
            "--fit-token-offset",
            offset,
            "--val-token-offset",
            offset,
            "--c4-token-offset",
            offset,
        ]
        print(f"[P6-B] starting {run_id}", flush=True)
        completed = subprocess.run(cmd, check=False)
        result_path = Path(args.out_dir) / run_id / "result.json"
        record: Dict[str, Any] = {
            "replicate": rep,
            "run_id": run_id,
            "returncode": int(completed.returncode),
            "result_path": str(result_path),
        }
        if result_path.exists():
            try:
                result = json.loads(result_path.read_text())
                record["status"] = result.get("status")
                record["elapsed_sec"] = result.get("elapsed_sec")
                record["systems"] = {
                    key: {
                        "score_validity": value.get("score_validity"),
                    }
                    for key, value in result.get("systems", {}).items()
                }
            except Exception as exc:
                record["summary_error"] = f"{type(exc).__name__}: {exc}"
        else:
            record["status"] = "missing_result"
        records.append(record)
        if completed.returncode != 0:
            print(f"[P6-B] replicate failed with returncode={completed.returncode}; stopping", flush=True)
            break

    out_dir = Path(args.out_dir) / args.run_prefix
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_prefix": args.run_prefix,
        "status": "complete" if len(records) == len(REPLICATES) and all(r.get("status") == "complete" for r in records) else "incomplete",
        "experiment": "CEGSP-P6-B preregistered seed/offset replication",
        "fixed_replicates": list(REPLICATES),
        "records": records,
        "elapsed_sec": time.time() - started,
        "protocol_note": "Only seed and fit/validation/C4 token offsets vary; no earlier result changes a later configuration.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"[P6-B] wrote {out_dir / 'summary.json'} status={summary['status']}", flush=True)
    if summary["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
