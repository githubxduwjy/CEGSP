#!/usr/bin/env python3
"""Machine-check R058: fixed H0/hard_l11 checkpoint-veto replication."""

import argparse
import json
import math
from pathlib import Path


DATASETS = ("wikitext2", "c4")
FUNCTIONAL = ("mean_token_nll", "cvar10_nll_increase")
EXPECTED_CANDIDATES = ("official", "hard_l10", "hard_l11", "hard_l10_l11")


def mean_delta(scores, candidate, dataset, split, metric):
    cand = [row for row in scores[candidate][dataset] if row["split"] == split]
    base = [row for row in scores["official"][dataset] if row["split"] == split]
    if len(cand) != 8 or len(base) != 8:
        raise ValueError(f"expected 8 rows for {candidate}/{dataset}/{split}")
    if [row["sequence"] for row in cand] != [row["sequence"] for row in base]:
        raise ValueError(f"sequence mismatch for {candidate}/{dataset}/{split}")
    return sum(float(x[metric]) - float(y[metric]) for x, y in zip(cand, base)) / 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.metrics.read_text())
    config = data["config"]
    expected = {
        "calib_nsamples": 8,
        "gate_nsamples": 8,
        "test_nsamples": 8,
        "score_start": 120,
        "seqlen": 2048,
        "blocksize": 128,
        "seed": 1,
        "validation_fraction": 0.25,
        "max_steps": 4,
        "window_layers": [10, 11],
        "mean_epsilon": 0.0,
        "cvar_epsilon": 0.0,
    }
    config_ok = all(config.get(key) == value for key, value in expected.items())
    scores = data["scores"]
    candidates_ok = tuple(sorted(scores)) == tuple(sorted(EXPECTED_CANDIDATES))
    rows = [row for cand in scores.values() for ds in cand.values() for row in ds]
    expected_sequences = set(range(120, 136))
    sequences_ok = all(
        {row["sequence"] for row in scores[candidate][dataset]} == expected_sequences
        for candidate in EXPECTED_CANDIDATES
        for dataset in DATASETS
    )
    split_ok = all(
        sum(row["split"] == "gate" for row in scores[candidate][dataset]) == 8
        and sum(row["split"] == "test" for row in scores[candidate][dataset]) == 8
        for candidate in EXPECTED_CANDIDATES
        for dataset in DATASETS
    )
    finite_ok = all(
        math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in {"sequence", "split"}
    )
    nonfinite_total = sum(int(row["nonfinite_count"]) for row in rows)

    functional = {
        split: {
            dataset: {
                metric: mean_delta(scores, "hard_l11", dataset, split, metric)
                for metric in FUNCTIONAL
            }
            for dataset in DATASETS
        }
        for split in ("gate", "test")
    }
    checkpoint = {
        dataset: {
            metric: mean_delta(scores, "hard_l11", dataset, "gate", metric)
            for metric in ("layer10_nmse", "layer11_nmse")
        }
        for dataset in DATASETS
    }
    gate_pass = all(value <= 0 for ds in functional["gate"].values() for value in ds.values())
    test_pass = all(value <= 0 for ds in functional["test"].values() for value in ds.values())
    checkpoint_veto = any(value > 0 for ds in checkpoint.values() for value in ds.values())
    integrity_ok = (
        config_ok and candidates_ok and len(rows) == 128 and sequences_ok
        and split_ok and finite_ok and nonfinite_total == 0
    )

    if not integrity_ok:
        decision = "invalid"
    elif not gate_pass:
        decision = "reject_candidate"
    elif not test_pass:
        decision = "fail_functional_generalization"
    elif checkpoint_veto:
        decision = "support_veto_overconservative"
    else:
        decision = "support_hard_l11_no_veto"

    result = {
        "decision": decision,
        "fixed_candidate": "hard_l11",
        "config": config,
        "functional_deltas": functional,
        "checkpoint_gate_deltas": checkpoint,
        "gate_pass": gate_pass,
        "test_pass": test_pass,
        "checkpoint_veto": checkpoint_veto,
        "integrity": {
            "config_ok": config_ok,
            "candidates_ok": candidates_ok,
            "total_score_rows": len(rows),
            "expected_score_rows": 128,
            "sequences_ok": sequences_ok,
            "split_ok": split_ok,
            "finite_ok": finite_ok,
            "nonfinite_total": nonfinite_total,
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
