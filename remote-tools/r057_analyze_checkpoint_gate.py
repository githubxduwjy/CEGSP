#!/usr/bin/env python3
"""Analyze the preregistered R057 checkpoint-monotonic gate."""

import argparse
import json
import math
from pathlib import Path


WINDOWS = ((10, 11), (30, 31))
DATASETS = ("wikitext2", "c4")


def metric_mean(payload, variant, dataset, split, metric):
    return payload["summary"][variant][dataset][split][metric]["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    window_results = []
    total_rows = 0
    total_nonfinite = 0

    for first, second in WINDOWS:
        path = root / f"layers_{first}_{second}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["config"]
        assert config["window_layers"] == [first, second]
        assert config["score_start"] == 88
        assert config["gate_nsamples"] == 8
        assert config["test_nsamples"] == 8
        assert config["mean_epsilon"] == 0.0
        assert config["cvar_epsilon"] == 0.0

        official = "official"
        candidates = [
            official,
            f"hard_l{first}",
            f"hard_l{second}",
            f"hard_l{first}_l{second}",
        ]
        assert set(payload["scores"]) == set(candidates)
        for candidate in candidates:
            for dataset in DATASETS:
                rows = payload["scores"][candidate][dataset]
                assert len(rows) == 16
                assert [row["sequence"] for row in rows] == list(range(88, 104))
                assert [row["split"] for row in rows] == ["gate"] * 8 + ["test"] * 8
                for row in rows:
                    total_rows += 1
                    total_nonfinite += row["nonfinite_count"]
                    assert all(
                        math.isfinite(value)
                        for value in row.values()
                        if isinstance(value, float)
                    )

        gate_rows = {}
        eligible = []
        for candidate in candidates:
            checkpoint_deltas = {}
            nonfinite = 0.0
            for dataset in DATASETS:
                checkpoint_deltas[dataset] = {
                    f"layer{first}_nmse": metric_mean(
                        payload, candidate, dataset, "gate", f"layer{first}_nmse"
                    )
                    - metric_mean(payload, official, dataset, "gate", f"layer{first}_nmse"),
                    f"layer{second}_nmse": metric_mean(
                        payload, candidate, dataset, "gate", f"layer{second}_nmse"
                    )
                    - metric_mean(payload, official, dataset, "gate", f"layer{second}_nmse"),
                }
                nonfinite += metric_mean(
                    payload, candidate, dataset, "gate", "nonfinite_count"
                )
            flat = [value for ds in checkpoint_deltas.values() for value in ds.values()]
            is_eligible = nonfinite == 0.0 and max(flat) <= 0.0
            gate_rows[candidate] = {
                "checkpoint_deltas": checkpoint_deltas,
                "worst_checkpoint_delta": max(flat),
                "eligible": is_eligible,
            }
            if is_eligible:
                eligible.append(candidate)

        selected = min(
            eligible,
            key=lambda candidate: (gate_rows[candidate]["worst_checkpoint_delta"], candidate),
        )
        test_deltas = {}
        safe = True
        for dataset in DATASETS:
            test_deltas[dataset] = {}
            for metric in ("mean_token_nll", "cvar10_nll_increase", "nonfinite_count"):
                delta = metric_mean(payload, selected, dataset, "test", metric) - metric_mean(
                    payload, official, dataset, "test", metric
                )
                test_deltas[dataset][metric] = delta
            safe = safe and test_deltas[dataset]["mean_token_nll"] <= 0.0
            safe = safe and test_deltas[dataset]["cvar10_nll_increase"] <= 0.0
            safe = safe and test_deltas[dataset]["nonfinite_count"] <= 0.0

        window_results.append(
            {
                "window": [first, second],
                "selected_variant": selected,
                "gate": gate_rows,
                "untouched_test_deltas": test_deltas,
                "safe_on_test": safe,
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_gpu_mib": payload["peak_gpu_mib"],
            }
        )

    nonofficial = sum(row["selected_variant"] != "official" for row in window_results)
    if nonofficial == 0:
        decision = "inconclusive_overconservative"
    elif all(row["safe_on_test"] for row in window_results):
        decision = "support"
    else:
        decision = "fail"
    result = {
        "preregistered_gate": {
            "checkpoint_threshold": 0.0,
            "require_both_checkpoints_and_domains": True,
            "require_at_least_one_nonofficial_selection": True,
            "require_untouched_mean_and_cvar_nonregression": True,
            "require_no_new_nonfinite": True,
        },
        "integrity": {
            "total_score_rows": total_rows,
            "expected_score_rows": 256,
            "nonfinite_total": total_nonfinite,
        },
        "windows": window_results,
        "nonofficial_selections": nonofficial,
        "decision": decision,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
