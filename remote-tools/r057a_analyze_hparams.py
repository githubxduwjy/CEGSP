#!/usr/bin/env python3
"""Machine-select the preregistered R057A hyperparameter/candidate pair."""

import argparse
import json
import math
from pathlib import Path


CONFIGS = (
    ("H0", 8, 128, 4),
    ("H1", 8, 128, 2),
    ("H2", 8, 128, 8),
    ("H3", 16, 128, 4),
    ("H4", 8, 64, 4),
)
DATASETS = ("wikitext2", "c4")
CANDIDATE_ORDER = ("hard_l11", "hard_l10", "hard_l10_l11")


def metric_mean(payload, variant, dataset, split, metric):
    return payload["summary"][variant][dataset][split][metric]["mean"]


def delta(payload, variant, dataset, split, metric):
    return metric_mean(payload, variant, dataset, split, metric) - metric_mean(
        payload, "official", dataset, split, metric
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    gate_records = []
    payloads = {}
    total_rows = 0
    total_nonfinite = 0

    for config_index, (config_id, nsamples, blocksize, max_steps) in enumerate(CONFIGS):
        path = root / config_id / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads[config_id] = payload
        config = payload["config"]
        assert config["window_layers"] == [10, 11]
        assert config["calib_nsamples"] == nsamples
        assert config["blocksize"] == blocksize
        assert config["max_steps"] == max_steps
        assert config["validation_fraction"] == 0.25
        assert config["seed"] == 0
        assert config["score_start"] == 88
        assert config["gate_nsamples"] == 8
        assert config["test_nsamples"] == 8
        assert config["mean_epsilon"] == 0.0
        assert config["cvar_epsilon"] == 0.0
        assert set(payload["scores"]) == {
            "official", "hard_l10", "hard_l11", "hard_l10_l11"
        }

        for candidate, datasets in payload["scores"].items():
            for dataset, rows in datasets.items():
                assert dataset in DATASETS
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

        for candidate_index, candidate in enumerate(CANDIDATE_ORDER):
            checkpoint_deltas = {
                dataset: {
                    "layer10_nmse": delta(
                        payload, candidate, dataset, "gate", "layer10_nmse"
                    ),
                    "layer11_nmse": delta(
                        payload, candidate, dataset, "gate", "layer11_nmse"
                    ),
                }
                for dataset in DATASETS
            }
            functional_deltas = {
                dataset: {
                    "mean_token_nll": delta(
                        payload, candidate, dataset, "gate", "mean_token_nll"
                    ),
                    "cvar10_nll_increase": delta(
                        payload, candidate, dataset, "gate", "cvar10_nll_increase"
                    ),
                }
                for dataset in DATASETS
            }
            checkpoint_flat = [
                value for metrics in checkpoint_deltas.values() for value in metrics.values()
            ]
            functional_flat = [
                value for metrics in functional_deltas.values() for value in metrics.values()
            ]
            gate_nonfinite = sum(
                metric_mean(payload, candidate, dataset, "gate", "nonfinite_count")
                for dataset in DATASETS
            )
            checkpoint_eligible = gate_nonfinite == 0.0 and max(checkpoint_flat) <= 0.0
            functional_eligible = max(functional_flat) <= 0.0
            gate_records.append(
                {
                    "config_id": config_id,
                    "config_index": config_index,
                    "candidate": candidate,
                    "candidate_index": candidate_index,
                    "hyperparameters": {
                        "calib_nsamples": nsamples,
                        "blocksize": blocksize,
                        "max_steps": max_steps,
                        "validation_fraction": 0.25,
                        "seed": 0,
                    },
                    "checkpoint_deltas": checkpoint_deltas,
                    "functional_deltas": functional_deltas,
                    "worst_checkpoint_delta": max(checkpoint_flat),
                    "worst_functional_delta": max(functional_flat),
                    "checkpoint_eligible": checkpoint_eligible,
                    "functional_eligible": functional_eligible,
                    "eligible": checkpoint_eligible and functional_eligible,
                }
            )

    eligible = [record for record in gate_records if record["eligible"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda record: (
                record["worst_functional_delta"],
                record["config_index"],
                record["candidate_index"],
            ),
        )
        payload = payloads[selected["config_id"]]
        test_deltas = {
            dataset: {
                metric: delta(payload, selected["candidate"], dataset, "test", metric)
                for metric in (
                    "mean_token_nll", "cvar10_nll_increase", "nonfinite_count"
                )
            }
            for dataset in DATASETS
        }
        safe = all(
            value <= 0.0
            for metrics in test_deltas.values()
            for value in metrics.values()
        )
        decision = "support_a" if safe else "fail_a_generalization"
    else:
        selected = {
            "config_id": "H0",
            "candidate": "official",
            "hyperparameters": {
                "calib_nsamples": 8,
                "blocksize": 128,
                "max_steps": 4,
                "validation_fraction": 0.25,
                "seed": 0,
            },
        }
        test_deltas = None
        decision = "inconclusive_overconservative"

    result = {
        "preregistered_configs": [
            {
                "config_id": config_id,
                "calib_nsamples": nsamples,
                "blocksize": blocksize,
                "max_steps": max_steps,
                "validation_fraction": 0.25,
                "seed": 0,
            }
            for config_id, nsamples, blocksize, max_steps in CONFIGS
        ],
        "selection_uses_test": False,
        "integrity": {
            "total_score_rows": total_rows,
            "expected_score_rows": 640,
            "nonfinite_total": total_nonfinite,
        },
        "gate_records": gate_records,
        "selected": selected,
        "selected_untouched_test_deltas": test_deltas,
        "decision": decision,
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "selected": selected,
        "selected_untouched_test_deltas": test_deltas,
        "integrity": result["integrity"],
    }, indent=2))


if __name__ == "__main__":
    main()
