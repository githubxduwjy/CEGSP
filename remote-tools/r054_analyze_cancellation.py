#!/usr/bin/env python3
"""Aggregate the preregistered R054 cancellation audit."""

import argparse
import json
import math
from pathlib import Path


WINDOWS = ((0, 1), (10, 11), (20, 21), (30, 31))


def mean(values):
    return sum(values) / len(values)


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for idx in order[start:end]:
            output[idx] = rank
        start = end
    return output


def pearson(left, right):
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def spearman(left, right):
    return pearson(ranks(left), ranks(right))


def summarize_split(rows):
    risk = [
        max(row["cancellation_index"], 0.0)
        * max(
            row["first_checkpoint_nmse_delta"],
            row["boundary_checkpoint_nmse_delta"],
            0.0,
        )
        for row in rows
    ]
    boundary = [row["boundary_checkpoint_nmse_delta"] for row in rows]
    mean_harm = [row["mean_token_nll_delta"] for row in rows]
    cvar_harm = [row["cvar10_nll_increase_delta"] for row in rows]
    cases = [
        row["cancellation_index"] > 0.0
        and max(
            row["first_checkpoint_nmse_delta"],
            row["boundary_checkpoint_nmse_delta"],
        )
        > 0.0
        for row in rows
    ]
    risk_correlations = {
        "mean_nll": spearman(risk, mean_harm),
        "cvar10": spearman(risk, cvar_harm),
    }
    boundary_correlations = {
        "mean_nll": spearman(boundary, mean_harm),
        "cvar10": spearman(boundary, cvar_harm),
    }
    return {
        "n": len(rows),
        "mechanism_cases": sum(cases),
        "risk_correlations": risk_correlations,
        "boundary_correlations": boundary_correlations,
        "risk_average_rho": mean(list(risk_correlations.values())),
        "boundary_average_rho": mean(list(boundary_correlations.values())),
        "mean_harm_rate_mechanism_cases": (
            mean([float(h > 0.0) for h, case in zip(mean_harm, cases) if case])
            if any(cases)
            else None
        ),
        "mean_harm_rate_other_cases": (
            mean([float(h > 0.0) for h, case in zip(mean_harm, cases) if not case])
            if not all(cases)
            else None
        ),
        "nonfinite_delta_total": sum(row["nonfinite_count_delta"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    combined = []
    integrity = []
    for first, second in WINDOWS:
        path = root / f"layers_{first}_{second}" / "metrics.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["config"]["window_layers"] == [first, second]
        assert payload["config"]["score_start"] == 72
        assert payload["config"]["compute_cancellation"] is True
        assert set(payload["cancellation"]) == {"wikitext2", "c4"}
        for dataset, rows in payload["cancellation"].items():
            assert len(rows) == 16
            assert [row["sequence"] for row in rows] == list(range(72, 88))
            for row in rows:
                assert all(
                    math.isfinite(value)
                    for value in row.values()
                    if isinstance(value, float)
                )
                combined.append(
                    {
                        "window": [first, second],
                        "dataset": dataset,
                        **row,
                    }
                )
        integrity.append(
            {
                "window": [first, second],
                "selected_variant": payload["selected_variant"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "peak_gpu_mib": payload["peak_gpu_mib"],
            }
        )

    split_summary = {
        split: summarize_split([row for row in combined if row["split"] == split])
        for split in ("gate", "test")
    }
    split_pass = {
        split: (
            row["mechanism_cases"] >= 8
            and row["risk_correlations"]["mean_nll"] >= 0.20
            and row["risk_correlations"]["cvar10"] >= 0.20
            and row["risk_average_rho"]
            >= row["boundary_average_rho"] + 0.05
            and row["nonfinite_delta_total"] <= 0
        )
        for split, row in split_summary.items()
    }
    if all(split_pass.values()):
        decision = "support"
    elif any(row["mechanism_cases"] < 8 for row in split_summary.values()):
        decision = "inconclusive_low_prevalence"
    else:
        decision = "fail"
    result = {
        "preregistered_gate": {
            "minimum_mechanism_cases_per_split": 8,
            "minimum_risk_rho_each_metric": 0.20,
            "minimum_average_rho_gain_vs_boundary": 0.05,
            "require_no_new_nonfinite": True,
            "require_both_splits": True,
        },
        "integrity": integrity,
        "rows": combined,
        "split_summary": split_summary,
        "split_pass": split_pass,
        "decision": decision,
    }
    (root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "split_summary": split_summary}, indent=2))


if __name__ == "__main__":
    main()
