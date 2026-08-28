#!/usr/bin/env python3
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def median(values):
    return statistics.median(values)


def mean(values):
    return statistics.fmean(values)


def bootstrap_ci(values, fn=median, repeats=10000, seed=0):
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(repeats))
    return draws[int(0.025 * repeats)], draws[int(0.975 * repeats)]


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2 + 1
        for k in range(i, j):
            result[order[k]] = rank
        i = j
    return result


def pearson(x, y):
    xm, ym = mean(x), mean(y)
    num = sum((a - xm) * (b - ym) for a, b in zip(x, y))
    den = math.sqrt(sum((a - xm) ** 2 for a in x) * sum((b - ym) ** 2 for b in y))
    return num / den if den else float("nan")


def fmt(value):
    return f"{value:.6f}"


def main():
    source = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else source.parent
    rows = []
    with source.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in (
                "pair_cosine_mean", "hf_energy_ratio", "ternary_zero_rate",
                "high_zero_rate", "weight_nmse", "activation_weighted_nmse",
                "haar_exact_rel_error",
            ):
                row[key] = None if row[key] in ("", "None") else float(row[key])
            row["block_start"] = int(row["block_start"])
            rows.append(row)

    by_key = {(r["module"], r["block_start"], r["strategy"]): r for r in rows}
    block_keys = sorted({(r["module"], r["block_start"]) for r in rows})
    strategies = ("identity", "adjacent", "random", "dissimilar", "ssr_order", "similarity")
    raw = {}
    for strategy in strategies:
        selected = [r for r in rows if r["strategy"] == strategy]
        raw[strategy] = {
            "n": len(selected),
            "median_hf_energy_ratio": None if strategy == "identity" else median([r["hf_energy_ratio"] for r in selected]),
            "median_high_zero_rate": None if strategy == "identity" else median([r["high_zero_rate"] for r in selected]),
            "median_weight_nmse": median([r["weight_nmse"] for r in selected]),
            "median_activation_weighted_nmse": median([r["activation_weighted_nmse"] for r in selected]),
        }

    comparisons = {}
    for candidate in ("adjacent", "dissimilar", "ssr_order", "similarity"):
        hf_reduction = []
        err_improvement = []
        zero_change = []
        for module, block_start in block_keys:
            c = by_key[(module, block_start, candidate)]
            r = by_key[(module, block_start, "random")]
            hf_reduction.append((r["hf_energy_ratio"] - c["hf_energy_ratio"]) / r["hf_energy_ratio"])
            err_improvement.append((r["activation_weighted_nmse"] - c["activation_weighted_nmse"]) / r["activation_weighted_nmse"])
            zero_change.append(c["high_zero_rate"] - r["high_zero_rate"])
        comparisons[candidate] = {
            "median_relative_hf_reduction": median(hf_reduction),
            "median_relative_hf_reduction_ci95": bootstrap_ci(hf_reduction, seed=11),
            "hf_reduction_positive_fraction": mean([x > 0 for x in hf_reduction]),
            "median_relative_weighted_error_improvement": median(err_improvement),
            "median_relative_weighted_error_improvement_ci95": bootstrap_ci(err_improvement, seed=17),
            "weighted_error_improvement_positive_fraction": mean([x > 0 for x in err_improvement]),
            "median_high_zero_rate_change": median(zero_change),
        }

    projection = {}
    groups = defaultdict(list)
    for module, block_start in block_keys:
        sim = by_key[(module, block_start, "similarity")]
        rnd = by_key[(module, block_start, "random")]
        layer_type = sim["layer_type"]
        groups[layer_type].append({
            "hf_reduction": (rnd["hf_energy_ratio"] - sim["hf_energy_ratio"]) / rnd["hf_energy_ratio"],
            "err_improvement": (rnd["activation_weighted_nmse"] - sim["activation_weighted_nmse"]) / rnd["activation_weighted_nmse"],
            "zero_change": sim["high_zero_rate"] - rnd["high_zero_rate"],
        })
    for layer_type, values in sorted(groups.items()):
        projection[layer_type] = {
            "n": len(values),
            "median_relative_hf_reduction": median([v["hf_reduction"] for v in values]),
            "median_relative_weighted_error_improvement": median([v["err_improvement"] for v in values]),
            "weighted_error_improvement_positive_fraction": mean([v["err_improvement"] > 0 for v in values]),
            "median_high_zero_rate_change": median([v["zero_change"] for v in values]),
        }

    sim_pairs = []
    for module, block_start in block_keys:
        sim = by_key[(module, block_start, "similarity")]
        rnd = by_key[(module, block_start, "random")]
        sim_pairs.append((
            (rnd["hf_energy_ratio"] - sim["hf_energy_ratio"]) / rnd["hf_energy_ratio"],
            (rnd["activation_weighted_nmse"] - sim["activation_weighted_nmse"]) / rnd["activation_weighted_nmse"],
        ))
    hf = [x[0] for x in sim_pairs]
    err = [x[1] for x in sim_pairs]
    correlation = {
        "pearson_hf_reduction_vs_weighted_error_improvement": pearson(hf, err),
        "spearman_hf_reduction_vs_weighted_error_improvement": pearson(ranks(hf), ranks(err)),
    }

    result = {
        "raw_strategy_medians": raw,
        "paired_vs_random": comparisons,
        "similarity_by_projection": projection,
        "correlation": correlation,
        "gate": {
            "pass": comparisons["similarity"]["hf_reduction_positive_fraction"] >= 0.70
            and comparisons["similarity"]["median_relative_weighted_error_improvement"] >= 0.05,
            "hf_positive_fraction_target": 0.70,
            "median_weighted_error_improvement_target": 0.05,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Haar M2 mechanism analysis",
        "",
        "## Raw strategy medians",
        "",
        "| Strategy | n | HF energy ratio | High zero rate | Weight NMSE | Activation-weighted NMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in strategies:
        item = raw[strategy]
        lines.append(
            f"| {strategy} | {item['n']} | "
            f"{'—' if item['median_hf_energy_ratio'] is None else fmt(item['median_hf_energy_ratio'])} | "
            f"{'—' if item['median_high_zero_rate'] is None else fmt(item['median_high_zero_rate'])} | "
            f"{fmt(item['median_weight_nmse'])} | {fmt(item['median_activation_weighted_nmse'])} |"
        )
    lines += ["", "## Similarity versus random by projection", "",
              "| Projection | n | Median HF reduction | Median weighted-error improvement | Win fraction | High-zero change |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, item in projection.items():
        lines.append(
            f"| {name} | {item['n']} | {item['median_relative_hf_reduction']:.2%} | "
            f"{item['median_relative_weighted_error_improvement']:.2%} | "
            f"{item['weighted_error_improvement_positive_fraction']:.2%} | "
            f"{item['median_high_zero_rate_change']:.2%} |"
        )
    sim = comparisons["similarity"]
    lines += [
        "",
        "## Gate",
        "",
        f"- Similarity lowers HF energy in {sim['hf_reduction_positive_fraction']:.2%} of paired blocks.",
        f"- Median weighted-error improvement is {sim['median_relative_weighted_error_improvement']:.2%} "
        f"(95% bootstrap CI {sim['median_relative_weighted_error_improvement_ci95'][0]:.2%} to "
        f"{sim['median_relative_weighted_error_improvement_ci95'][1]:.2%}).",
        f"- Spearman correlation between HF reduction and weighted-error improvement is "
        f"{correlation['spearman_hf_reduction_vs_weighted_error_improvement']:.3f}.",
        f"- Gate pass: **{result['gate']['pass']}**.",
    ]
    (output_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
