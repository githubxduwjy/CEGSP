#!/usr/bin/env python3
"""Audit TernRefine result or patch JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def get_nested(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def is_finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, int) or isinstance(value, str) or value is None or isinstance(value, bool):
        return True
    if isinstance(value, list):
        return all(is_finite_tree(v) for v in value)
    if isinstance(value, dict):
        return all(is_finite_tree(v) for v in value.values())
    return True


def collect_checks(data: dict[str, Any]) -> list[tuple[str, bool | None, str]]:
    checks: list[tuple[str, bool | None, str]] = []
    status = data.get("status")
    checks.append(("status_complete", None if status is None else status == "complete", f"status={status!r}"))
    checks.append(("finite_json_values", is_finite_tree(data), "all numeric values are finite"))

    for name in (
        "state_parity_pass",
        "same_frozen_pt2_state",
        "all_selected_patches_legal",
        "all_selected_patches_finite",
        "all_exact_relocations",
        "all_exact_changed_coordinates",
        "one_backward_each",
        "no_rerank",
    ):
        value = get_nested(data, ("gate", name))
        checks.append((name, None if value is None else bool(value), f"value={value!r}"))

    for key in ("cardinality_violations", "selected_cardinality_violations"):
        value = data.get(key)
        if value is not None:
            checks.append((key, int(value) == 0, f"value={value!r}"))

    reloc = data.get("num_relocations", data.get("selected_relocations"))
    changed = data.get("changed_coordinates", data.get("selected_changed_coordinates"))
    if reloc is not None and changed is not None:
        checks.append(("changed_coordinates_equal_2x_relocations", int(changed) == 2 * int(reloc), f"{changed} vs 2*{reloc}"))

    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to a result, summary, or selected_patch JSON file.")
    args = parser.parse_args()

    path = Path(args.result)
    data = json.loads(path.read_text(encoding="utf-8"))
    checks = collect_checks(data)
    failed = False
    for name, ok, detail in checks:
        if ok is None:
            print(f"[SKIP] {name}: field not present")
        elif ok:
            print(f"[PASS] {name}: {detail}")
        else:
            print(f"[FAIL] {name}: {detail}")
            failed = True
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

