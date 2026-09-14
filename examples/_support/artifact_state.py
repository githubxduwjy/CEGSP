"""Small checkpoint utilities used by the anonymous Llama PT2 example.

This module intentionally contains no experiment policy.  It only restores a
previously exported PT2 deployment state into an already constructed model so
the TernRefine example can reuse a frozen PT2 initializer without re-running
the expensive PT2 quantization pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from _support.large_model_affine import get_decoder_layers


def _module_by_name(layer: torch.nn.Module, name: str) -> torch.nn.Module:
    current = layer
    for part in name.split("."):
        current = getattr(current, part)
    return current


def load_full_pt2_state(model: torch.nn.Module, checkpoint: Path) -> Dict[str, Any]:
    """Load a CEGSP-exported frozen PT2 state into ``model``.

    Supported formats:
    - ``CEGSP_PT2_FULL_STATE_V1``: all modules stored as ternary sidecar state.
    - ``CEGSP_PT2_FULL_STATE_V2_MIXED``: ternary modules plus optional raw
      deployed-weight entries for modules that are not edited by TernRefine.
    """

    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("format") not in {"CEGSP_PT2_FULL_STATE_V1", "CEGSP_PT2_FULL_STATE_V2_MIXED"}:
        raise RuntimeError(f"unsupported PT2 checkpoint format: {payload.get('format')}")

    layers = get_decoder_layers(model)
    for row in payload["modules"].values():
        layer = layers[int(row["layer"])]
        module = _module_by_name(layer, str(row["name"]))
        if row.get("encoding") == "raw_weight":
            weight = row["weight"]
            module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
            continue

        ternary = row["T"].to(torch.float32)
        q_perm = (row["mu"].float() + row["alpha"].float() * ternary).view(
            int(row["original_shape"][0]), -1
        )
        q_perm = q_perm[:, : int(row["original_shape"][1])]
        inverse = torch.argsort(row["perm"].long())
        q_original = q_perm[:, inverse]
        module.weight.data.copy_(q_original.to(device=module.weight.device, dtype=module.weight.dtype))

    return {
        "format": payload["format"],
        "module_count": int(payload["module_count"]),
        "ternary_module_count": int(payload.get("ternary_module_count", payload.get("module_count", 0))),
        "raw_module_count": int(payload.get("raw_module_count", 0)),
        "checkpoint": str(checkpoint),
    }
