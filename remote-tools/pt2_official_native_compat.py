#!/usr/bin/env python3
"""Run the official PT2 quantize.py with a narrow OPT API compatibility shim.

The cloud image has Transformers 4.46.3 while the released PT2 entrypoint
passes an OPT-only ``position_embeddings`` keyword that this installed OPT
decoder does not accept.  For OPT the released model path does not provide a
separate position-embedding object, so the shim removes only that unsupported
keyword and delegates every other argument unchanged.
"""

from __future__ import annotations

import inspect
import runpy
import sys

from transformers.models.opt.modeling_opt import OPTDecoderLayer


def patch_opt_decoder() -> bool:
    forward = OPTDecoderLayer.forward
    if "position_embeddings" in inspect.signature(forward).parameters:
        return False

    def wrapped_forward(*args, **kwargs):
        kwargs.pop("position_embeddings", None)
        return forward(*args, **kwargs)

    OPTDecoderLayer.forward = wrapped_forward
    return True


if __name__ == "__main__":
    patched = patch_opt_decoder()
    print(f"PT2_NATIVE_COMPAT opt_position_embeddings_kwarg_dropped={patched}", flush=True)
    runpy.run_path("/root/PT2-LLM-full/quantize.py", run_name="__main__")
