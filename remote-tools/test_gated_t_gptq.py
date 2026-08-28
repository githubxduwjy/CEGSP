import torch

import gated_t_gptq_quantize as integration


class DummyOriginal:
    groupsize = 8


class DummyLayer:
    global_name = "/root/models/Llama-2-7b-hf.model.layers.0.self_attn.q_proj"


class DummyOwner:
    def __init__(self):
        self.inp = torch.randn(8, 32, 8)
        self.layer = DummyLayer()


def test_proxy_returns_finite_ternary_grid_and_tracks_split():
    torch.manual_seed(17)
    integration.GATED_STATS.clear()
    owner = DummyOwner()
    proxy = integration.ValidationGatedQuantizerProxy(owner, DummyOriginal(), 8)
    weight = torch.randn(16, 8)
    quantized, ternary = proxy.quantize(weight)

    assert quantized.shape == weight.shape
    assert torch.isfinite(quantized).all()
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert len(integration.GATED_STATS) == 1
    assert integration.GATED_STATS[0]["fit_samples"] == 6
    assert integration.GATED_STATS[0]["validation_samples"] == 2
    assert integration.GATED_STATS[0]["block_start"] == 0


def test_stable_itf_handles_degenerate_constant_rows():
    torch.manual_seed(23)
    weight = torch.randn(16, 8)
    weight[0] = 0.125
    weight[1] = 0.0
    ternary, alpha, mean, fallback_rows = integration.stable_weight_itf(weight)

    assert torch.isfinite(alpha).all()
    assert torch.isfinite(mean).all()
    assert set(ternary.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert fallback_rows >= 2
