import torch

from haar_mechanism_diagnostics import (
    haar_inverse,
    haar_transform,
    pairing_order,
    shared_atq,
    zero_center_atq,
)


def main():
    torch.manual_seed(7)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight = torch.randn(17, 128, device=device)
    acts = torch.randn(31, 128, device=device)
    for strategy in ("adjacent", "random", "dissimilar", "ssr_order", "similarity"):
        order = pairing_order(strategy, weight, seed=11)
        assert torch.equal(torch.sort(order).values, torch.arange(128, device=device))
        coeff, transformed = haar_transform(weight, acts, order)
        reconstructed = haar_inverse(coeff, order)
        assert torch.allclose(reconstructed, weight, atol=2e-6, rtol=2e-6)
        original_output = weight @ acts.T
        transformed_output = coeff @ transformed.T
        assert torch.allclose(original_output, transformed_output, atol=2e-5, rtol=2e-5)
    covariance = acts.T @ acts / acts.shape[0]
    quantized, ternary, fallback = shared_atq(weight, covariance)
    assert quantized.shape == weight.shape
    assert ternary.shape == weight.shape
    assert torch.isfinite(quantized).all()
    assert set(torch.unique(ternary).tolist()).issubset({-1, 0, 1})
    high_q, high_t, high_fallback = zero_center_atq(weight, covariance)
    assert torch.isfinite(high_q).all()
    assert set(torch.unique(high_t).tolist()).issubset({-1, 0, 1})
    nonzero = high_t != 0
    assert torch.all(torch.sign(high_q[nonzero]) == torch.sign(high_t[nonzero]))
    print("HAAR_TESTS_OK", device, fallback, high_fallback)


if __name__ == "__main__":
    main()
