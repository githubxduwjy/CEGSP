import torch

from hessian_gated_ternary_diagnostics import (
    activation_grid,
    aggregate_nmse,
    reconstruct,
    refine_ternary,
    weight_itf,
)


def test_gated_refinement_never_worsens_validation_loss():
    torch.manual_seed(7)
    weight = torch.randn(32, 16)
    fit_x = torch.randn(128, 16)
    validation_x = torch.randn(128, 16)
    fit_s = fit_x.T @ fit_x / fit_x.shape[0]
    validation_s = validation_x.T @ validation_x / validation_x.shape[0]

    ternary, alpha, mean = weight_itf(weight)
    alpha, mean, _ = activation_grid(ternary, weight, fit_s, alpha, mean)
    before = reconstruct(ternary, alpha, mean)
    after, refined_t, info = refine_ternary(
        weight, ternary, fit_s, validation_s, max_steps=3
    )

    assert set(refined_t.unique().tolist()).issubset({-1.0, 0.0, 1.0})
    assert aggregate_nmse(weight, after, fit_s) <= aggregate_nmse(weight, before, fit_s) + 1e-7
    assert aggregate_nmse(weight, after, validation_s) <= aggregate_nmse(
        weight, before, validation_s
    ) + 1e-7
    assert 0.0 <= info["changed_fraction"] <= 3 / 16


def test_activation_grid_is_no_worse_than_weight_grid_on_target_covariance():
    torch.manual_seed(11)
    weight = torch.randn(24, 12)
    x = torch.randn(96, 12) * torch.linspace(0.5, 2.0, 12)
    covariance = x.T @ x / x.shape[0]
    ternary, alpha, mean = weight_itf(weight)
    before = reconstruct(ternary, alpha, mean)
    alpha2, mean2, _ = activation_grid(ternary, weight, covariance, alpha, mean)
    after = reconstruct(ternary, alpha2, mean2)
    assert aggregate_nmse(weight, after, covariance) <= aggregate_nmse(
        weight, before, covariance
    ) + 1e-7
