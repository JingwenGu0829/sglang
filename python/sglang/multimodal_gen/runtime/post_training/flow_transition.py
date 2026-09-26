# SPDX-License-Identifier: Apache-2.0
"""Request-independent flow transitions. Randomness is supplied by the caller.

The legacy score option intentionally preserves negative squared residuals;
it is not a normalized Gaussian density. Keep its reduction unchanged while
existing trainers migrate to an explicit likelihood contract.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch

FlowMethod = Literal["sde", "cps", "ode"]


@dataclass(frozen=True)
class FlowStepResult:
    sample: torch.Tensor
    mean: torch.Tensor
    noise_std: torch.Tensor
    model_output: torch.Tensor
    score_sum: torch.Tensor
    element_count: torch.Tensor


def flow_step(
    *,
    sample: torch.Tensor,
    model_output: torch.Tensor,
    current_sigma: torch.Tensor,
    next_sigma: torch.Tensor,
    sigma_max: float,
    method: FlowMethod,
    noise_level: float,
    legacy_score: bool,
    variance_noise: torch.Tensor | None,
    score_noise: torch.Tensor | None = None,
    score_shape: tuple[int, ...] | None = None,
) -> FlowStepResult:
    """Return the transition and per-sample score sum/count without side effects.

    SP callers may provide global score_noise alongside local variance_noise;
    the resulting sums/counts are then already global and must not be reduced again.
    """
    if method not in ("sde", "cps", "ode"):
        raise ValueError(f"Unsupported flow method: {method}")
    if sample.shape != model_output.shape or sample.ndim < 2:
        raise ValueError("Flow state and prediction must have matching [B, ...] shapes")
    dt = next_sigma - current_sigma
    if method == "ode":
        # Preserve wrapped-scalar promotion and serving ODE operation order.
        mean = sample + dt * model_output
        next_sample = mean
        noise_std = sample.new_zeros(())
        shape = score_shape or tuple(sample.shape)
        score_sum = torch.zeros(shape[0], device=sample.device, dtype=torch.float32)
        count = score_sum.new_full((shape[0],), float(math.prod(shape[1:])))
        return FlowStepResult(
            next_sample, mean, noise_std, model_output, score_sum, count
        )
    else:
        if variance_noise is None or variance_noise.shape != sample.shape:
            raise ValueError("Stochastic transitions require matching supplied noise")
        if not legacy_score and noise_level <= 0:
            raise ValueError("Gaussian density requires nonzero noise")
        sample, model_output = sample.float(), model_output.float()
        if method == "sde":
            std_dev = (
                torch.sqrt(
                    current_sigma
                    / (
                        1
                        - torch.where(
                            torch.isclose(current_sigma, current_sigma.new_tensor(1.0)),
                            sigma_max,
                            current_sigma,
                        )
                    )
                )
                * noise_level
            )
            noise_std = std_dev * torch.sqrt(-dt)
            mean = (
                sample * (1 + std_dev**2 / (2 * current_sigma) * dt)
                + model_output
                * (1 + std_dev**2 * (1 - current_sigma) / (2 * current_sigma))
                * dt
            )
        else:
            noise_std = next_sigma * math.sin(noise_level * math.pi / 2)
            original = sample - current_sigma * model_output
            estimate = sample + model_output * (1 - current_sigma)
            mean = original * (1 - next_sigma) + estimate * torch.sqrt(
                next_sigma**2 - noise_std**2
            )
        next_sample = mean + variance_noise * noise_std
        density_noise = variance_noise if score_noise is None else score_noise
        residual_score = -((density_noise * noise_std) ** 2)
        if not legacy_score:
            if torch.any(noise_std <= 0):
                raise ValueError("A deterministic transition has no Gaussian density")
            residual_score = (
                residual_score / (2 * noise_std**2)
                - torch.log(noise_std)
                - math.log(math.sqrt(2 * math.pi))
            )
    score_sum = residual_score.sum(dim=tuple(range(1, residual_score.ndim)))
    count = score_sum.new_full(
        (residual_score.shape[0],), float(math.prod(residual_score.shape[1:]))
    )
    return FlowStepResult(next_sample, mean, noise_std, model_output, score_sum, count)
