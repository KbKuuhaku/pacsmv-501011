import numpy as np
import torch

"""
Reference: https://colab.research.google.com/drive/1CUj3dS42BVQ93eztyEeUKLQSFnsQ_rQc#scrollTo=3FwiV_bpSD7Q

- marginal prob std

- diffusion coeff

"""


def compute_marginal_prob_std(t: torch.Tensor, sigma: float) -> torch.Tensor:
    numerator = (sigma ** (2 * t)) - 1
    denominator = 2 * np.log(sigma)
    ret = torch.sqrt(numerator / denominator)

    return ret.reshape(t.shape[0], 1, 1, 1)


def compute_diffusion_coeff(t: torch.Tensor, sigma: float) -> torch.Tensor:
    ret = sigma**t
    return ret.reshape(t.shape[0], 1, 1, 1)
