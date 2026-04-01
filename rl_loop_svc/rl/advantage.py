"""
Generalised Advantage Estimation (GAE).

delta_t   = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t       = delta_t + (gamma * lambda) * A_{t+1}

Reference: Schulman et al., "High-Dimensional Continuous Control Using
           Generalised Advantage Estimation", 2015.
"""

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute GAE advantages for a single episode / batch.

    Args:
        rewards:  Tensor of shape (T,) — scalar rewards r_t.
        values:   Tensor of shape (T,) — value estimates V(s_t).
        gamma:    Discount factor.
        lam:      GAE lambda (bias-variance trade-off).

    Returns:
        advantages: Tensor of shape (T,) — normalised GAE advantages.
    """
    T = rewards.shape[0]
    advantages = torch.zeros(T, dtype=torch.float32)

    gae = 0.0
    # Bootstrap with 0 value after the last step (episode termination)
    next_value = 0.0

    for t in reversed(range(T)):
        delta = rewards[t].item() + gamma * next_value - values[t].item()
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t].item()

    # Normalise to zero mean / unit variance for stability
    if normalize and T > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages
