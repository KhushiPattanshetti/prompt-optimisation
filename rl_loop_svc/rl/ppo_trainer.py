from dataclasses import dataclass

import torch

from .rollout_buffer import RolloutBatch


@dataclass
class PPOLossComponents:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy_bonus: float
    kl_penalty: float


class PPOTrainer:
    def __init__(
        self,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        beta: float = 0.01,
    ) -> None:
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.beta = beta

    def update(
        self,
        batch: RolloutBatch,
        log_probs_new: torch.Tensor,
        values_new: torch.Tensor,
        entropy: torch.Tensor,
        kl_penalty: torch.Tensor,
    ) -> tuple[torch.Tensor, PPOLossComponents]:
        sample_weights = torch.clamp(batch.sample_weights.to(log_probs_new.device), min=0.0)
        normalizer = torch.clamp(sample_weights.sum(), min=1e-8)

        def _weighted_mean(values: torch.Tensor) -> torch.Tensor:
            return (values * sample_weights).sum() / normalizer

        log_ratio = log_probs_new - batch.log_probs_old
        ratio = torch.exp(log_ratio)

        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch.advantages
        policy_loss = -_weighted_mean(torch.min(surr1, surr2))

        value_pred_clipped = batch.values + (values_new - batch.values).clamp(
            -self.value_clip,
            self.value_clip,
        )
        value_loss_unclipped = (values_new - batch.returns).pow(2)
        value_loss_clipped = (value_pred_clipped - batch.returns).pow(2)
        value_loss = 0.5 * _weighted_mean(torch.max(value_loss_unclipped, value_loss_clipped))

        kl_loss = _weighted_mean(torch.clamp(kl_penalty, min=0.0))
        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
            + self.beta * kl_loss
        )

        components = PPOLossComponents(
            total_loss=float(total_loss.detach().item()),
            policy_loss=float(policy_loss.detach().item()),
            value_loss=float(value_loss.detach().item()),
            entropy_bonus=float(entropy.detach().item()),
            kl_penalty=float(kl_loss.detach().item()),
        )
        return total_loss, components
