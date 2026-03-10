"""
Value head: a small MLP that maps the pooled hidden state from the
policy model to a scalar state-value estimate V(s).
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Two-layer MLP value head.

    Args:
        hidden_size: Dimensionality of the input hidden state.
        dropout: Dropout probability applied between layers.
    """

    def __init__(self, hidden_size: int = 768, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Tensor of shape (B, T, H).

        Returns:
            values: Tensor of shape (B,) — mean-pooled value estimate.
        """
        pooled = hidden_states.mean(dim=1)  # (B, H)
        return self.net(pooled).squeeze(-1)  # (B,)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=device))
