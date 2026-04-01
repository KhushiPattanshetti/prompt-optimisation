"""
Value head: MLP mapping pooled hidden state → scalar V(s).

FIX SUMMARY (from review):
  - Default hidden_size was 768 (GPT-2).
    Phi-3-mini hidden size is 3072.
    ValueHead(768→1) would throw a shape mismatch when fed
    Phi-3 hidden states of shape (B, T, 3072).
  - Default changed to 3072. Settings.hidden_size also fixed to 3072.
"""

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """
    Two-layer MLP value head.

    Args:
        hidden_size: Dimensionality of the input hidden state.
                     Must match the policy model hidden size.
                     Phi-3-mini = 3072.
        dropout:     Dropout probability applied between layers.
    """

    def __init__(self, hidden_size: int = 3072, dropout: float = 0.1) -> None:
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
        pooled = hidden_states.mean(dim=1)      # (B, H)
        return self.net(pooled).squeeze(-1)     # (B,)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        self.load_state_dict(torch.load(path, map_location=device))

# """
# Value head: a small MLP that maps the pooled hidden state from the
# policy model to a scalar state-value estimate V(s).
# """

# import torch
# import torch.nn as nn


# class ValueHead(nn.Module):
#     """
#     Two-layer MLP value head.

#     Args:
#         hidden_size: Dimensionality of the input hidden state.
#         dropout: Dropout probability applied between layers.
#     """

#     def __init__(self, hidden_size: int = 768, dropout: float = 0.1) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.Tanh(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_size // 2, 1),
#         )

#     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             hidden_states: Tensor of shape (B, T, H).

#         Returns:
#             values: Tensor of shape (B,) — mean-pooled value estimate.
#         """
#         pooled = hidden_states.mean(dim=1)  # (B, H)
#         return self.net(pooled).squeeze(-1)  # (B,)

#     def save(self, path: str) -> None:
#         torch.save(self.state_dict(), path)

#     def load(self, path: str, device: str = "cpu") -> None:
#         self.load_state_dict(torch.load(path, map_location=device))
