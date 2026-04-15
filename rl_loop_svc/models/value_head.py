import torch
import torch.nn as nn


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int = 3072, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1)
        param_dtype = next(self.layers.parameters()).dtype
        if pooled.dtype != param_dtype:
            pooled = pooled.to(param_dtype)
        return self.layers(pooled).squeeze(-1)
