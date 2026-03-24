"""MoE router scaffolding for KURONO.

S2 will replace this with backend-specific implementation (e.g. MegaBlocks/Tutel).
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RouterOutput:
    topk_idx: torch.Tensor
    topk_scores: torch.Tensor
    aux_loss: torch.Tensor
    z_loss: torch.Tensor


class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, k: int = 2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        logits = self.gate(x)
        scores = torch.softmax(logits, dim=-1)
        topk_scores, topk_idx = torch.topk(scores, k=self.k, dim=-1)
        aux_loss = scores.mean() * 0.0
        z_loss = (logits.float() ** 2).mean()
        return RouterOutput(topk_idx=topk_idx, topk_scores=topk_scores, aux_loss=aux_loss, z_loss=z_loss)

