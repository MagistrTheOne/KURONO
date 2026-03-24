"""Top-2 MoE FFN with softmax gating and load-balancing auxiliary loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoEFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        mlp_ratio: float = 4.0,
        top_k: int = 2,
        capacity_factor: float | None = 1.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        h = int(dim * mlp_ratio)
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(dim, h) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, d = x.shape
        flat = x.reshape(b * n, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(flat.dtype)
        top_w, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)

        if self.capacity_factor is not None and self.capacity_factor > 0:
            cap = max(1, int(self.capacity_factor * (flat.shape[0] * self.top_k) / self.num_experts))
            for e in range(self.num_experts):
                for slot in range(self.top_k):
                    m = top_idx[:, slot] == e
                    if not m.any():
                        continue
                    idxs = m.nonzero(as_tuple=False).squeeze(-1)
                    if idxs.numel() <= cap:
                        continue
                    pw = top_w[idxs, slot]
                    keep = idxs[torch.topk(pw, k=cap, largest=True).indices]
                    drop_mask = torch.ones_like(m, dtype=torch.bool)
                    drop_mask[keep] = False
                    m_drop = m & drop_mask
                    top_w[m_drop] = 0.0
            rs = top_w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            top_w = top_w / rs

        out = torch.zeros_like(flat)
        for e in range(self.num_experts):
            for k in range(self.top_k):
                m = top_idx[:, k] == e
                if not m.any():
                    continue
                w = top_w[m, k].unsqueeze(-1)
                out[m] = out[m] + w * self.experts[e](flat[m])

        load = torch.zeros(self.num_experts, device=flat.device, dtype=probs.dtype)
        for k in range(self.top_k):
            for e in range(self.num_experts):
                m = top_idx[:, k] == e
                if m.any():
                    load[e] = load[e] + top_w[m, k].sum()
        load = load / max(flat.shape[0], 1)
        importance = probs.mean(0)
        aux = (importance * load * float(self.num_experts)).sum()

        return out.view(b, n, d), aux
