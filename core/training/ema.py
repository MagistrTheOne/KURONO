from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.backup_params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._init_from_model(model)

    def _init_from_model(self, model: nn.Module) -> None:
        self.shadow_params.clear()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow_params:
                self.shadow_params[name] = param.detach().clone()
                continue
            shadow = self.shadow_params[name]
            shadow.mul_(self.decay).add_(param.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module) -> None:
        self.backup_params.clear()
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow_params:
                continue
            self.backup_params[name] = param.detach().clone()
            param.data.copy_(self.shadow_params[name].to(device=param.device, dtype=param.dtype))

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name].to(device=param.device, dtype=param.dtype))
        self.backup_params.clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow_params": {k: v.detach().clone() for k, v in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.decay = float(state_dict["decay"])
        shadow_params = state_dict["shadow_params"]
        if not isinstance(shadow_params, dict):
            raise TypeError("EMA state_dict['shadow_params'] must be a dict")
        self.shadow_params = OrderedDict((k, v.detach().clone()) for k, v in shadow_params.items())
