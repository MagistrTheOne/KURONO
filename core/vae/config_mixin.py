"""Minimal config registration (replaces diffusers ConfigMixin for inference-only VAE)."""

from __future__ import annotations

import functools
import inspect
import json
from pathlib import Path
from typing import Any, Union


class ConfigMixin:
    config_name = "config.json"
    ignore_for_config: list[str] = []

    def register_to_config(self, **kwargs: Any) -> None:
        kwargs.pop("kwargs", None)
        kwargs.pop("_use_default_values", None)
        if not hasattr(self, "_internal_dict"):
            self._internal_dict = dict(kwargs)
        else:
            self._internal_dict = {**dict(self._internal_dict), **kwargs}

    @classmethod
    def from_config(cls, config: Union[str, Path, dict[str, Any]]):
        if isinstance(config, dict):
            d = dict(config)
        else:
            with Path(config).open(encoding="utf-8") as f:
                d = json.load(f)
        for k in list(d.keys()):
            if k.startswith("_"):
                d.pop(k)
        return cls(**d)


def register_to_config(init):
    @functools.wraps(init)
    def inner_init(self, *args, **kwargs):
        init_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}
        config_init_kwargs = {k: v for k, v in kwargs.items() if k.startswith("_")}
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"@register_to_config on {self.__class__.__name__} requires ConfigMixin inheritance."
            )
        ignore = getattr(self, "ignore_for_config", [])
        signature = inspect.signature(init)
        parameters = {
            name: p.default
            for i, (name, p) in enumerate(signature.parameters.items())
            if i > 0 and name not in ignore and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        new_kwargs = {}
        for arg, name in zip(args, parameters.keys()):
            new_kwargs[name] = arg
        new_kwargs.update(
            {
                k: init_kwargs.get(k, default)
                for k, default in parameters.items()
                if k not in ignore and k not in new_kwargs
            }
        )
        if len(set(new_kwargs.keys()) - set(init_kwargs)) > 0:
            new_kwargs["_use_default_values"] = list(set(new_kwargs.keys()) - set(init_kwargs))
        merged = {**config_init_kwargs, **new_kwargs}
        self.register_to_config(**merged)
        init(self, *args, **init_kwargs)

    return inner_init
