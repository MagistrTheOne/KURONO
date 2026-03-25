import glob
import os
from typing import Optional, Union

import torch
from torch import nn

from .config_mixin import ConfigMixin


class VideoBaseAE(nn.Module, ConfigMixin):
    config_name = "config.json"

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def encode(self, x: torch.Tensor, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.encode() must be implemented.")

    def decode(self, encoding: torch.Tensor, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.decode() must be implemented.")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        **kwargs,
    ):
        root = str(pretrained_model_name_or_path)
        ckpt_files = glob.glob(os.path.join(root, "*.ckpt"))
        if ckpt_files:
            last_ckpt_file = ckpt_files[-1]
            config_file = os.path.join(root, cls.config_name)
            model = cls.from_config(config_file)
            model.init_from_ckpt(last_ckpt_file)
            return model
        raise FileNotFoundError(
            f"No *.ckpt found in {root}. Internal VAE expects a local WF-VAE export (config.json + weights)."
        )
