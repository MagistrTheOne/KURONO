import glob
import os
from pathlib import Path
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
        raw = str(pretrained_model_name_or_path).strip()
        p = Path(os.path.expanduser(raw))
        if not p.is_dir():
            raise FileNotFoundError(
                f"WF-VAE pretrained must be a local directory: {raw}. "
                "Copy weights onto the machine (see README: WF-VAE weights)."
            )
        root = str(p.resolve())

        config_file = os.path.join(root, cls.config_name)
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Missing {cls.config_name} under {root}")

        st_hf = os.path.join(root, "diffusion_pytorch_model.safetensors")
        ckpt_files = sorted(glob.glob(os.path.join(root, "*.ckpt")))
        st_other = sorted(glob.glob(os.path.join(root, "*.safetensors")))

        if os.path.isfile(st_hf):
            weights_path = st_hf
        elif ckpt_files:
            weights_path = ckpt_files[-1]
        elif st_other:
            weights_path = st_other[-1]
        else:
            raise FileNotFoundError(
                f"No weights found in {root}. Expected diffusion_pytorch_model.safetensors, "
                "*.safetensors, or *.ckpt."
            )

        model = cls.from_config(config_file)
        model.init_from_ckpt(weights_path)
        return model
