from dataclasses import dataclass
from typing import Optional

import torch

from .utils.distrib_utils import DiagonalGaussianDistribution


@dataclass
class AutoencoderKLOutput:
    latent_dist: DiagonalGaussianDistribution
    extra_output: Optional[tuple] = None


@dataclass
class DecoderOutput:
    sample: torch.Tensor
    commit_loss: Optional[torch.FloatTensor] = None
    extra_output: Optional[tuple] = None


@dataclass
class ForwardOutput:
    sample: torch.Tensor
    latent_dist: DiagonalGaussianDistribution
    extra_output: Optional[tuple] = None
