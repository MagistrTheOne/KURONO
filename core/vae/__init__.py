from .model import ModelRegistry

__all__ = ["ModelRegistry", "WFVAEModel"]


def __getattr__(name: str):
    if name == "WFVAEModel":
        from .modeling_wfvae import WFVAEModel

        return WFVAEModel
    raise AttributeError(name)
