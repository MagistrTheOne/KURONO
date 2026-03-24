import importlib

MODULES_BASE = "core.vae.modules."


def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
