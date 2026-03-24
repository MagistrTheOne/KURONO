"""WF-VAE model registry (inference)."""


class ModelRegistry:
    _models: dict[str, type] = {}

    @classmethod
    def register(cls, model_name: str):
        def decorator(model_class: type) -> type:
            cls._models[model_name] = model_class
            return model_class

        return decorator

    @classmethod
    def get_model(cls, model_name: str):
        return cls._models.get(model_name)
