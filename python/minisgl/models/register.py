import importlib

from .config import ModelConfig

_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "Qwen3_5ForCausalLM": (".qwen3_5", "Qwen3_5ForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    # Some Qwen3.5 checkpoints may reuse generic architecture names.
    if model_config.model_type == "qwen3_5":
        model_architecture = "Qwen3_5ForCausalLM"

    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
