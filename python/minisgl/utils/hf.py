from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import LlamaConfig


@functools.cache
def _load_config(model_path: str) -> Any:
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> LlamaConfig:
    # deep copy the config to avoid modifying the original config
    config = _load_config(model_path)
    return type(config)(**config.to_dict())


def get_rope_config(config):
    """Get (rope_theta, rope_scaling) from config, supporting both v4 and v5.

    In transformers v5, rope_theta/rope_scaling are accessed via the computed
    property config.rope_parameters. Trust-remote-code configs or parent configs
    passed to sub-models may not have this property or may return None.
    Falls back to the v4-style config.rope_theta / config.rope_scaling attributes.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None:
        return rope_params["rope_theta"], rope_params
    return config.rope_theta, getattr(config, "rope_scaling", None)
