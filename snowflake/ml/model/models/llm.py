import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set

_PEFT_CONFIG_NAME = "adapter_config.json"


class SupportedLLMType(Enum):
    LLAMA_MODEL_TYPE = "llama"
    OPT_MODEL_TYPE = "opt"

    @classmethod
    def valid_values(cls) -> Set[str]:
        return {member.value for member in cls}


@dataclass(frozen=True)
class LLMOptions:
    """
    This is the option class for LLM.

    Args:
        revision: Revision of HF model. Defaults to None.
        token: The token to use as HTTP bearer authorization for remote files. Defaults to None.
        max_batch_size: Max batch size allowed for single inferenced. Defaults to 1.
    """

    revision: Optional[str] = field(default=None)
    token: Optional[str] = field(default=None)
    max_batch_size: int = field(default=1)


class LLM:
    def __init__(
        self,
        model_id_or_path: str,
        *,
        options: Optional[LLMOptions] = None,
    ) -> None:
        """

        Args:
            model_id_or_path: Local dir to PEFT weights.
            options: Options for LLM. Defaults to be None.

        Raises:
            ValueError: When unsupported.
        """
        if not (os.path.isdir(model_id_or_path) and os.path.isfile(os.path.join(model_id_or_path, _PEFT_CONFIG_NAME))):
            raise ValueError("Peft config is not found.")
        import peft
        import transformers

        if not options:
            options = LLMOptions()

        hub_kwargs = {
            "revision": options.revision,
            "token": options.token,
        }
        peft_config = peft.PeftConfig.from_pretrained(model_id_or_path, **hub_kwargs)  # type: ignore[attr-defined]
        if peft_config.peft_type != peft.PeftType.LORA:  # type: ignore[attr-defined]
            raise ValueError("Only LORA is supported.")
        if peft_config.task_type != peft.TaskType.CAUSAL_LM:  # type: ignore[attr-defined]
            raise ValueError("Only CAUSAL_LM is supported.")
        base_model = peft_config.base_model_name_or_path
        base_config = transformers.AutoConfig.from_pretrained(base_model, **hub_kwargs)
        assert base_config.model_type in SupportedLLMType.valid_values(), f"{base_config.model_type} is not supported."

        self.model_id_or_path = model_id_or_path
        self.token = options.token
        self.revision = options.revision
        self.max_batch_size = options.max_batch_size
        self.model_type = base_config.model_type
