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
    enable_tp: bool = field(default=False)
    # TODO(halu): Below could be per query call param instead.
    temperature: float = field(default=0.01)
    top_p: float = field(default=1.0)
    max_tokens: int = field(default=100)


class LLM:
    class Mode(Enum):
        LOCAL_LORA = "local_lora"
        REMOTE_PRETRAIN = "remote_pretrain"

    def __init__(
        self,
        model_id_or_path: str,
        *,
        options: Optional[LLMOptions] = None,
    ) -> None:
        """

        Args:
            model_id_or_path: model_id or local dir to PEFT lora weights.
            options: Options for LLM. Defaults to be None.

        Raises:
            ValueError: When unsupported.
        """
        if not options:
            options = LLMOptions()
        hub_kwargs = {
            "revision": options.revision,
            "token": options.token,
        }
        import transformers

        if os.path.isdir(model_id_or_path):
            if not os.path.isfile(os.path.join(model_id_or_path, _PEFT_CONFIG_NAME)):
                raise ValueError("Peft config is not found.")

            import peft

            peft_config = peft.PeftConfig.from_pretrained(model_id_or_path, **hub_kwargs)  # type: ignore[attr-defined]
            if peft_config.peft_type != peft.PeftType.LORA:  # type: ignore[attr-defined]
                raise ValueError("Only LORA is supported.")
            if peft_config.task_type != peft.TaskType.CAUSAL_LM:  # type: ignore[attr-defined]
                raise ValueError("Only CAUSAL_LM is supported.")
            base_model = peft_config.base_model_name_or_path
            base_config = transformers.AutoConfig.from_pretrained(base_model, **hub_kwargs)
            assert (
                base_config.model_type in SupportedLLMType.valid_values()
            ), f"{base_config.model_type} is not supported."
            self.mode = LLM.Mode.LOCAL_LORA
            self.model_type = base_config.model_type
        else:
            # We support pre-train model as well
            model_config = transformers.AutoConfig.from_pretrained(
                model_id_or_path,
                **hub_kwargs,
            )
            assert (
                model_config.model_type in SupportedLLMType.valid_values()
            ), f"{model_config.model_type} is not supported."
            self.mode = LLM.Mode.REMOTE_PRETRAIN
            self.model_type = model_config.model_type

        self.model_id_or_path = model_id_or_path
        self.token = options.token
        self.revision = options.revision
        self.max_batch_size = options.max_batch_size
        self.temperature = options.temperature
        self.top_p = options.top_p
        self.max_tokens = options.max_tokens
        self.enable_tp = options.enable_tp
