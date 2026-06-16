import enum
from typing import Union


class InferenceEngine(enum.Enum):
    VLLM = "vllm"
    PYTHON_GENERIC = "python_generic"

    @classmethod
    def from_value(cls, value: Union["InferenceEngine", str]) -> "InferenceEngine":
        """Parse an inference engine from an enum member or case-insensitive string.

        Args:
            value: An InferenceEngine enum member or a string matching a member name or value.

        Returns:
            The matching InferenceEngine enum member.

        Raises:
            ValueError: If the value is not a supported inference engine.
        """
        if isinstance(value, cls):
            return value

        if not isinstance(value, str):
            supported_engines = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unsupported inference engine type {type(value).__name__}. " f"Supported engines: {supported_engines}."
            )

        normalized_value = value.strip().lower()
        for member in cls:
            if member.value.lower() == normalized_value or member.name.lower() == normalized_value:
                return member

        supported_engines = ", ".join(member.value for member in cls)
        raise ValueError(f"Unsupported inference engine '{value}'. Supported engines: {supported_engines}.")
