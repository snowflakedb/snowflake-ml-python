import dataclasses
import functools
import types
from typing import Callable, Optional

from snowflake.ml import model
from snowflake.ml.registry._manager import model_manager


@dataclasses.dataclass(frozen=True)
class ExperimentInfo:
    """Serializable information identifying a Experiment"""

    fully_qualified_name: str
    run_name: str


class ExperimentInfoPatcher:
    """Context manager that patches ModelManager.log_model to include experiment information.

    This class maintains a stack of active experiment contexts and ensures that
    log_model calls are automatically tagged with the appropriate experiment info.
    """

    # Store original method at class definition time to avoid recursive patching
    _original_log_model: Callable[..., model.ModelVersion] = model_manager.ModelManager.log_model

    # Stack of active experiment_info contexts for nested experiment support
    _experiment_info_stack: list[ExperimentInfo] = []

    def __init__(self, experiment_info: ExperimentInfo) -> None:
        self._experiment_info = experiment_info

    def __enter__(self) -> "ExperimentInfoPatcher":
        # Only patch ModelManager.log_model if we're the first patcher to avoid nested patching
        if not ExperimentInfoPatcher._experiment_info_stack:

            @functools.wraps(ExperimentInfoPatcher._original_log_model)
            def patched(*args, **kwargs) -> model.ModelVersion:  # type: ignore[no-untyped-def]
                # Use the most recent (top of stack) experiment_info for nested contexts
                current_experiment_info = ExperimentInfoPatcher._experiment_info_stack[-1]
                return ExperimentInfoPatcher._original_log_model(
                    *args, **kwargs, experiment_info=current_experiment_info
                )

            model_manager.ModelManager.log_model = patched  # type: ignore[method-assign]

        ExperimentInfoPatcher._experiment_info_stack.append(self._experiment_info)
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[types.TracebackType],
    ) -> None:
        ExperimentInfoPatcher._experiment_info_stack.pop()

        # Restore original method when no patches are active to clean up properly
        if not ExperimentInfoPatcher._experiment_info_stack:
            model_manager.ModelManager.log_model = (  # type: ignore[method-assign]
                ExperimentInfoPatcher._original_log_model
            )
