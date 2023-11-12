import pathlib
from typing import Optional, TypedDict

import importlib_resources
from typing_extensions import NotRequired


class HandlerGenerateOptions(TypedDict):
    max_batch_size: NotRequired[int]


class HandlerGenerator:
    HANDLER_NAME = "infer"

    def __init__(
        self,
        model_file_stage_path: pathlib.PurePosixPath,
    ) -> None:
        self.model_file_stage_path = model_file_stage_path

    def generate(
        self,
        handler_file_path: pathlib.Path,
        target_method: str,
        options: Optional[HandlerGenerateOptions] = None,
    ) -> None:
        if options is None:
            options = {}
        handler_template = (
            importlib_resources.files("snowflake.ml.model._module_model.module_method")
            .joinpath("infer_handler.py_template")  # type: ignore[no-untyped-call]
            .read_text()
        )

        udf_code = handler_template.format(
            model_file_name=self.model_file_stage_path.name,
            target_method=target_method,
            max_batch_size=options.get("max_batch_size", None),
            handler_name=HandlerGenerator.HANDLER_NAME,
        )
        with open(handler_file_path, "w", encoding="utf-8") as f:
            f.write(udf_code)
            f.flush()
