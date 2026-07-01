from typing import Any, Optional

import yaml

from snowflake.ml.model._client.model import batch_inference_specs


class InferenceJobServiceSpec:
    """Builds the YAML body for ``EXECUTE INFERENCE JOB SERVICE``."""

    def __init__(self) -> None:
        self._input: Optional[dict[str, Any]] = None
        self._output: Optional[dict[str, Any]] = None
        self._resources: Optional[dict[str, Any]] = None
        self._inference: Optional[dict[str, Any]] = None
        self._image_build: Optional[dict[str, Any]] = None

    def clear(self) -> None:
        self._input = None
        self._output = None
        self._resources = None
        self._inference = None
        self._image_build = None

    def add_input_spec(self, input_spec: batch_inference_specs.Input) -> "InferenceJobServiceSpec":
        # ``params`` and ``column_handling`` are emitted as raw dicts; the
        # server handles encoding for both.
        self._input = input_spec.model_dump(mode="json", exclude_none=True)
        return self

    def add_output_spec(self, output_spec: batch_inference_specs.Output) -> "InferenceJobServiceSpec":
        self._output = output_spec.model_dump(mode="json", exclude_none=True)
        return self

    def add_resources_spec(self, resources_spec: batch_inference_specs.Resources) -> "InferenceJobServiceSpec":
        dumped = resources_spec.model_dump(mode="json", exclude_none=True)
        self._resources = dumped if dumped else None
        return self

    def add_inference_spec(self, inference_spec: batch_inference_specs.Inference) -> "InferenceJobServiceSpec":
        dumped = inference_spec.model_dump(mode="json", exclude_none=True)
        self._inference = dumped if dumped else None
        return self

    def add_image_build_spec(self, image_build_spec: batch_inference_specs.ImageBuild) -> "InferenceJobServiceSpec":
        self._image_build = image_build_spec.model_dump(mode="json", exclude_none=True)
        return self

    def save(self) -> str:
        """Return the YAML body as a string.

        Raises:
            ValueError: If ``output`` has not been added.

        Returns:
            YAML string of the batch inference spec body.
        """
        if self._output is None:
            raise ValueError("batch inference job: output spec is required. Call add_output_spec().")
        body: dict[str, Any] = {"output": self._output}
        if self._input is not None:
            body["input"] = self._input
        if self._resources is not None:
            body["resources"] = self._resources
        if self._inference is not None:
            body["inference"] = self._inference
        if self._image_build is not None:
            body["image_build"] = self._image_build
        # Order keys to match the design doc for readability: input, output,
        # resources, inference, image_build.
        ordered: dict[str, Any] = {}
        for key in ("input", "output", "resources", "inference", "image_build"):
            if key in body:
                ordered[key] = body[key]
        return yaml.safe_dump(ordered, sort_keys=False)
