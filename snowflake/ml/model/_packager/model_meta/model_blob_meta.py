from typing import cast

from typing_extensions import Unpack

from snowflake.ml.model._packager.model_meta import model_meta_schema


class ModelBlobMeta:
    """Metadata of an individual model blob (sub-model) in the packed model.

    Attributes:
        name: The name to refer the sub-model.
        model_type: The type of the model and handler to use.
        path: Path to the picked model file. It is a relative path from the model blob directory.
        handler_version: The version of the handler.
        artifacts: Optional, used in custom model to show the mapping between artifact name and relative path
            from the model blob directory.
        options: Optional, used for some model specific metadata storage
    """

    def __init__(self, **kwargs: Unpack[model_meta_schema.ModelBlobMetadataDict]) -> None:
        self.name = kwargs["name"]
        self.model_type = kwargs["model_type"]
        self.path = kwargs["path"]
        self.handler_version = kwargs["handler_version"]
        self.function_properties = kwargs.get("function_properties", {})

        self.artifacts: dict[str, str] = {}
        artifacts = kwargs.get("artifacts", None)
        if artifacts:
            self.artifacts = artifacts

        self.options: model_meta_schema.ModelBlobOptions = cast(
            model_meta_schema.ModelBlobOptions, kwargs.get("options", {})
        )

    def to_dict(self) -> model_meta_schema.ModelBlobMetadataDict:
        return model_meta_schema.ModelBlobMetadataDict(
            name=self.name,
            model_type=self.model_type,
            path=self.path,
            handler_version=self.handler_version,
            function_properties=self.function_properties,
            artifacts=self.artifacts,
            options=self.options,
        )

    @classmethod
    def from_dict(cls, blob_dict: model_meta_schema.ModelBlobMetadataDict) -> "ModelBlobMeta":
        return cls(**blob_dict)
