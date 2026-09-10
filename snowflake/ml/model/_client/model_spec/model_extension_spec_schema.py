"""Wire types for model extension specification version 2.0."""

from typing import Any, TypedDict

from typing_extensions import Required


class ModelBlobArtifactSpec(TypedDict, total=False):
    """An entry in ``model.details.models.*.artifacts``."""

    path: str
    type: str


class ModelBlobExplainabilitySpec(TypedDict, total=False):
    """Explainability metadata on a USER_MODEL blob."""

    algorithm: str
    background_data: str


class ModelBlobSpec(TypedDict, total=False):
    """A model artifact entry in ``model.details.models``."""

    model_type: str
    path: str
    handler_version: str
    artifacts: dict[str, ModelBlobArtifactSpec]
    explainability: ModelBlobExplainabilitySpec
    options: dict[str, Any] | None


class ModelDetailsSpec(TypedDict, total=False):
    """Model details, including named model artifacts."""

    models: Required[dict[str, ModelBlobSpec]]


class ModelClientSpec(TypedDict, total=False):
    """Client identity recorded on a version 2.0 specification."""

    name: Required[str]
    version: Required[str]
    min_version: str


class ModelSectionSpec(TypedDict, total=False):
    """The model section of a version 2.0 specification."""

    type: Required[str]
    framework: str
    client: ModelClientSpec
    details: Required[ModelDetailsSpec]
    target_platforms: list[str] | None
    task: str


class ServingEnvironmentSpec(TypedDict, total=False):
    """A named serving environment."""

    type: str
    python_version: str
    snowpark_ml_version: str
    conda: str
    pip: str
    cuda_version: str | None
    imports: list[str]
    artifact_repository_map: dict[str, Any]
    resource_constraint: dict[str, Any]


ServingEnvironmentsSpec = dict[str, ServingEnvironmentSpec]


class ServingFunctionPropertiesSpec(TypedDict, total=False):
    """Per-function properties on a version 2.0 specification."""

    method: str
    max_batch_size: int
    is_partition: bool
    is_explain: bool
    case_sensitive: bool


class ServingFunctionSpec(TypedDict, total=False):
    """A function exposed by the model specification."""

    env: list[str]
    type: str
    handler: str
    volatility: str
    signature: Required[dict[str, Any]]
    properties: ServingFunctionPropertiesSpec | None


class ServingSectionSpec(TypedDict, total=False):
    """The serving section of a version 2.0 specification."""

    env: ServingEnvironmentsSpec | None
    functions: Required[dict[str, ServingFunctionSpec]]


class ModelExtensionSpecSchema(TypedDict, total=False):
    """Top-level model extension specification version 2.0.

    The service may add fields to any envelope. These wire types describe the
    fields consumed by the client; parsing retains the complete input mapping.
    """

    version: Required[str | float]
    model: Required[ModelSectionSpec]
    serving: Required[ServingSectionSpec]
    user_data: dict[str, Any]
