==================
snowflake.ml.model
==================

.. automodule:: snowflake.ml.model
    :noindex:

.. currentmodule:: snowflake.ml.model

.. rubric:: Core Classes

.. autosummary::
    :toctree: api/model

    Model
    ModelVersion
    HuggingFacePipelineModel
    TransformersPipeline

.. rubric:: Batch Inference

.. autosummary::
    :toctree: api/model

    InputSpec
    OutputSpec
    JobSpec
    SaveMode
    InputFormat
    FileEncoding
    ColumnHandlingOptions

.. rubric:: Model Logging Options

.. autosummary::
    :toctree: api/model

    CodePath
    ExportMode
    Volatility

snowflake.ml.model.custom_model
---------------------------------

.. currentmodule:: snowflake.ml.model.custom_model

.. rubric:: Classes

.. autosummary::
    :toctree: api/model

    MethodRef
    ModelRef
    ModelContext
    CustomModel

.. rubric:: Decorators

.. autosummary::
    :toctree: api/model

    inference_api
    partitioned_api

snowflake.ml.model.model_signature
----------------------------------

.. currentmodule:: snowflake.ml.model.model_signature

.. rubric:: Classes

.. autosummary::
    :toctree: api/model

    DataType
    BaseFeatureSpec
    FeatureSpec
    FeatureGroupSpec
    ParamSpec
    ParamGroupSpec
    ModelSignature

.. rubric:: Methods

.. autosummary::
    :toctree: api/model

    infer_signature

snowflake.ml.model.openai_signatures
------------------------------------

.. currentmodule:: snowflake.ml.model.openai_signatures

.. rubric:: Attributes

.. autosummary::
    :toctree: api/model

    OPENAI_CHAT_SIGNATURE
    OPENAI_CHAT_SIGNATURE_WITH_CONTENT_FORMAT_STRING
    OPENAI_CHAT_WITH_PARAMS_SIGNATURE
    OPENAI_CHAT_WITH_PARAMS_SIGNATURE_WITH_CONTENT_FORMAT_STRING
