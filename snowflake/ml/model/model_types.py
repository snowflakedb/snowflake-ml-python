from typing import TYPE_CHECKING, TypeVar, Union

if TYPE_CHECKING:
    import sklearn.base
    import sklearn.pipeline

    import snowflake.ml.model.custom_model


CustomModelType = TypeVar("CustomModelType", bound="snowflake.ml.model.custom_model.CustomModel")

ModelType = Union[
    "snowflake.ml.model.custom_model.CustomModel", "sklearn.base.BaseEstimator", "sklearn.pipeline.Pipeline"
]
"""This is defined as the type that Snowflake native model packaging could accept.
To avoid importing a lot of modules that users might not need too earlier by defining this Union type,
we have to use all string-based typing hints here.

Here is all acceptable types of Snowflake native model packaging and its handler file in _handlers/ folder.

| Type                            | Handler File | Handler             |
|---------------------------------|--------------|---------------------|
| snowflake.ml.model.custom_model.CustomModel | custom.py    | _CustomModelHandler |
| sklearn.base.BaseEstimator      | sklearn.py   | _SKLModelHandler    |
| sklearn.pipeline.Pipeline       | sklearn.py   | _SKLModelHandler    |
"""
