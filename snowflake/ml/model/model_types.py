from typing import Any

ModelType = Any
"""This is defined as the type that Snowflake native model packaging could accept.
It should have been a Union, however, to avoid importing a lot of modules that users might not need too earlier by
defining this Union type, we decide to use Any here and use lazy type check in very handler.

Here is all acceptable types of Snowflake native model packaging and its handler file in _handlers/ folder.

| Type                            | Handler File | Handler             |
|---------------------------------|--------------|---------------------|
| snowflake.ml.model.custom_model | custom.py    | _CustomModelHandler |
| sklearn.base.BaseEstimator      | sklearn.py   | _SKLModelHandler    |
| sklearn.pipeline.Pipeline       | sklearn.py   | _SKLModelHandler    |
"""
