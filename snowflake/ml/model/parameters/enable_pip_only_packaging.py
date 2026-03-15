"""Enables pip-only packaging for models (no conda.yml when there are no conda dependencies)."""

from snowflake.ml.model._packager.model_env import model_env

model_env._ENABLE_PIP_ONLY_PACKAGING = True
