"""Disables the distributed implementation of Grid Search and Randomized Search CV"""
from snowflake.ml.modeling._internal.model_trainer_builder import ModelTrainerBuilder

ModelTrainerBuilder._ENABLE_DISTRIBUTED = False
