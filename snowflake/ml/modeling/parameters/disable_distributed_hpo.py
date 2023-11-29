"""Disables the distributed implementation of Grid Search and Randomized Search CV"""
from snowflake.ml.modeling.model_selection.grid_search_cv import GridSearchCV
from snowflake.ml.modeling.model_selection.randomized_search_cv import (
    RandomizedSearchCV,
)

GridSearchCV._ENABLE_DISTRIBUTED = False
RandomizedSearchCV._ENABLE_DISTRIBUTED = False
