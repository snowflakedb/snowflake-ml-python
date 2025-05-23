from typing import Optional, Union

import pandas as pd
from sklearn import model_selection

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling._internal.estimator_utils import (
    get_module_name,
    is_single_node,
)
from snowflake.ml.modeling._internal.local_implementations.pandas_trainer import (
    PandasModelTrainer,
)
from snowflake.ml.modeling._internal.model_trainer import ModelTrainer
from snowflake.ml.modeling._internal.snowpark_implementations.distributed_hpo_trainer import (
    DistributedHPOTrainer,
)
from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_trainer import (
    SnowparkModelTrainer,
)
from snowflake.ml.modeling._internal.snowpark_implementations.xgboost_external_memory_trainer import (
    XGBoostExternalMemoryTrainer,
)
from snowflake.snowpark import DataFrame, Session

_PROJECT = "ModelDevelopment"


class ModelTrainerBuilder:
    """
    A builder class to create instances of ModelTrainer for different models and training conditions.

    This class provides methods to build instances of ModelTrainer tailored to specific machine learning
    models and training configurations like dataset's location etc. It abstracts the creation process,
    allowing the user to obtain a configured ModelTrainer for a particular model architecture or configuration.
    """

    _ENABLE_DISTRIBUTED = True

    @classmethod
    def _check_if_distributed_hpo_enabled(cls, session: Session) -> bool:
        return not is_single_node(session) and ModelTrainerBuilder._ENABLE_DISTRIBUTED is True

    @classmethod
    def _validate_external_memory_params(cls, estimator: object, batch_size: int) -> None:
        """
        Validate the params are set appropriately for external memory training.

        Args:
            estimator: Model object
            batch_size: Number of rows in each batch of data processed during training.

        Raises:
            SnowflakeMLException: If the params are not appropriate for the external memory training feature.
        """
        module_name = get_module_name(model=estimator)
        root_module_name = module_name.split(".")[0]
        if root_module_name != "xgboost":
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=RuntimeError("External memory training is only supported for XGBoost models."),
            )
        if batch_size <= 0:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ARGUMENT,
                original_exception=RuntimeError("Batch size must be >= 0 when using external memory training feature."),
            )

    @classmethod
    def build(
        cls,
        estimator: object,
        dataset: Union[DataFrame, pd.DataFrame],
        input_cols: Optional[list[str]] = None,
        label_cols: Optional[list[str]] = None,
        sample_weight_col: Optional[str] = None,
        autogenerated: bool = False,
        subproject: str = "",
        use_external_memory_version: bool = False,
        batch_size: int = -1,
    ) -> ModelTrainer:
        """
        Builder method that creates an appropriate ModelTrainer instance based on the given params.
        """
        assert input_cols is not None  # Make MyPy happy
        if isinstance(dataset, pd.DataFrame):
            return PandasModelTrainer(
                estimator=estimator,
                dataset=dataset,
                input_cols=input_cols,
                label_cols=label_cols,
                sample_weight_col=sample_weight_col,
            )
        elif isinstance(dataset, DataFrame):
            init_args = {
                "estimator": estimator,
                "dataset": dataset,
                "session": dataset._session,
                "input_cols": input_cols,
                "label_cols": label_cols,
                "sample_weight_col": sample_weight_col,
                "autogenerated": autogenerated,
                "subproject": subproject,
            }
            trainer_klass = SnowparkModelTrainer

            assert dataset._session is not None  # Make MyPy happy
            if isinstance(estimator, model_selection.GridSearchCV) or isinstance(
                estimator, model_selection.RandomizedSearchCV
            ):
                if ModelTrainerBuilder._check_if_distributed_hpo_enabled(session=dataset._session):
                    trainer_klass = DistributedHPOTrainer
            elif use_external_memory_version:
                ModelTrainerBuilder._validate_external_memory_params(
                    estimator=estimator,
                    batch_size=batch_size,
                )
                trainer_klass = XGBoostExternalMemoryTrainer
                init_args["batch_size"] = batch_size

            return trainer_klass(**init_args)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

    @classmethod
    def build_fit_predict(
        cls,
        estimator: object,
        dataset: Union[DataFrame, pd.DataFrame],
        input_cols: list[str],
        autogenerated: bool = False,
        subproject: str = "",
    ) -> ModelTrainer:
        """
        Builder method that creates an appropriate ModelTrainer instance based on the given params.
        """
        if isinstance(dataset, pd.DataFrame):
            return PandasModelTrainer(
                estimator=estimator,
                dataset=dataset,
                input_cols=input_cols,
                label_cols=None,
                sample_weight_col=None,
            )
        elif isinstance(dataset, DataFrame):
            trainer_klass = SnowparkModelTrainer
            init_args = {
                "estimator": estimator,
                "dataset": dataset,
                "session": dataset._session,
                "input_cols": input_cols,
                "label_cols": None,
                "sample_weight_col": None,
                "autogenerated": autogenerated,
                "subproject": subproject,
            }
            return trainer_klass(**init_args)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )

    @classmethod
    def build_fit_transform(
        cls,
        estimator: object,
        dataset: Union[DataFrame, pd.DataFrame],
        input_cols: list[str],
        label_cols: Optional[list[str]] = None,
        sample_weight_col: Optional[str] = None,
        autogenerated: bool = False,
        subproject: str = "",
    ) -> ModelTrainer:
        """
        Builder method that creates an appropriate ModelTrainer instance based on the given params.
        """
        if isinstance(dataset, pd.DataFrame):
            return PandasModelTrainer(
                estimator=estimator,
                dataset=dataset,
                input_cols=input_cols,
                label_cols=label_cols,
                sample_weight_col=sample_weight_col,
            )
        elif isinstance(dataset, DataFrame):
            trainer_klass = SnowparkModelTrainer
            init_args = {
                "estimator": estimator,
                "dataset": dataset,
                "session": dataset._session,
                "input_cols": input_cols,
                "label_cols": label_cols,
                "sample_weight_col": sample_weight_col,
                "autogenerated": autogenerated,
                "subproject": subproject,
            }
            return trainer_klass(**init_args)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"Unexpected dataset type: {type(dataset)}."
                "Supported dataset types: snowpark.DataFrame, pandas.DataFrame."
            )
