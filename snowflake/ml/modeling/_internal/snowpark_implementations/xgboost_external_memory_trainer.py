import inspect
import os
import tempfile
from typing import Any, Optional

import cloudpickle as cp
import pandas as pd
import pyarrow.parquet as pq

from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions,
    modeling_error_messages,
)
from snowflake.ml._internal.utils import pkg_version_utils, temp_file_utils
from snowflake.ml._internal.utils.query_result_checker import ResultValidator
from snowflake.ml._internal.utils.snowpark_dataframe_utils import (
    cast_snowpark_dataframe,
)
from snowflake.ml.modeling._internal import estimator_utils
from snowflake.ml.modeling._internal.model_specifications import (
    ModelSpecifications,
    ModelSpecificationsBuilder,
)
from snowflake.ml.modeling._internal.snowpark_implementations.snowpark_trainer import (
    SnowparkModelTrainer,
)
from snowflake.snowpark import (
    DataFrame,
    Session,
    exceptions as snowpark_exceptions,
    functions as F,
)
from snowflake.snowpark._internal.utils import (
    TempObjectType,
    random_name_for_temp_object,
)

_PROJECT = "ModelDevelopment"


def get_data_iterator(
    file_paths: list[str],
    batch_size: int,
    input_cols: list[str],
    label_cols: list[str],
    sample_weight_col: Optional[str] = None,
) -> Any:
    from typing import Optional

    import xgboost

    class ParquetDataIterator(xgboost.DataIter):
        """
        This iterator reads parquet data stored in a specified files and returns
        deserialized data, enabling seamless integration with the xgboost framework for
        machine learning tasks.
        """

        def __init__(
            self,
            file_paths: list[str],
            batch_size: int,
            input_cols: list[str],
            label_cols: list[str],
            sample_weight_col: Optional[str] = None,
        ) -> None:
            """
            Initialize the DataIterator.

            Args:
                file_paths: List of file paths containing the data.
                batch_size: Target number of rows in each batch.
                input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be used for
                    training.
                label_cols: The name(s) of one or more columns in a DataFrame representing the target variable(s)
                    to learn.
                sample_weight_col: The column name representing the weight of training examples.
            """
            self._file_paths = file_paths
            self._batch_size = batch_size
            self._input_cols = input_cols
            self._label_cols = label_cols
            self._sample_weight_col = sample_weight_col

            # File index
            self._it = 0
            # Pandas dataframe containing temp data
            self._df = None
            # XGBoost will generate some cache files under current directory with the prefix
            # "cache"
            cache_dir_name = tempfile.mkdtemp()
            super().__init__(cache_prefix=os.path.join(cache_dir_name, "cache"))

        def next(self, batch_consumer_fn) -> bool | int:  # type: ignore[no-untyped-def]
            """Advance the iterator by 1 step and pass the data to XGBoost's batch_consumer_fn.
            This function is called by XGBoost during the construction of ``DMatrix``

            Args:
                batch_consumer_fn: batch consumer function

            Returns:
                False/0 if there is no more data, else True/1.
            """
            while (self._df is None) or (self._df.shape[0] < self._batch_size):
                # Read files and append data to temp df until batch size is reached.
                if self._it == len(self._file_paths):
                    break
                new_df = pq.read_table(self._file_paths[self._it]).to_pandas()
                self._it += 1

                if self._df is None:
                    self._df = new_df
                else:
                    self._df = pd.concat([self._df, new_df], ignore_index=True)

            if (self._df is None) or (self._df.shape[0] == 0):
                # No more data
                return False

            # Slice the temp df and save the remainder in the temp df
            batch_end_index = min(self._batch_size, self._df.shape[0])
            batch_df = self._df.iloc[:batch_end_index]
            self._df = self._df.truncate(before=batch_end_index).reset_index(drop=True)

            # TODO(snandamuri): Make it proper to support categorical features, etc.
            func_args = {
                "data": batch_df[self._input_cols],
                "label": batch_df[self._label_cols].squeeze(),
            }
            if self._sample_weight_col is not None:
                func_args["weight"] = batch_df[self._sample_weight_col].squeeze()

            batch_consumer_fn(**func_args)
            # Return True to let XGBoost know we haven't seen all the files yet.
            return True

        def reset(self) -> None:
            """Reset the iterator to its beginning"""
            self._it = 0

    return ParquetDataIterator(
        file_paths=file_paths,
        batch_size=batch_size,
        input_cols=input_cols,
        label_cols=label_cols,
        sample_weight_col=sample_weight_col,
    )


def train_xgboost_model(
    estimator: object,
    file_paths: list[str],
    batch_size: int,
    input_cols: list[str],
    label_cols: list[str],
    sample_weight_col: Optional[str] = None,
) -> object:
    """
    Function to train XGBoost models using the external memory version of XGBoost.
    """
    import xgboost

    def _objective_decorator(func):  # type: ignore[no-untyped-def]
        def inner(preds, dmatrix):  # type: ignore[no-untyped-def]
            """internal function"""
            labels = dmatrix.get_label()
            return func(labels, preds)

        return inner

    assert isinstance(estimator, xgboost.XGBModel)
    params = estimator.get_xgb_params()
    obj = None

    if isinstance(estimator, xgboost.XGBClassifier):
        # TODO (snandamuri): Find better way to get expected_classes
        # Set: self.classes_, self.n_classes_
        expected_classes = pd.unique(pq.read_table(file_paths[0]).to_pandas()[label_cols].squeeze())
        estimator.n_classes_ = len(expected_classes)
        if callable(estimator.objective):
            obj = _objective_decorator(estimator.objective)  # type: ignore[no-untyped-call]
            # Use default value. Is it really not used ?
            params["objective"] = "binary:logistic"

        if len(expected_classes) > 2:
            # Switch to using a multiclass objective in the underlying XGB instance
            if params.get("objective", None) != "multi:softmax":
                params["objective"] = "multi:softprob"
            params["num_class"] = len(expected_classes)

    if "tree_method" not in params.keys() or params["tree_method"] is None or params["tree_method"].lower() == "exact":
        params["tree_method"] = "hist"

    if (
        "grow_policy" not in params.keys()
        or params["grow_policy"] is None
        or params["grow_policy"].lower() != "depthwise"
    ):
        params["grow_policy"] = "depthwise"

    it = get_data_iterator(
        file_paths=file_paths,
        batch_size=batch_size,
        input_cols=input_cols,
        label_cols=label_cols,
        sample_weight_col=sample_weight_col,
    )
    Xy = xgboost.DMatrix(it)
    estimator._Booster = xgboost.train(
        params,
        Xy,
        estimator.get_num_boosting_rounds(),
        evals=[],
        early_stopping_rounds=estimator.early_stopping_rounds,
        evals_result=None,
        obj=obj,
        custom_metric=estimator.eval_metric,
        verbose_eval=None,
        xgb_model=None,
        callbacks=None,
    )
    return estimator


cp.register_pickle_by_value(inspect.getmodule(get_data_iterator))
cp.register_pickle_by_value(inspect.getmodule(train_xgboost_model))


class XGBoostExternalMemoryTrainer(SnowparkModelTrainer):
    """
    When working with large datasets, training XGBoost models traditionally requires loading the entire dataset into
    memory, which can be costly and sometimes infeasible due to memory constraints. To solve this problem, XGBoost
    provides support for loading data from external memory using a built-in data parser. With this feature enabled,
    the training process occurs in a two-step approach:
        Preprocessing Step: Input data is read and parsed into an internal format, such as CSR, CSC, or sorted CSC.
            Processed state is appended to an in-memory buffer. Once the buffer reaches a predefined size, it is
            written out to disk as a page.
        Tree Construction Step: During the tree construction phase, the data pages stored on disk are streamed via
            a multi-threaded pre-fetcher, allowing the model to efficiently access and process the data without
            overloading memory.
    """

    def __init__(
        self,
        estimator: object,
        dataset: DataFrame,
        session: Session,
        input_cols: list[str],
        label_cols: Optional[list[str]],
        sample_weight_col: Optional[str],
        autogenerated: bool = False,
        subproject: str = "",
        batch_size: int = 10000,
    ) -> None:
        """
        Initializes the XGBoostExternalMemoryTrainer with a model, a Snowpark DataFrame, feature, and label column
        names, etc.

        Args:
            estimator: SKLearn compatible estimator or transformer object.
            dataset: The dataset used for training the model.
            session: Snowflake session object to be used for training.
            input_cols: The name(s) of one or more columns in a DataFrame containing a feature to be used for training.
            label_cols: The name(s) of one or more columns in a DataFrame representing the target variable(s) to learn.
            sample_weight_col: The column name representing the weight of training examples.
            autogenerated: A boolean denoting if the trainer is being used by autogenerated code or not.
            subproject: subproject name to be used in telemetry.
            batch_size: Number of the rows in the each batch processed during training.
        """
        super().__init__(
            estimator=estimator,
            dataset=dataset,
            session=session,
            input_cols=input_cols,
            label_cols=label_cols,
            sample_weight_col=sample_weight_col,
            autogenerated=autogenerated,
            subproject=subproject,
        )
        self._batch_size = batch_size

    def _get_xgb_external_memory_fit_wrapper_sproc(
        self,
        model_spec: ModelSpecifications,
        session: Session,
        statement_params: dict[str, str],
        import_file_paths: list[str],
    ) -> Any:
        fit_sproc_name = random_name_for_temp_object(TempObjectType.PROCEDURE)

        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=model_spec.pkgDependencies, session=self.session
        )

        @F.sproc(
            is_permanent=False,
            name=fit_sproc_name,
            packages=list(["snowflake-snowpark-python"] + relaxed_dependencies),
            replace=True,
            session=session,
            statement_params=statement_params,
            anonymous=True,
            imports=list(import_file_paths),
        )  # type: ignore[misc]
        def fit_wrapper_sproc(
            session: Session,
            dataset_stage_name: str,
            batch_size: int,
            input_cols: list[str],
            label_cols: list[str],
            sample_weight_col: Optional[str],
            statement_params: dict[str, str],
        ) -> str:
            import os
            import sys

            import cloudpickle as cp

            local_transform_file_name = temp_file_utils.get_temp_file_path()

            session.file.get(
                stage_location=dataset_stage_name,
                target_directory=local_transform_file_name,
                statement_params=statement_params,
            )

            local_transform_file_path = os.path.join(
                local_transform_file_name, os.listdir(local_transform_file_name)[0]
            )
            with open(local_transform_file_path, mode="r+b") as local_transform_file_obj:
                estimator = cp.load(local_transform_file_obj)

            data_files = [
                os.path.join(sys._xoptions["snowflake_import_directory"], filename)
                for filename in os.listdir(sys._xoptions["snowflake_import_directory"])
                if filename.startswith(dataset_stage_name)
            ]

            estimator = train_xgboost_model(
                estimator=estimator,
                file_paths=data_files,
                batch_size=batch_size,
                input_cols=input_cols,
                label_cols=label_cols,
                sample_weight_col=sample_weight_col,
            )

            local_result_file_name = temp_file_utils.get_temp_file_path()
            with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                cp.dump(estimator, local_result_file_obj)

            session.file.put(
                local_file_name=local_result_file_name,
                stage_location=dataset_stage_name,
                auto_compress=False,
                overwrite=True,
                statement_params=statement_params,
            )

            # Note: you can add something like  + "|" + str(df) to the return string
            # to pass debug information to the caller.
            return str(os.path.basename(local_result_file_name))

        return fit_wrapper_sproc

    def _write_training_data_to_stage(self, dataset_stage_name: str) -> list[str]:
        """
        Materializes the training to the specified stage and returns the list of stage file paths.

        Args:
            dataset_stage_name: Target stage to materialize training data.

        Returns:
            List of stage file paths that contain the materialized data.
        """
        # Stage data.
        dataset = cast_snowpark_dataframe(self.dataset)
        remote_file_path = f"{dataset_stage_name}/{dataset_stage_name}.parquet"
        copy_response = dataset.write.copy_into_location(  # type:ignore[call-overload]
            remote_file_path, file_format_type="parquet", header=True, overwrite=True
        )
        ResultValidator(result=copy_response).has_dimensions(expected_rows=1).validate()
        data_file_paths = [f"@{row.name}" for row in self.session.sql(f"LIST @{dataset_stage_name}").collect()]
        return data_file_paths

    def train(self) -> object:
        """
        Runs hyper parameter optimization by distributing the tasks across warehouse.

        Returns:
            Trained model

        Raises:
            SnowflakeMLException: For known types of user and system errors.
            e: For every unexpected exception from SnowflakeClient.
        """
        statement_params = telemetry.get_function_usage_statement_params(
            project=_PROJECT,
            subproject=self._subproject,
            function_name=telemetry.get_statement_params_full_func_name(inspect.currentframe(), self._class_name),
            api_calls=[Session.call],
            custom_tags=None,
        )
        temp_stage_name = estimator_utils.create_temp_stage(self.session)
        estimator_utils.upload_model_to_stage(
            stage_name=temp_stage_name,
            estimator=self.estimator,
            session=self.session,
            statement_params=statement_params,
        )
        data_file_paths = self._write_training_data_to_stage(dataset_stage_name=temp_stage_name)

        # Call fit sproc
        model_spec = ModelSpecificationsBuilder.build(model=self.estimator)
        fit_wrapper = self._get_xgb_external_memory_fit_wrapper_sproc(
            model_spec=model_spec,
            session=self.session,
            statement_params=statement_params,
            import_file_paths=data_file_paths,
        )

        try:
            sproc_export_file_name = fit_wrapper(
                self.session,
                temp_stage_name,
                self._batch_size,
                self.input_cols,
                self.label_cols,
                self.sample_weight_col,
                statement_params,
            )
        except snowpark_exceptions.SnowparkClientException as e:
            if "fit() missing 1 required positional argument: 'y'" in str(e):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.NOT_FOUND,
                    original_exception=RuntimeError(modeling_error_messages.ATTRIBUTE_NOT_SET.format("label_cols")),
                ) from e
            raise e

        if "|" in sproc_export_file_name:
            fields = sproc_export_file_name.strip().split("|")
            sproc_export_file_name = fields[0]

        return self._fetch_model_from_stage(
            dir_path=temp_stage_name,
            file_name=sproc_export_file_name,
            statement_params=statement_params,
        )
