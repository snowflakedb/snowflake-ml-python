#!/usr/bin/env python3
import inspect
import os
import posixpath
import tempfile
from itertools import chain
from typing import Any, Callable, Optional, Union

import cloudpickle as cp
import numpy as np
import pandas as pd
from sklearn import __version__ as skversion, pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import metaestimators

from snowflake import snowpark
from snowflake.ml._internal import file_utils, telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.lineage import lineage_utils
from snowflake.ml._internal.utils import snowpark_dataframe_utils, temp_file_utils
from snowflake.ml.data import data_source
from snowflake.ml.model.model_signature import (
    ModelSignature,
    _infer_signature,
    _truncate_data,
)
from snowflake.ml.modeling._internal.model_transformer_builder import (
    ModelTransformerBuilder,
)
from snowflake.ml.modeling.framework import _utils, base
from snowflake.snowpark import Session, functions as F
from snowflake.snowpark._internal import utils as snowpark_utils

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Framework"

INFER_SIGNATURE_MAX_ROWS = 100


def _final_step_has(attr: str) -> Callable[..., bool]:
    """Check that final_estimator has `attr`. Used together with `available_if` in `Pipeline`."""

    def check(self: "Pipeline") -> bool:
        # Raises original `AttributeError` if `attr` does not exist.
        # If `attr`` exists but is not callable, then False will be returned.
        return callable(getattr(self.steps[-1][1], attr))

    return check


def has_callable_attr(obj: object, attr: str) -> bool:
    """
    Check if the object `obj` has a callable attribute with the name `attr`.

    Args:
        obj: The object to check.
        attr: The name of the attribute to check for.

    Returns:
        True if the attribute is callable, False otherwise.
    """
    return callable(getattr(obj, attr, None))


def _get_column_indices(all_columns: list[str], target_columns: list[str]) -> list[int]:
    """
    Extract the indices of the target_columns from all_columns.

    Args:
        all_columns: List of all the columns in a dataframe.
        target_columns: List of target column names to be extracted.

    Returns:
        Return the list of indices of target column in the original column array.

    Raises:
        SnowflakeMLException: If the target column is not present in the original column array.
    """
    column_indices = []
    for col in target_columns:
        found = False
        for i, c in enumerate(all_columns):
            if c == col:
                column_indices.append(i)
                found = True
                break
        if not found:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.NOT_FOUND,
                original_exception=ValueError(
                    f"Selected column {col} is not found in the input dataframe. Columns in the input df: {all_columns}"
                ),
            )
    return column_indices


class Pipeline(base.BaseTransformer):
    def __init__(self, steps: list[tuple[str, Any]]) -> None:
        """
        Pipeline of transforms.

        Sequentially apply a list of transforms.
        Intermediate steps of the pipeline must be 'transforms', that is, they
        must implement `fit` and `transform` methods.
        The final step can be a transform or estimator, that is, it must implement
        `fit` and `transform`/`predict` methods.

        Args:
            steps: List of (name, transform) tuples (implementing `fit`/`transform`) that
                are chained in sequential order. The last transform can be an
                estimator.
        """
        super().__init__()
        self.steps = steps
        # TODO(snandamuri): SKLearn pipeline expects last step(and only the last step) to be an estimator obj or a dummy
        # estimator(like None or passthrough). Currently this Pipeline class works with a list of all
        # transforms or a list of transforms ending with an estimator. Should we change this implementation
        # to only work with list of steps ending with an estimator or a dummy estimator like SKLearn?
        self._is_final_step_estimator = Pipeline._is_estimator(steps[-1][1])
        self._is_fitted = False
        self._feature_names_in: list[np.ndarray[Any, np.dtype[Any]]] = []
        self._n_features_in: list[int] = []
        self._transformers_to_input_indices: dict[str, list[int]] = {}
        self._modifies_label_or_sample_weight = True

        self._model_signature_dict: Optional[dict[str, ModelSignature]] = None

        deps: set[str] = {f"pandas=={pd.__version__}", f"scikit-learn=={skversion}"}
        for _, obj in steps:
            if isinstance(obj, base.BaseTransformer):
                deps = deps | set(obj._get_dependencies())
        self._deps = list(deps)
        self._sklearn_object = None
        self.label_cols = self._get_label_cols()
        self._is_convertible_to_sklearn = self._is_convertible_to_sklearn_object()

        self._send_pipeline_configuration_telemetry()

    @staticmethod
    def _is_estimator(obj: object) -> bool:
        # TODO(SNOW-723770): Figure out a better way to identify estimator objects.
        return has_callable_attr(obj, "fit") and has_callable_attr(obj, "predict")

    @staticmethod
    def _is_transformer(obj: object) -> bool:
        return has_callable_attr(obj, "fit") and has_callable_attr(obj, "transform")

    def _get_transformers(self) -> list[tuple[str, Any]]:
        return self.steps[:-1] if self._is_final_step_estimator else self.steps

    def _get_estimator(self) -> Optional[tuple[str, Any]]:
        return self.steps[-1] if self._is_final_step_estimator else None

    def _validate_steps(self) -> None:
        for name, t in self._get_transformers():
            if not Pipeline._is_transformer(t):
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.INVALID_ATTRIBUTE,
                    original_exception=TypeError(
                        "All intermediate steps should be "
                        "transformers and implement both fit() and transform() methods, but"
                        f"{name} (type {type(t)}) doesn't."
                    ),
                )

    def _reset(self) -> None:
        super()._reset()
        self._feature_names_in = []
        self._n_features_in = []
        self._transformers_to_input_indices = {}

    def _is_convertible_to_sklearn_object(self) -> bool:
        """Checks if the pipeline can be converted to a native sklearn pipeline.
        - We can not create an sklearn pipeline if its label or sample weight column are
          modified in the pipeline.
        - We can not create an sklearn pipeline if any of its steps cannot be converted to an sklearn pipeline
        - We can not create an sklearn pipeline if input columns are specified in any step other than
          the first step

        Returns:
            True if the pipeline can be converted to a native sklearn pipeline, else false.
        """
        if self._is_pipeline_modifying_label_or_sample_weight():
            return False

        # check that nested pipelines can be converted to sklearn
        for _, base_estimator in self.steps:
            if hasattr(base_estimator, "_is_convertible_to_sklearn_object"):
                if not base_estimator._is_convertible_to_sklearn_object():
                    return False

        # check that no column after the first column has 'input columns' set.
        for _, base_estimator in self.steps[1:]:
            if base_estimator.get_input_cols():
                # We only want Falsy values - None and []
                return False
        return True

    def _is_pipeline_modifying_label_or_sample_weight(self) -> bool:
        """
        Checks if pipeline is modifying label or sample_weight columns.

        Returns:
            True if pipeline is processing label or sample_weight columns, False otherwise.
        """
        estimator_step = self._get_estimator()
        if not estimator_step:
            return False

        target_cols = set(
            estimator_step[1].get_label_cols()
            + ([] if not estimator_step[1].get_sample_weight_col() else [estimator_step[1].get_sample_weight_col()])
        )
        processed_cols = set(chain.from_iterable([trans.get_input_cols() for (_, trans) in self._get_transformers()]))
        return len(target_cols & processed_cols) > 0

    def _get_sanitized_list_of_columns(self, columns: list[str]) -> list[str]:
        """
        Removes the label and sample_weight columns from the input list of columns and returns the results for the
        purpous of computing column indices for SKLearn ColumnTransformer objects.

        Args:
            columns: List if input columns for a transformer step.

        Returns:
            Returns a list of columns without label and sample_weight columns.
        """
        estimator_step = self._get_estimator()
        if not estimator_step:
            return columns

        target_cols = set(
            estimator_step[1].get_label_cols()
            + ([] if not estimator_step[1].get_sample_weight_col() else [estimator_step[1].get_sample_weight_col()])
        )

        return [c for c in columns if c not in target_cols]

    def _append_step_feature_consumption_info(self, step_name: str, all_cols: list[str], input_cols: list[str]) -> None:
        if self._modifies_label_or_sample_weight:
            all_cols = self._get_sanitized_list_of_columns(all_cols)
            self._feature_names_in.append(np.asarray(all_cols, dtype=object))
            self._n_features_in.append(len(all_cols))
            self._transformers_to_input_indices[step_name] = _get_column_indices(
                all_columns=all_cols, target_columns=input_cols
            )

    def _transform_dataset(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        transformed_dataset = dataset
        for _, trans in self._get_transformers():
            transformed_dataset = trans.transform(transformed_dataset)
        return transformed_dataset

    def _fit_transform_dataset(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        self._reset()
        self._modifies_label_or_sample_weight = not self._is_pipeline_modifying_label_or_sample_weight()
        transformed_dataset = dataset
        for name, trans in self._get_transformers():
            self._append_step_feature_consumption_info(
                step_name=name, all_cols=transformed_dataset.columns[:], input_cols=trans.get_input_cols()
            )
            trans.fit(transformed_dataset)
            transformed_dataset = trans.transform(transformed_dataset)

        return transformed_dataset

    def _upload_model_to_stage(self, stage_name: str, estimator: object, session: Session) -> tuple[str, str]:
        """
        Util method to pickle and upload the model to a temp Snowflake stage.

        Args:
            stage_name: Stage name to save model.
            estimator: the pipeline estimator itself
            session: Session object

        Returns:
            a tuple containing stage file paths for pickled input model for training and location to store trained
            models(response from training sproc).
        """
        # Create a temp file and dump the transform to that file.
        local_transform_file_name = temp_file_utils.get_temp_file_path()
        with open(local_transform_file_name, mode="w+b") as local_transform_file:
            cp.dump(estimator, local_transform_file)

        # Use posixpath to construct stage paths
        stage_transform_file_name = posixpath.join(stage_name, os.path.basename(local_transform_file_name))
        stage_result_file_name = posixpath.join(stage_name, os.path.basename(local_transform_file_name))

        # Put locally serialized transform on stage.
        session.file.put(
            local_transform_file_name,
            stage_transform_file_name,
            auto_compress=False,
            overwrite=True,
        )

        temp_file_utils.cleanup_temp_files([local_transform_file_name])
        return (stage_transform_file_name, stage_result_file_name)

    def _fit_snowpark_dataframe_within_one_sproc(self, session: Session, dataset: snowpark.DataFrame) -> None:
        # Extract queries that generated the dataframe. We will need to pass it to score procedure.
        sql_queries = dataset.queries["queries"]

        # Zip the current snowml package
        with tempfile.TemporaryDirectory() as tmpdir:
            snowml_zip_module_filename = os.path.join(tmpdir, "snowflake-ml-python.zip")
            file_utils.zip_python_package(snowml_zip_module_filename, "snowflake.ml")
            imports = [snowml_zip_module_filename]

            sproc_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.PROCEDURE)
            required_deps = self._deps
            sproc_statement_params = telemetry.get_function_usage_statement_params(
                project=_PROJECT,
                subproject="PIPELINE",
                function_name=telemetry.get_statement_params_full_func_name(
                    inspect.currentframe(), self.__class__.__name__
                ),
                api_calls=[F.sproc],
            )
            transform_stage_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
            stage_creation_query = f"CREATE OR REPLACE TEMPORARY STAGE {transform_stage_name};"
            session.sql(stage_creation_query).collect()
            (stage_estimator_file_name, stage_result_file_name) = self._upload_model_to_stage(
                transform_stage_name, self, session
            )

            def pipeline_within_one_sproc(
                session: Session,
                sql_queries: list[str],
                stage_estimator_file_name: str,
                stage_result_file_name: str,
                sproc_statement_params: dict[str, str],
            ) -> str:
                import os

                import cloudpickle as cp
                import pandas as pd

                for query in sql_queries[:-1]:
                    _ = session.sql(query).collect(statement_params=sproc_statement_params)
                sp_df = session.sql(sql_queries[-1])
                df: pd.DataFrame = sp_df.to_pandas(statement_params=sproc_statement_params)
                df.columns = sp_df.columns

                local_estimator_file_name = temp_file_utils.get_temp_file_path()

                session.file.get(stage_estimator_file_name, local_estimator_file_name)

                local_estimator_file_path = os.path.join(
                    local_estimator_file_name, os.listdir(local_estimator_file_name)[0]
                )
                with open(local_estimator_file_path, mode="r+b") as local_estimator_file_obj:
                    estimator = cp.load(local_estimator_file_obj)

                estimator.fit(df)

                local_result_file_name = temp_file_utils.get_temp_file_path()

                with open(local_result_file_name, mode="w+b") as local_result_file_obj:
                    cp.dump(estimator, local_result_file_obj)

                session.file.put(
                    local_result_file_name,
                    stage_result_file_name,
                    auto_compress=False,
                    overwrite=True,
                    statement_params=sproc_statement_params,
                )

                return str(os.path.basename(local_result_file_name))

            session.sproc.register(
                func=pipeline_within_one_sproc,
                is_permanent=False,
                name=sproc_name,
                packages=required_deps,  # type: ignore[arg-type]
                replace=True,
                session=session,
                anonymous=True,
                imports=imports,  # type: ignore[arg-type]
                statement_params=sproc_statement_params,
            )

            sproc_export_file_name: str = pipeline_within_one_sproc(
                session,
                sql_queries,
                stage_estimator_file_name,
                stage_result_file_name,
                sproc_statement_params,
            )

            local_result_file_name = temp_file_utils.get_temp_file_path()
            session.file.get(
                posixpath.join(stage_estimator_file_name, sproc_export_file_name),
                local_result_file_name,
                statement_params=sproc_statement_params,
            )

            with open(os.path.join(local_result_file_name, sproc_export_file_name), mode="r+b") as result_file_obj:
                fit_estimator = cp.load(result_file_obj)

            temp_file_utils.cleanup_temp_files([local_result_file_name])
            for key, val in vars(fit_estimator).items():
                setattr(self, key, val)

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame], squash: Optional[bool] = False) -> "Pipeline":
        """
        Fit the entire pipeline using the dataset.

        Args:
            dataset: Input dataset.
            squash: Run the whole pipeline within a stored procedure

        Returns:
            Fitted pipeline.
        """

        self._validate_steps()
        dataset = (
            snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
            if isinstance(dataset, snowpark.DataFrame)
            else dataset
        )

        # Extract lineage information here since we're overriding fit() directly
        data_sources = lineage_utils.get_data_sources(dataset)
        if not data_sources and isinstance(dataset, snowpark.DataFrame):
            data_sources = [data_source.DataFrameInfo(dataset.queries["queries"][-1])]
        lineage_utils.set_data_sources(self, data_sources)

        if squash and isinstance(dataset, snowpark.DataFrame):
            session = dataset._session
            assert session is not None
            self._fit_snowpark_dataframe_within_one_sproc(session=session, dataset=dataset)

        else:
            transformed_dataset = self._fit_transform_dataset(dataset)

            estimator = self._get_estimator()
            if estimator:
                all_cols = transformed_dataset.columns[:]
                estimator[1].fit(transformed_dataset)

                self._append_step_feature_consumption_info(
                    step_name=estimator[0], all_cols=all_cols, input_cols=estimator[1].get_input_cols()
                )

            self._generate_model_signatures(dataset=dataset)

        self._is_fitted = True

        return self

    @metaestimators.available_if(_final_step_has("transform"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def transform(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Call `transform` of each transformer in the pipeline.

        Args:
            dataset: Input dataset.

        Returns:
            Transformed data. Output datatype will be same as input datatype.
        """
        self._enforce_fit()
        dataset = (
            snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
            if isinstance(dataset, snowpark.DataFrame)
            else dataset
        )

        if self._sklearn_object is not None:
            handler = ModelTransformerBuilder.build(
                dataset=dataset,
                estimator=self._sklearn_object,
                class_name="Pipeline",
                subproject="",
                autogenerated=False,
            )
            return handler.batch_inference(
                inference_method="transform",
                input_cols=self.input_cols if self.input_cols else self._infer_input_cols(dataset),
                expected_output_cols=self._infer_output_cols(),
                session=dataset._session,
                dependencies=self._deps,
            )

        transformed_dataset = self._transform_dataset(dataset=dataset)
        estimator = self._get_estimator()
        if estimator:
            return estimator[1].transform(transformed_dataset)
        return transformed_dataset

    def _final_step_can_fit_transform(self) -> bool:
        return has_callable_attr(self.steps[-1][1], "fit_transform") or (
            has_callable_attr(self.steps[-1][1], "fit") and has_callable_attr(self.steps[-1][1], "transform")
        )

    @metaestimators.available_if(_final_step_can_fit_transform)  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit_transform(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Fits all the transformer objs one after another and transforms the data. Then fits and transforms data using the
        estimator. This will only be available if the estimator (or final step) has fit_transform or transform
        methods.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """

        self._validate_steps()
        dataset = (
            snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
            if isinstance(dataset, snowpark.DataFrame)
            else dataset
        )

        transformed_dataset = self._fit_transform_dataset(dataset=dataset)

        estimator = self._get_estimator()
        if estimator:
            if has_callable_attr(estimator[1], "fit_transform"):
                res: snowpark.DataFrame = estimator[1].fit_transform(transformed_dataset)
            else:
                res = estimator[1].fit(transformed_dataset).transform(transformed_dataset)
            return res

        self._generate_model_signatures(dataset=dataset)
        self._is_fitted = True
        return transformed_dataset

    def _final_step_can_fit_predict(self) -> bool:
        return has_callable_attr(self.steps[-1][1], "fit_predict") or (
            has_callable_attr(self.steps[-1][1], "fit") and has_callable_attr(self.steps[-1][1], "predict")
        )

    @metaestimators.available_if(_final_step_can_fit_predict)  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit_predict(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Fits all the transformer objs one after another and transforms the data. Then fits and predicts using the
        estimator. This will only be available if the estimator (or final step) has fit_predict or predict
        methods.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """

        self._validate_steps()
        dataset = (
            snowpark_dataframe_utils.cast_snowpark_dataframe_column_types(dataset)
            if isinstance(dataset, snowpark.DataFrame)
            else dataset
        )

        transformed_dataset = self._fit_transform_dataset(dataset=dataset)

        estimator = self._get_estimator()
        if estimator:
            if has_callable_attr(estimator[1], "fit_predict"):
                transformed_dataset = estimator[1].fit_predict(transformed_dataset)
            else:
                transformed_dataset = estimator[1].fit(transformed_dataset).predict(transformed_dataset)

        self._generate_model_signatures(dataset=dataset)
        self._is_fitted = True
        return transformed_dataset

    @metaestimators.available_if(_final_step_has("predict"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and predict using the estimator.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        return self._invoke_estimator_func("predict", dataset)

    @metaestimators.available_if(_final_step_has("score_samples"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def score_samples(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and predict using the estimator.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        return self._invoke_estimator_func("score_samples", dataset)

    @metaestimators.available_if(_final_step_has("predict_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict_proba(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and apply `predict_proba` using the estimator.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        return self._invoke_estimator_func("predict_proba", dataset)

    @metaestimators.available_if(_final_step_has("predict_log_proba"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def predict_log_proba(
        self, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and apply `predict_log_proba` using the
        estimator.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """
        return self._invoke_estimator_func("predict_log_proba", dataset)

    @metaestimators.available_if(_final_step_has("score"))  # type: ignore[misc]
    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def score(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and apply `score` using the estimator.

        Args:
            dataset: Input dataset.

        Returns:
            Output dataset.
        """

        return self._invoke_estimator_func("score", dataset)

    def _invoke_estimator_func(
        self, func_name: str, dataset: Union[snowpark.DataFrame, pd.DataFrame]
    ) -> Union[snowpark.DataFrame, pd.DataFrame]:
        """
        Transform the dataset by applying all the transformers in order and apply specified estimator function.

        Args:
            func_name: Target function name.
            dataset: Input dataset.

        Raises:
            SnowflakeMLException: If the pipeline is not fitted first.

        Returns:
            Output dataset.
        """
        if not self._is_fitted:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.METHOD_NOT_ALLOWED,
                original_exception=RuntimeError(f"Pipeline is not fitted before calling {func_name}()."),
            )

        transformed_dataset = self._transform_dataset(dataset=dataset)
        estimator = self._get_estimator()
        assert estimator is not None
        res: snowpark.DataFrame = getattr(estimator[1], func_name)(transformed_dataset)
        return res

    def _construct_fitted_column_transformer_object(
        self,
        step_name_in_pipeline: str,
        step_index_in_pipeline: int,
        step_name_in_ct: str,
        step_transformer_obj: Any,
        remainder_action: str,
    ) -> ColumnTransformer:
        """
        Constructs a fitted column transformer object with one step.

        Args:
            step_name_in_pipeline: Name of the step in original pipeline.
            step_index_in_pipeline: Index of the step in the original pipeline.
            step_name_in_ct: Name of the step in column transformer.
            step_transformer_obj: SKLearn object for the transformer or "passthrough".
            remainder_action: Action to take on the remainder of input. Possible options "drop" or "passthrough".

        Returns:
            Returns a fitted column transformer object.
        """
        input_col_indices = self._transformers_to_input_indices[step_name_in_pipeline]
        ct = ColumnTransformer(
            transformers=[(step_name_in_ct, step_transformer_obj, input_col_indices)], remainder="passthrough"
        )
        if step_index_in_pipeline == 0:
            # Add column name check for only first transformer. Everything else works with ndarrays as input.
            ct.feature_names_in_ = self._feature_names_in[step_index_in_pipeline]
        ct.n_features_in_ = self._n_features_in[step_index_in_pipeline]
        ct._columns = [input_col_indices]
        ct._n_features = self._n_features_in[step_index_in_pipeline]
        remaining = sorted(set(range(self._n_features_in[step_index_in_pipeline])) - set(input_col_indices))
        ct._remainder = ("remainder", remainder_action, remaining)
        ct._transformer_to_input_indices = {step_name_in_ct: input_col_indices, "remainder": remaining}
        ct.transformers_ = [
            (step_name_in_ct, step_transformer_obj, input_col_indices),
            ("remainder", remainder_action, remaining),
        ]
        ct.sparse_output_ = False

        # ColumnTransformer internally replaces the "passthrough" string in the "remainder" step with a
        # fitted FunctionTransformer during the fit() call. So we need to manually replace the "passthrough"
        # string with a fitted FunctionTransformer
        for i, (step, transform, indices) in enumerate(ct.transformers_):
            if transform == "passthrough":
                ft = FunctionTransformer(
                    accept_sparse=True,
                    check_inverse=False,
                    feature_names_out="one-to-one",
                )
                if step == "remainder":
                    ft.feature_names_in_ = remaining
                    ft.n_features_in_ = len(remaining)
                else:
                    ft.feature_names_in_ = self._feature_names_in[step_index_in_pipeline]
                    ft.n_features_in_ = self._n_features_in[step_index_in_pipeline]
                ct.transformers_[i] = (step, ft, indices)

        return ct

    def _get_label_cols(self) -> list[str]:
        """Util function to get the label columns from the pipeline.
        The label column is only present in the estimator

        Returns:
            List of label columns, or empty list if no label cols.
        """
        label_cols = []
        estimator = self._get_estimator()
        if estimator is not None:
            label_cols = estimator[1].get_label_cols()

        return label_cols

    @staticmethod
    def _wrap_transformer_in_column_transformer(
        transformer_name: str, transformer: base.BaseTransformer
    ) -> ColumnTransformer:
        """A helper function to convert a transformer object to an sklearn object and wrap in an sklearn
            ColumnTransformer.

        Args:
            transformer_name: Name of the transformer to be wrapped.
            transformer: The transformer object to be wrapped.

        Returns:
            A column transformer sklearn object that uses the input columns from the initial snowpark ml transformer.
        """
        column_transformer = ColumnTransformer(
            transformers=[(transformer_name, Pipeline._get_native_object(transformer), transformer.get_input_cols())],
            remainder="passthrough",
        )
        return column_transformer

    def _create_unfitted_sklearn_object(self) -> pipeline.Pipeline:
        """Create a sklearn pipeline from the current snowml pipeline.
        ColumnTransformers are used to wrap transformers as their input columns can be specified
        as a subset of the pipeline's input columns.

        Returns:
            An unfit pipeline that can be fit using the ML runtime client.
        """

        sklearn_pipeline_steps = []

        first_step_name, first_step_object = self.steps[0]

        # Only the first step can have the input_cols field not None/empty.
        if first_step_object.get_input_cols():
            first_step_column_transformer = Pipeline._wrap_transformer_in_column_transformer(
                first_step_name, first_step_object
            )
            first_step_skl = (first_step_name, first_step_column_transformer)
        else:
            first_step_skl = (first_step_name, Pipeline._get_native_object(first_step_object))

        sklearn_pipeline_steps.append(first_step_skl)

        for step_name, step_object in self.steps[1:]:
            skl_step = (step_name, Pipeline._get_native_object(step_object))
            sklearn_pipeline_steps.append(skl_step)

        return pipeline.Pipeline(sklearn_pipeline_steps)

    def _create_sklearn_object(self) -> pipeline.Pipeline:
        if not self._is_fitted:
            return self._create_unfitted_sklearn_object()

        if not self._modifies_label_or_sample_weight:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.METHOD_NOT_ALLOWED,
                original_exception=ValueError(
                    "The pipeline can't be converted to SKLearn equivalent because it modifies processing label or "
                    "sample_weight columns as part of pipeline preprocessing steps which is not allowed in SKLearn."
                ),
            )

        # Create a fitted sklearn pipeline object by translating each non-estimator step in pipeline with with
        # a fitted column transformer.
        sksteps = []
        i = 0
        for i, (name, trans) in enumerate(self._get_transformers()):
            if isinstance(trans, base.BaseTransformer):
                trans = self._construct_fitted_column_transformer_object(
                    step_name_in_pipeline=name,
                    step_index_in_pipeline=i,
                    step_name_in_ct=name,
                    step_transformer_obj=_utils.to_native_format(trans),
                    remainder_action="passthrough",
                )

            sksteps.append(tuple([name, trans]))

        estimator_step = self._get_estimator()
        if estimator_step:
            if isinstance(estimator_step[1], base.BaseTransformer):
                ct = self._construct_fitted_column_transformer_object(
                    step_name_in_pipeline=estimator_step[0],
                    step_index_in_pipeline=i,
                    step_name_in_ct="filter_input_cols_for_estimator",
                    step_transformer_obj="passthrough",
                    remainder_action="drop",
                )

                sksteps.append(tuple(["filter_input_cols_for_estimator", ct]))
                sksteps.append(tuple([estimator_step[0], _utils.to_native_format(estimator_step[1])]))
            else:
                sksteps.append(estimator_step)

        return pipeline.Pipeline(steps=sksteps)

    def _get_dependencies(self) -> list[str]:
        return self._deps

    def _generate_model_signatures(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> None:
        self._model_signature_dict = dict()

        input_columns = self._get_sanitized_list_of_columns(dataset.columns)
        inputs_signature = _infer_signature(
            _truncate_data(dataset[input_columns], INFER_SIGNATURE_MAX_ROWS), "input", use_snowflake_identifiers=True
        )

        estimator_step = self._get_estimator()
        if estimator_step:
            estimator_signatures = estimator_step[1].model_signatures
            for method, signature in estimator_signatures.items():
                # Add the inferred input signature to the model signature dictionary for each method
                self._model_signature_dict[method] = ModelSignature(
                    inputs=inputs_signature,
                    outputs=(
                        # If _drop_input_cols is True, do not include any input columns in the output signature
                        []
                        if self._drop_input_cols
                        else [
                            # Include input columns in the output signature if they are not already present
                            # Those already present means they are overwritten by the output of the estimator
                            spec
                            for spec in inputs_signature
                            if spec.name not in [_spec.name for _spec in signature.outputs]
                        ]
                    )
                    + signature.outputs,  # Append the existing output signature
                )

    @property
    def model_signatures(self) -> dict[str, ModelSignature]:
        if self._model_signature_dict is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Estimator not fitted before accessing property model_signatures!"),
            )
        return self._model_signature_dict

    @staticmethod
    def _get_native_object(estimator: base.BaseEstimator) -> object:
        """A helper function to get the native(sklearn, xgboost, or lightgbm)
        object from a snowpark ml estimator.
        TODO - better type hinting - is there a common base class for all xgb/lgbm estimators?

        Args:
            estimator: the estimator from which to derive the native object.

        Returns:
            a native estimator object

        Raises:
            ValueError: The estimator is not an sklearn, xgboost, or lightgbm estimator.
        """
        methods = ["to_sklearn", "to_xgboost", "to_lightgbm"]
        for method_name in methods:
            if hasattr(estimator, method_name):
                try:
                    result = getattr(estimator, method_name)()
                    return result
                except exceptions.SnowflakeMLException:
                    pass  # Do nothing and continue to the next method
        raise ValueError("The estimator must be an sklearn, xgboost, or lightgbm estimator.")

    def to_sklearn(self) -> pipeline.Pipeline:
        """Returns an sklearn Pipeline representing the object, if possible.

        Returns:
            previously fit sklearn Pipeline if present, else an unfit pipeline

        Raises:
            ValueError: The pipeline cannot be represented as an sklearn pipeline.
        """
        if self._is_fitted:
            if self._sklearn_object is not None:
                return self._sklearn_object
            else:
                return self._create_sklearn_object()
        else:
            if self._is_convertible_to_sklearn:
                return self._create_unfitted_sklearn_object()
            else:
                raise ValueError("This pipeline can not be converted to an sklearn pipeline.")

    def _send_pipeline_configuration_telemetry(self) -> None:
        """Track information about the pipeline setup. Currently, we want to track:
        - Whether the pipeline is converible to an sklearn pipeline
        - Whether the pipeline is being used in the SPCS ml runtime.
        """

        telemetry_data = {
            "pipeline_is_convertible_to_sklearn": self._is_convertible_to_sklearn,
        }
        telemetry.send_custom_usage(
            project=_PROJECT,
            subproject=_SUBPROJECT,
            telemetry_type=telemetry.TelemetryField.TYPE_SNOWML_PIPELINE_USAGE.value,
            data=telemetry_data,
        )
