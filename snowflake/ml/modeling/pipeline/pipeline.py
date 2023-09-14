#!/usr/bin/env python3
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import __version__ as skversion, pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import metaestimators

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.model.model_signature import ModelSignature, _infer_signature
from snowflake.ml.modeling.framework import _utils, base

_PROJECT = "ModelDevelopment"
_SUBPROJECT = "Framework"


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


def _get_column_indices(all_columns: List[str], target_columns: List[str]) -> List[int]:
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
    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        """
        Pipeline of transforms.

        Sequentially apply a list of transforms.
        Intermediate steps of the pipeline must be 'transforms', that is, they
        must implement `fit` and `transform` methods.
        The final step can be a transform or estimator, that is, it must implement
        `fit` and `transform`/`predict` methods.
        TODO: SKLearn pipeline expects last step(and only the last step) to be an estimator obj or a dummy
                estimator(like None or passthrough). Currently this Pipeline class works with a list of all
                transforms or a list of transforms ending with an estimator. Should we change this implementation
                to only work with list of steps ending with an estimator or a dummy estimator like SKLearn?

        Args:
            steps: List of (name, transform) tuples (implementing `fit`/`transform`) that
                are chained in sequential order. The last transform can be an
                estimator.
        """
        super().__init__()
        self.steps = steps
        self._is_final_step_estimator = Pipeline._is_estimator(steps[-1][1])
        self._is_fitted = False
        self._feature_names_in: List[np.ndarray[Any, np.dtype[Any]]] = []
        self._n_features_in: List[int] = []
        self._transformers_to_input_indices: Dict[str, List[int]] = {}
        self._is_convertible_to_sklearn = True

        self._model_signature_dict: Optional[Dict[str, ModelSignature]] = None

        deps: Set[str] = {f"pandas=={pd.__version__}", f"scikit-learn=={skversion}"}
        for _, obj in steps:
            if isinstance(obj, base.BaseTransformer):
                deps = deps | set(obj._get_dependencies())
        self._deps = list(deps)

    @staticmethod
    def _is_estimator(obj: object) -> bool:
        # TODO(SNOW-723770): Figure out a better way to identify estimator objects.
        return has_callable_attr(obj, "fit") and has_callable_attr(obj, "predict")

    @staticmethod
    def _is_transformer(obj: object) -> bool:
        return has_callable_attr(obj, "fit") and has_callable_attr(obj, "transform")

    def _get_transformers(self) -> List[Tuple[str, Any]]:
        return self.steps[:-1] if self._is_final_step_estimator else self.steps

    def _get_estimator(self) -> Optional[Tuple[str, Any]]:
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

    def _get_sanitized_list_of_columns(self, columns: List[str]) -> List[str]:
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

    def _append_step_feature_consumption_info(self, step_name: str, all_cols: List[str], input_cols: List[str]) -> None:
        if self._is_convertible_to_sklearn:
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
        self._is_convertible_to_sklearn = not self._is_pipeline_modifying_label_or_sample_weight()
        transformed_dataset = dataset
        for name, trans in self._get_transformers():
            self._append_step_feature_consumption_info(
                step_name=name, all_cols=transformed_dataset.columns[:], input_cols=trans.get_input_cols()
            )
            if has_callable_attr(trans, "fit_transform"):
                transformed_dataset = trans.fit_transform(transformed_dataset)
            else:
                trans.fit(transformed_dataset)
                transformed_dataset = trans.transform(transformed_dataset)

        return transformed_dataset

    @telemetry.send_api_usage_telemetry(
        project=_PROJECT,
        subproject=_SUBPROJECT,
    )
    def fit(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> "Pipeline":
        """
        Fit the entire pipeline using the dataset.

        Args:
            dataset: Input dataset.

        Returns:
            Fitted pipeline.
        """

        self._validate_steps()
        transformed_dataset = self._fit_transform_dataset(dataset)

        estimator = self._get_estimator()
        if estimator:
            all_cols = transformed_dataset.columns[:]
            estimator[1].fit(transformed_dataset)

            self._append_step_feature_consumption_info(
                step_name=estimator[0], all_cols=all_cols, input_cols=estimator[1].get_input_cols()
            )

        self._get_model_signatures(dataset=dataset)
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

        transformed_dataset = self._fit_transform_dataset(dataset=dataset)

        estimator = self._get_estimator()
        if estimator:
            if has_callable_attr(estimator[1], "fit_transform"):
                res: snowpark.DataFrame = estimator[1].fit_transform(transformed_dataset)
            else:
                res = estimator[1].fit(transformed_dataset).transform(transformed_dataset)
            return res

        self._get_model_signatures(dataset=dataset)
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

        transformed_dataset = self._fit_transform_dataset(dataset=dataset)

        estimator = self._get_estimator()
        if estimator:
            if has_callable_attr(estimator[1], "fit_predict"):
                transformed_dataset = estimator[1].fit_predict(transformed_dataset)
            else:
                transformed_dataset = estimator[1].fit(transformed_dataset).predict(transformed_dataset)

        self._get_model_signatures(dataset=dataset)
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

    def _create_unfitted_sklearn_object(self) -> pipeline.Pipeline:
        sksteps = []
        for step in self.steps:
            if isinstance(step[1], base.BaseTransformer):
                sksteps.append(tuple([step[0], _utils.to_native_format(step[1])]))
            else:
                sksteps.append(tuple([step[0], step[1]]))
        return pipeline.Pipeline(steps=sksteps)

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
        # fitted FunctionTransformer, saved in the _name_to_fitted_passthrough dict, during the transform()
        # call. So we need to populate _name_to_fitted_passthrough dict with fitted FunctionTransformer so
        # that the replacements works correctly during the transform() call.
        ft = FunctionTransformer(
            accept_sparse=True,
            check_inverse=False,
            feature_names_out="one-to-one",
        )

        if remainder_action == "passthrough":
            ft.n_features_in_ = len(remaining)
            ct._name_to_fitted_passthrough = {"remainder": ft}
        elif step_transformer_obj == "passthrough":
            ft.n_features_in_ = self._n_features_in[step_index_in_pipeline]
            ct._name_to_fitted_passthrough = {step_name_in_ct: ft}
        return ct

    def _create_sklearn_object(self) -> pipeline.Pipeline:
        if not self._is_fitted:
            return self._create_unfitted_sklearn_object()

        if not self._is_convertible_to_sklearn:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.METHOD_NOT_ALLOWED,
                original_exception=ValueError(
                    "The pipeline can't be converted to SKLearn equivalent because it processing label or "
                    "sample_weight columns as part of pipeline preprocessing steps which is not allowed in SKLearn."
                ),
            )

        # Create a fitted sklearn pipeline object by translating each non-estimator step in pipeline with with
        # a fitted column transformer.
        sksteps = []
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

    def _get_dependencies(self) -> List[str]:
        return self._deps

    def _get_model_signatures(self, dataset: Union[snowpark.DataFrame, pd.DataFrame]) -> None:
        self._model_signature_dict = dict()

        input_columns = self._get_sanitized_list_of_columns(dataset.columns)
        inputs_signature = _infer_signature(dataset[input_columns], "input")

        estimator_step = self._get_estimator()
        if estimator_step:
            estimator_signatures = estimator_step[1].model_signatures
            for method, signature in estimator_signatures.items():
                self._model_signature_dict[method] = ModelSignature(inputs=inputs_signature, outputs=signature.outputs)

    @property
    def model_signatures(self) -> Dict[str, ModelSignature]:
        if self._model_signature_dict is None:
            raise exceptions.SnowflakeMLException(
                error_code=error_codes.INVALID_ATTRIBUTE,
                original_exception=RuntimeError("Estimator not fitted before accessing property model_signatures!"),
            )
        return self._model_signature_dict
