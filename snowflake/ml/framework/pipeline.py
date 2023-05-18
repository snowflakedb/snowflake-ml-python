#!/usr/bin/env python3
#
# Copyright (c) 2012-2022 Snowflake Computing Inc. All rights reserved.
#
from typing import Any, Callable, List, Optional, Tuple, Union

import pandas as pd
from sklearn import pipeline
from sklearn.utils import metaestimators

from snowflake import snowpark
from snowflake.ml._internal import telemetry
from snowflake.ml.framework import base

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
                raise TypeError(
                    "All intermediate steps should be "
                    "transformers and implement both fit() and transform() methods, but"
                    f"{name} (type {type(t)}) doesn't."
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
        transformed_dataset = dataset
        for _, trans in self._get_transformers():
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
            estimator[1].fit(transformed_dataset)

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

        Raises:
            RuntimeError: If the pipeline is not fitted first.
        """

        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted before calling transform().")

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
            RuntimeError: If the pipeline is not fitted first.

        Returns:
            Output dataset.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Pipeline is not fitted before calling {func_name}().")

        transformed_dataset = self._transform_dataset(dataset=dataset)
        estimator = self._get_estimator()
        assert estimator is not None, "estimator cannot be None"
        res: snowpark.DataFrame = getattr(estimator[1], func_name)(transformed_dataset)
        return res

    def _create_unfitted_sklearn_object(self) -> pipeline.Pipeline:
        sksteps = []
        for step in self.steps:
            if isinstance(step[1], base.BaseTransformer):
                sksteps.append(tuple([step[0], step[1].get_sklearn_object()]))
            else:
                sksteps.append(tuple([step[0], step[1]]))
        return pipeline.Pipeline(steps=sksteps)

    def _create_sklearn_object(self) -> pipeline.Pipeline:
        return self._create_unfitted_sklearn_object()
