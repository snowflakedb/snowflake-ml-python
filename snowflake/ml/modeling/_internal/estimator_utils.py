import inspect
import numbers
import os
from typing import Any, Callable

import cloudpickle as cp
import numpy as np
from numpy import typing as npt

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml._internal.utils import temp_file_utils
from snowflake.ml._internal.utils.query_result_checker import SqlResultValidator
from snowflake.ml.modeling.framework._utils import to_native_format
from snowflake.ml.modeling.framework.base import BaseTransformer
from snowflake.snowpark import Session
from snowflake.snowpark._internal import utils as snowpark_utils


def validate_sklearn_args(args: dict[str, tuple[Any, Any, bool]], klass: type) -> dict[str, Any]:
    """Validate if all the keyword args are supported by current version of SKLearn/XGBoost object.

    Args:
        args: Dictionary with kwarg as key. Values is a list with three entries: the kwarg value, default value, and
              whether default is included in signature.
        klass: Underlying SKLearn/XGBoost class object.

    Returns:
        result: sklearn arguments

    Raises:
        SnowflakeMLException: if a user specified arg is not supported by current version of sklearn/xgboost.
    """
    result = {}
    signature = inspect.signature(klass.__init__)  # type: ignore[misc]
    for k, v in args.items():
        if k not in signature.parameters.keys():  # Arg is not supported.
            if v[2] or (  # Arg doesn't have default value in the signature.
                v[0] != v[1]  # Value is not same as default.
                and not (isinstance(v[0], float) and np.isnan(v[0]) and np.isnan(v[1]))
            ):  # both are not NANs
                raise exceptions.SnowflakeMLException(
                    error_code=error_codes.DEPENDENCY_VERSION_ERROR,
                    original_exception=RuntimeError(f"Arg {k} is not supported by current version of SKLearn/XGBoost."),
                )
        elif v[0] == v[1] and v[0] != signature.parameters[k].default:
            # If default value (pulled at autogen time) is not the same as the installed library's default value,
            # we need to validate the parameter value against the parameter constraints.
            # If the parameter value is invalid, we drop it.
            try:
                from sklearn.utils._param_validation import (
                    InvalidParameterError,
                    validate_parameter_constraints,
                )

                try:
                    validate_parameter_constraints(
                        klass._parameter_constraints,  # type: ignore[attr-defined]
                        {k: v[0]},
                        klass.__name__,
                    )
                except InvalidParameterError:
                    continue  # Let the underlying estimator fill in the default value.
            except (ImportError, AttributeError, TypeError):
                result[k] = v[0]  # Try to use the value as is.
        else:
            result[k] = v[0]
    return result


def transform_snowml_obj_to_sklearn_obj(obj: Any) -> Any:
    """Converts SnowML Estimator and Transformer objects to equivalent SKLearn objects.

    Args:
        obj: Source object that needs to be converted. Source object could of any type, example, lists, tuples, etc.

    Returns:
        An equivalent object with SnowML estimators and transforms replaced with equivalent SKLearn objects.
    """

    if isinstance(obj, list):
        # Apply transform function to each element in the list
        return list(map(transform_snowml_obj_to_sklearn_obj, obj))
    elif isinstance(obj, tuple):
        # Apply transform function to each element in the tuple
        return tuple(map(transform_snowml_obj_to_sklearn_obj, obj))
    elif isinstance(obj, BaseTransformer):
        # Convert SnowML object to equivalent SKLearn object
        return to_native_format(obj)
    else:
        # Return all other objects as it is.
        return obj


def gather_dependencies(obj: Any) -> set[str]:
    """Gathers dependencies from the SnowML Estimator and Transformer objects.

    Args:
        obj: Source object to collect dependencies from. Source object could of any type, example, lists, tuples, etc.

    Returns:
        A set of dependencies required to work with the object.
    """

    if isinstance(obj, list) or isinstance(obj, tuple):
        deps: set[str] = set()
        for elem in obj:
            deps = deps | set(gather_dependencies(elem))
        return deps
    elif isinstance(obj, BaseTransformer):
        return set(obj._get_dependencies())
    else:
        return set()


def original_estimator_has_callable(attr: str) -> Callable[[Any], bool]:
    """Checks that the original estimator has callable `attr`.

    Args:
        attr: Attribute to check for.

    Returns:
        A function which checks for the existence of callable `attr` on the given object.
    """
    from typing_extensions import TypeGuard

    def check(self: BaseTransformer) -> TypeGuard[Callable[..., object]]:
        """Check for the existence of callable `attr` in self.

        Args:
            self: BaseTransformer object

        Returns:
            True of the callable `attr` exists in self, False otherwise.
        """
        return callable(getattr(self._sklearn_object, attr, None))

    return check


def is_single_node(session: Session) -> bool:
    """Retrieve the current session's warehouse type and warehouse size, and depends on those information
    to identify if it is single node or not

    Args:
        session (Session): session object that is used by user currently

    Returns:
        bool: single node or not. True stands for yes.
    """
    warehouse_name = session.get_current_warehouse()
    if warehouse_name:
        warehouse_name = warehouse_name.replace('"', "")
        df_list = session.sql(f"SHOW WAREHOUSES like '{warehouse_name}';")['"type"', '"size"'].collect()
        # If no warehouse data is found, default to True (single node)
        if not df_list:
            return True
        df = df_list[0]
        # filter out the conditions when it is single node
        single_node: bool = (df[0] == "SNOWPARK-OPTIMIZED" and df[1] == "Medium") or (
            df[0] == "STANDARD" and df[1] == "X-Small"
        )
        return single_node
    # If current session cannot retrieve the warehouse name back,
    # Default as True; Let HPO fall back to stored procedure implementation
    return True


def get_module_name(model: object) -> str:
    """Returns the source module of the given object.

    Args:
        model: Object to inspect.

    Returns:
        Source module of the given object.

    Raises:
        SnowflakeMLException: If the source module of the given object is not found.
    """
    module = inspect.getmodule(model)
    if module is None:
        raise exceptions.SnowflakeMLException(
            error_code=error_codes.INVALID_TYPE,
            original_exception=ValueError(f"Unable to infer the source module of the given object {model}."),
        )
    return module.__name__


def handle_inference_result(
    inference_res: Any, output_cols: list[str], inference_method: str, within_udf: bool = False
) -> tuple[npt.NDArray[Any], list[str]]:
    if isinstance(inference_res, list) and len(inference_res) > 0 and isinstance(inference_res[0], np.ndarray):
        # In case of multioutput estimators, predict_proba, decision_function etc., functions return a list of
        # ndarrays. We need to concatenate them.

        # First compute output column names
        if len(output_cols) == len(inference_res):
            actual_output_cols = []
            for idx, np_arr in enumerate(inference_res):
                for i in range(1 if len(np_arr.shape) <= 1 else np_arr.shape[1]):
                    actual_output_cols.append(f"{output_cols[idx]}_{i}")
            output_cols = actual_output_cols

        # Concatenate np arrays
        transformed_numpy_array = np.concatenate(inference_res, axis=1)
    elif isinstance(inference_res, tuple) and len(inference_res) > 0 and isinstance(inference_res[0], np.ndarray):
        # In case of kneighbors, functions return a tuple of ndarrays.
        transformed_numpy_array = np.stack(inference_res, axis=1)
    elif isinstance(inference_res, numbers.Number):
        # In case of BernoulliRBM, functions return a float
        transformed_numpy_array = np.array([inference_res])
    else:
        transformed_numpy_array = inference_res

    if (len(transformed_numpy_array.shape) == 3) and inference_method != "kneighbors":
        # VotingClassifier will return results of shape (n_classifiers, n_samples, n_classes)
        # when voting = "soft" and flatten_transform = False. We can't handle unflatten transforms,
        # so we ignore flatten_transform flag and flatten the results.
        transformed_numpy_array = np.hstack(transformed_numpy_array)  # type: ignore[call-overload]

    if len(transformed_numpy_array.shape) == 1:
        # Within a vectorized UDF, a single-row batch often yields a 1D array of length n_components.
        # That must be reshaped to (1, n_components) to keep the number of rows aligned with the input batch.
        if len(output_cols) > 1:
            transformed_numpy_array = np.reshape(transformed_numpy_array, (1, -1))
        else:
            transformed_numpy_array = np.reshape(transformed_numpy_array, (-1, 1))

    shape = transformed_numpy_array.shape
    if len(shape) > 1:
        if shape[1] != len(output_cols):
            # Within UDF, it is not feasible to change the output cols because we need to
            # query the output cols after UDF by the expected output cols
            if not within_udf:
                # The following lines are to generate the output cols to match the length of
                # transformed_numpy_array
                actual_output_cols = []
                for i in range(shape[1]):
                    actual_output_cols.append(f"{output_cols[0]}_{i}")
                output_cols = actual_output_cols
            else:
                # HeterogeneousEnsemble's transform method produce results with varying shapes
                # from (n_samples, n_estimators) to (n_samples, n_estimators * n_classes).
                # It is hard to predict the response shape without using fragile introspection logic.
                # So, to avoid that we are packing the results into a dataframe of shape (n_samples, 1) with
                # each element being a list.
                if len(output_cols) != 1:
                    raise TypeError(
                        "expected_output_cols must be same length as transformed array or should be of length 1."
                        f"Currently expected_output_cols shape is {len(output_cols)}, "
                        f"transformed array shape is {shape}. "
                    )

    return transformed_numpy_array, output_cols


def create_temp_stage(session: Session) -> str:
    """Creates temporary stage.

    Args:
        session: Session

    Returns:
        Temp stage name.
    """
    # Create temp stage to upload pickled model file.
    transform_stage_name = snowpark_utils.random_name_for_temp_object(snowpark_utils.TempObjectType.STAGE)
    stage_creation_query = f"CREATE OR REPLACE TEMPORARY STAGE {transform_stage_name};"
    SqlResultValidator(session=session, query=stage_creation_query).has_dimensions(
        expected_rows=1, expected_cols=1
    ).validate()
    return transform_stage_name


def upload_model_to_stage(
    stage_name: str, estimator: object, session: Session, statement_params: dict[str, str]
) -> str:
    """Util method to pickle and upload the model to a temp Snowflake stage.


    Args:
        stage_name: Stage name to save model.
        estimator: Estimator object to upload to stage (sklearn model object)
        session: The snowpark session to use.
        statement_params: Statement parameters for query telemetry.

    Returns:
        a tuple containing stage file paths for pickled input model for training and location to store trained
        models(response from training sproc).
    """
    # Create a temp file and dump the transform to that file.
    local_transform_file_name = temp_file_utils.get_temp_file_path()
    with open(local_transform_file_name, mode="w+b") as local_transform_file:
        cp.dump(estimator, local_transform_file)

    # Put locally serialized transform on stage.
    session.file.put(
        local_file_name=local_transform_file_name,
        stage_location=stage_name,
        auto_compress=False,
        overwrite=True,
        statement_params=statement_params,
    )

    temp_file_utils.cleanup_temp_files([local_transform_file_name])
    return os.path.basename(local_transform_file_name)


def should_include_sample_weight(estimator: object, method_name: str) -> bool:
    # If this is a Grid Search or Randomized Search estimator, check the underlying estimator.
    underlying_estimator = (
        estimator.estimator if ("_search" in estimator.__module__ and hasattr(estimator, "estimator")) else estimator
    )
    method = getattr(underlying_estimator, method_name)
    underlying_estimator_params = inspect.signature(method).parameters
    if "sample_weight" in underlying_estimator_params:
        return True

    return False


def is_multi_task_estimator(estimator: object) -> bool:
    """
    Check if the estimator is a multi-task estimator that requires 2D targets.

    Args:
        estimator: The estimator to check

    Returns:
        True if the estimator is a multi-task estimator, False otherwise
    """
    # List of known multi-task estimators that require 2D targets
    multi_task_estimators = {"MultiTaskElasticNet", "MultiTaskElasticNetCV", "MultiTaskLasso", "MultiTaskLassoCV"}

    estimator_name = estimator.__class__.__name__
    return estimator_name in multi_task_estimators
