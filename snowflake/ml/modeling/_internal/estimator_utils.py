import inspect
from typing import Any, Callable, Dict, Set, Tuple

import numpy as np
from typing_extensions import TypeGuard

from snowflake.ml._internal.exceptions import error_codes, exceptions
from snowflake.ml.modeling.framework._utils import to_native_format
from snowflake.ml.modeling.framework.base import BaseTransformer
from snowflake.snowpark import Session


def validate_sklearn_args(args: Dict[str, Tuple[Any, Any, bool]], klass: type) -> Dict[str, Any]:
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


def gather_dependencies(obj: Any) -> Set[str]:
    """Gathers dependencies from the SnowML Estimator and Transformer objects.

    Args:
        obj: Source object to collect dependencies from. Source object could of any type, example, lists, tuples, etc.

    Returns:
        A set of dependencies required to work with the object.
    """

    if isinstance(obj, list) or isinstance(obj, tuple):
        deps: Set[str] = set()
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
        df = session.sql(f"SHOW WAREHOUSES like '{warehouse_name}';")['"type"', '"size"'].collect()[0]
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
