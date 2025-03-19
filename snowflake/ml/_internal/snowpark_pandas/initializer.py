import importlib
import inspect
import sys
from typing import Any, Callable, List, Optional

import cloudpickle as cp
import pandas as pd

from snowflake import snowpark
from snowflake.ml._internal.snowpark_pandas import imports, patch
from snowflake.ml._internal.utils import identifier, pkg_version_utils
from snowflake.snowpark import functions as F, types as T

# Import the modules and native modules
modules = imports.import_modules()
native_modules = imports.import_native_modules()
INDEX = "_INDEX"
TABLE_NAME = "_TABLE_NAME"

cp.register_pickle_by_value(inspect.getmodule(identifier.get_unescaped_names))


def _invoke_once(func: Callable[..., Any]) -> Callable[..., Any]:
    """Invoke once decorator.

    Args:
        func: The function to be invoked once. For usage, see the `init` function.

    Returns:
        The wrapper function.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not hasattr(func, "invoked"):
            func.invoked = False  # type: ignore[attr-defined]

        if func.invoked:  # type: ignore[attr-defined]
            return
        else:
            func.invoked = True  # type: ignore[attr-defined]
            return func(*args, **kwargs)

    return wrapper


@_invoke_once
def init(session: Optional[snowpark.Session] = None, enable_relax_version: Optional[bool] = True) -> None:
    """Initialize the snowpark_pandas module.

    Args:
        session: Snowpark session, optional. Defaults to None.
        enable_relax_version: Enable relax version for packages. Defaults to True.
            relax version control - only major_version.minor_version.* will be enforced.
            the result would be ordered by, the version that closest to user's version, and the latest.

    """
    for module_name in modules:
        module = sys.modules[module_name]

        for class_name, native_module_name in modules[module_name][imports.TYPE_CLASS]:
            klass = getattr(module, class_name)
            # Check if the class is a native class (impute_pkg, preprocessing_pkg)
            native_module = sys.modules[native_module_name] if native_module_name else None

            # Patch the fit method
            sproc_update_functions = ["fit"]
            for func in sproc_update_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    if native_module_name:
                        # Create a native patch for native classes
                        patch_func = patch.Patch.create_native_patch(
                            session, patch.PatchMode.UPDATE, getattr(klass, func), getattr(native_module, class_name)
                        )
                    else:
                        # Create a stored procedure patch for non-native classes
                        patch_func = patch.Patch.create_sproc_patch(
                            session, patch.PatchMode.UPDATE, getattr(klass, func)
                        )
                    setattr(klass, func, patch_func)

            # Patch the score method
            sproc_return_functions = ["score"]
            for func in sproc_return_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    if native_module_name:
                        patch_func = patch.Patch.create_native_patch(
                            session, patch.PatchMode.RETURN, getattr(klass, func), getattr(native_module, class_name)
                        )
                    else:
                        patch_func = patch.Patch.create_sproc_patch(
                            session, patch.PatchMode.RETURN, getattr(klass, func)
                        )
                    setattr(klass, func, patch_func)

            # Patch the batch_inference methods, which are predict, transform, predict_proba,
            # predict_log_proba, decision_function, using UDFs
            udf_functions = ["predict", "transform", "predict_proba", "predict_log_proba", "decision_function"]
            for func in udf_functions:
                if patch.Patch.is_patchable(klass=klass, method_name=func):
                    if native_module_name:
                        patch_func = patch.Patch.create_native_patch(
                            session,
                            patch.PatchMode.RETURN,
                            getattr(klass, func),
                            getattr(native_module, class_name),
                            is_inference=True,
                        )
                    else:
                        patch_func = patch.Patch.create_udf_patch(session, getattr(klass, func))
                    setattr(klass, func, patch_func)

        for function_name, native_module_name in modules[module_name][imports.TYPE_FUNCTION]:
            if patch.Patch.is_patchable(module=module, method_name=function_name):
                if native_module_name:
                    patch_func = patch.Patch.create_native_patch(
                        session, patch.PatchMode.RETURN, getattr(module, function_name), None, static=True
                    )
                else:
                    patch_func = patch.Patch.create_sproc_patch(
                        session, patch.PatchMode.RETURN, getattr(module, function_name), static=True
                    )
                setattr(module, function_name, patch_func)

    deps = _gather_deps()
    # Add pandas and modin to the dependencies, snowpark pandas requires pandas at least 2.2.1
    snowpark_pandas_dep = ["pandas==2.2.1", "modin==0.28.1"]

    if enable_relax_version and session is not None:
        assert session is not None
        relaxed_dependencies = pkg_version_utils.get_valid_pkg_versions_supported_in_snowflake_conda_channel(
            pkg_versions=deps + snowpark_pandas_dep, session=session
        )
    else:
        relaxed_dependencies = deps + snowpark_pandas_dep

    @F.sproc(  # type: ignore[misc]
        name=patch.PATCH_SPROC_NAME,
        replace=True,
        packages=relaxed_dependencies,  # type: ignore[arg-type]
        is_permanent=False,
    )
    def patch_sproc(
        session: snowpark.Session,
        model_file_location: str,
        method_data: bytes,
        pickled_args_data: bytes,
        pickled_kwargs_data: bytes,
    ) -> bytes:
        from snowflake.snowpark import files
        from snowflake.snowpark.modin import (  # type: ignore[attr-defined]
            pandas as SnowparkPandas,
        )

        def replace_snowflake_table_name_with_pandas(obj: Any) -> Any:
            if isinstance(obj, dict) and obj.get(TABLE_NAME, None) is not None:
                snowflake_table_name = obj.get(TABLE_NAME)
                sp_df = SnowparkPandas.read_snowflake(snowflake_table_name, index_col=INDEX)
                df = sp_df.to_pandas()
                df.columns = sp_df.columns
                return df
            # Covering all the iterable objects, including Lists, tuples, dictionaries, and sets
            if isinstance(obj, set):
                return {replace_snowflake_table_name_with_pandas(item) for item in obj}
            elif isinstance(obj, list):
                return [replace_snowflake_table_name_with_pandas(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: replace_snowflake_table_name_with_pandas(value) for key, value in obj.items()}
            elif isinstance(obj, tuple):
                return tuple(replace_snowflake_table_name_with_pandas(item) for item in obj)
            else:
                return obj

        def deserialize_snowpark_pandas_data_to_pandas_obj(data: bytes) -> Any:
            obj = cp.loads(data)
            return replace_snowflake_table_name_with_pandas(obj)

        with files.SnowflakeFile.open(model_file_location, mode="rb", require_scoped_url=False) as model_file:
            model = cp.load(model_file)
        method = cp.loads(method_data)
        args = deserialize_snowpark_pandas_data_to_pandas_obj(pickled_args_data)
        kwargs = deserialize_snowpark_pandas_data_to_pandas_obj(pickled_kwargs_data)
        return cp.dumps(getattr(model, method.__name__)(*args, **kwargs))  # type: ignore[no-any-return]

    @F.pandas_udf(  # type: ignore[arg-type, misc]
        name=patch.PATCH_UDF_NAME,
        replace=True,
        packages=relaxed_dependencies,  # type: ignore[arg-type]
        is_permanent=False,
    )
    def patch_udf(
        ds: T.PandasSeries[dict],  # type: ignore[type-arg]
        model_file_location: T.PandasSeries[str],
        method_name: T.PandasSeries[str],
        cols: T.PandasSeries[List[str]],
        args_data: T.PandasSeries[bytes],
        kwargs_data: T.PandasSeries[bytes],
    ) -> T.PandasSeries[dict]:  # type: ignore[type-arg]
        from snowflake.snowpark import files

        dataset = pd.json_normalize(ds)
        dataset.columns = identifier.get_unescaped_names(list(dataset.columns))
        dataset = dataset[cols[0]]
        with files.SnowflakeFile.open(model_file_location[0], mode="rb") as model_file:
            model = cp.load(model_file)
        args = cp.loads(args_data[0])
        kwargs = cp.loads(kwargs_data[0])
        res_arr = getattr(model, method_name[0])(dataset.drop(columns=[INDEX]), *args, **kwargs)

        # # reuse autogen
        # if isinstance(res_arr, list) and len(res_arr) > 0 and isinstance(res_arr[0], np.ndarray):
        #     res_arr = np.concatenate(res_arr, axis=1)
        #
        # if len(res_arr.shape) == 3:
        #     res_arr = np.hstack(res_arr)

        if len(res_arr.shape) > 1:
            series = pd.Series(res_arr.tolist())
            res_df = pd.DataFrame(series, columns=["RES"])
        else:
            res_df = pd.DataFrame(res_arr, columns=["RES"])
        res_df[INDEX] = dataset[INDEX]
        return res_df.to_dict("records")  # type: ignore[no-any-return]


def _gather_deps() -> List[str]:
    deps = [
        f"cloudpickle=={cp.__version__}",
        f"snowflake-snowpark-python=={snowpark.__version__}",
    ]

    # optional deps
    for module in ["xgboost", "lightgbm"]:
        if module in modules:
            deps.append(f"{module}=={sys.modules[module].__version__}")

    # scikit-learn is also optional, and is imported as 'sklearn'.
    if any("sklearn" in key for key in modules.keys()):
        sklearn = importlib.import_module("sklearn")
        deps.append(f"scikit-learn=={sklearn.__version__}")

    return deps
